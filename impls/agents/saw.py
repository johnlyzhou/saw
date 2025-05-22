import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Identity, LengthNormalize, GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue


class SAWAgent(flax.struct.PyTreeNode):
    """Subgoal Advantage-Weighted (SAW) Policy Bootstrapping agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @jax.jit
    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    @jax.jit
    def target_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss. Note that we use this to bootstrap a flat policy."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('low_actor')(batch['observations'], batch['low_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def actor_loss(self, batch, grad_params):
        """Compute regular actor loss w.r.t. high-actor goals."""
        # Compute standard 1-step AWR loss
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['awr_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        awr_loss = -(exp_a * log_prob).mean()

        awr_info = {
            'awr_loss': awr_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            awr_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return awr_loss, awr_info

    @jax.jit
    def waypoint_loss(self, batch, grad_params):
        """Compute the waypoint bootstrapping loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        wv1, wv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        wv = (wv1 + wv2) / 2
        wadv = wv - v

        exp_w = jnp.exp(wadv * self.config['kl_alpha'])
        exp_w = jnp.minimum(exp_w, 100.0)

        # Compute waypoint KL divergence term
        dist = self.network.select('actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        w_dist = self.network.select('low_actor')(batch['observations'], batch['high_actor_targets'])
        if self.config['const_std']:
            w_mode = jax.lax.stop_gradient(w_dist.mode())
            kld = jnp.sum((dist.mode() - w_mode) ** 2, axis=-1)
        else:
            kld = w_dist.kl_divergence(dist)
        waypoint_loss = (exp_w * kld).mean()  # Want to minimize this, no negative sign

        waypoint_info = {
            'waypoint_loss': waypoint_loss,
            'kld': kld.mean(),
            'wadv': wadv.mean(),
        }

        return waypoint_loss, waypoint_info

    @jax.jit
    def total_loss(self, batch, grad_params=None, rng=None):
        """Compute the total loss (only for val)."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.target_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        waypoint_loss, waypoint_info = self.waypoint_loss(batch, grad_params)
        for k, v in waypoint_info.items():
            info[f'waypoint/{k}'] = v

        loss = value_loss + low_actor_loss + actor_loss + waypoint_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        # This is default for the value function, similar to HIQL
        goal_rep_def = nn.Sequential(goal_rep_seq)
        actor_goal_rep_def = goal_rep_def if config['share_goal_rep'] else copy.deepcopy(goal_rep_def)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            encoders['target_value'] = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            encoders['actor'] = GCEncoder(state_encoder=encoder_module(), concat_encoder=actor_goal_rep_def)
            encoders['low_actor'] = GCEncoder(concat_encoder=encoder_module())
        else:
            encoders['value'] = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            encoders['target_value'] = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            encoders['actor'] = GCEncoder(state_encoder=Identity(), concat_encoder=actor_goal_rep_def)

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('value'),
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('target_value'),
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['low_actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('low_actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
            low_actor_def = GCActor(
                hidden_dims=config['low_actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('low_actor'),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            goal_rep=(goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
        )
        if not config['share_goal_rep']:
            network_info.update(
                actor_goal_rep_def=(actor_goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='saw',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            low_actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Temperature in low-level actor.
            awr_alpha=3.0,  # Temperature in AWR.
            kl_alpha=3.0,  # Temperature for waypoint advantage.
            subgoal_steps=25,  # Number of steps to sample waypoints.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            share_goal_rep=False,  # Whether to share the goal representation between the actor and value function (grads included).
            rep_dim=10,  # Dimension of the goal representation.
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the current state.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            actor_geom_discount=0.99,  # Discount factor for actor goals if using geometric sampling.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
