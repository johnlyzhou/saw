# Flattening Hierarchies with Policy Bootstrapping
[Project Page](https://johnlyzhou.github.io/saw/) | [Paper](https://arxiv.org/abs/2505.14975) | [Thread](https://x.com/johnlyzhou/status/1932496679909064980)

This is the official implementation of **Subgoal Advantage-Weighted Policy Bootstrapping** (**SAW**), built on a fork of the excellent [OGBench](https://github.com/seohongpark/ogbench) codebase by Park et al. (2024). Run commands with hyperparameters from the paper can be found [here](https://github.com/johnlyzhou/saw/blob/master/impls/hyperparameters.sh#L1). In addition to the original reference implementations provided by OGBench and SAW, we also include implementations of an offline version of RIS (Chane-Sane et al., 2021) and GCWAE (see section 4.2 of the paper). 

### Citation

```bibtex
@article{zhou_flattening_2025,
  title = {Flattening Hierarchies with Policy Bootstrapping},
  url = {http://arxiv.org/abs/2505.14975},
  doi = {10.48550/arXiv.2505.14975},
  publisher = {arXiv},
  author = {Zhou, John L. and Kao, Jonathan C.},
  year = {2025},
}
```

We append the installation instructions from the original OGBench README below for ease of reference.

#

### Installation

OGBench can be easily installed via PyPI:

```shell
pip install ogbench
```

It requires Python 3.8+ and has only three dependencies: `mujoco >= 3.1.6`, `dm_control >= 1.0.20`,
and `gymnasium`.

To use OGBench for **offline goal-conditioned RL**,
go to [this section](#usage-for-offline-goal-conditioned-rl).
To use OGBench for **standard (non-goal-conditioned) offline RL**,
go to [this section](#usage-for-standard-non-goal-conditioned-offline-rl).

### Reference Implementations

OGBench also provides JAX-based reference implementations of six offline goal-conditioned RL algorithms
(GCBC, GCIVL, GCIQL, QRL, CRL and HIQL), in addition to SAW, GCWAE, and RIS.
They are provided in the `impls` directory as a **standalone** codebase.
You can safely remove the other parts of the repository if you only need the reference implementations
and do not want to modify the environments.

Our reference implementations require Python 3.9+ and additional dependencies, including `jax >= 0.4.26`.
To install these dependencies, run:

```shell
cd impls
pip install -r requirements.txt
```

By default, it uses the PyPI version of OGBench.
If you want to use a local version of OGBench (e.g., for training methods on modified environments),
run instead `pip install -e ".[train]"` in the root directory.

### Running the reference implementations

Each algorithm is implemented in a separate file in the `agents` directory.
We provide implementations of the following offline goal-conditioned RL algorithms, with additions to the original OGBench algorithms in **bold**:

- `saw.py`: **Subgoal Advantage-Weighted Bootstrapping (SAW)**
- `gcwae.py`: **Goal-Conditioned Waypoint Advantage Estimation (GCWAE)**
- `ris.py`: **Reinforcement Learning with Imagined Subgoals (RIS)**
- `gcbc.py`: Goal-Conditioned Behavioral Cloning (GCBC)
- `gcivl.py`: Goal-Conditioned Implicit V-Learning (GCIVL)
- `gciql.py`: Goal-Conditioned Implicit Q-Learning (GCIQL)
- `qrl.py`: Quasimetric Reinforcement Learning (QRL)
- `crl.py`: Contrastive Reinforcement Learning (CRL)
- `hiql.py`: Hierarchical Implicit Q-Learning (HIQL)

To train an agent, you can run the `main.py` script.
Training metrics, evaluation metrics, and videos are logged via `wandb` by default.
Here are some example commands (see [hyperparameters.sh](impls/hyperparameters.sh) for the full list of commands):

```shell
# antmaze-large-navigate-v0 (SAW)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/saw.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.kl_alpha=3.0
# antmaze-large-navigate-v0 (GCWAE)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcwae.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
# antmaze-large-navigate-v0 (RIS)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/ris.py --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.kl_alpha=3.0
# antmaze-large-navigate-v0 (GCBC)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcbc.py
# antmaze-large-navigate-v0 (GCIVL)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl.py --agent.alpha=10.0
# antmaze-large-navigate-v0 (GCIQL)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gciql.py --agent.alpha=0.3
# antmaze-large-navigate-v0 (QRL)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/qrl.py --agent.alpha=0.003
# antmaze-large-navigate-v0 (CRL)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl.py --agent.alpha=0.1
# antmaze-large-navigate-v0 (HIQL)
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/hiql.py --agent.high_alpha=3.0 --agent.low_alpha=3.0
```

Each run typically takes 2-5 hours (on state-based tasks)
or 5-12 hours (on pixel-based tasks) on a single A5000 GPU.
For large pixel-based datasets (e.g., `visual-puzzle-4x6-play-v0` with 5M transitions),
up to 120GB of RAM may be required.

### Credit

If you use our method or code in your research, we encourage you to additionally cite the OGBench benchmark, on top of which our implementations are built.

```bibtex
@inproceedings{ogbench_park2025,
  title={OGBench: Benchmarking Offline Goal-Conditioned RL},
  author={Park, Seohong and Frans, Kevin and Eysenbach, Benjamin and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
}
```
