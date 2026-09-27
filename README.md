# My Tool Box

**Networked multi-agent control environments and research utilities for Python.**

[![CI](https://github.com/EricDMW/My_Tool_Box/actions/workflows/ci.yml/badge.svg)](https://github.com/EricDMW/My_Tool_Box/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-2563EB)
![Gymnasium](https://img.shields.io/badge/gymnasium-%E2%89%A5%201.0-059669)
![License](https://img.shields.io/badge/license-MIT-6B7280)

My Tool Box simulates systems in which many agents act on continuous physical
quantities and interact through a network: generators holding the grid
frequency, vehicles keeping a safe gap, oscillators synchronising, robots
tracking targets as a team. Every environment speaks the Gymnasium API,
simulates thousands of copies as one batch, and comes with a classical
controller to compare against. The `toolkit` package covers the rest of a
study: publication-quality plots, neural-network building blocks and
reproducible experiment parameters.

<p align="center">
  <img src="docs/images/hero.gif" width="820" alt="Platoon-v0 under cooperative adaptive cruise control">
</p>

**Introduction slides:** [`docs/slides/my_tool_box_slides.pdf`](docs/slides/my_tool_box_slides.pdf)
&nbsp;&middot;&nbsp; **Handbook:** [`docs/manual/main.pdf`](docs/manual/main.pdf)
&nbsp;&middot;&nbsp; **Changes:** [`CHANGELOG.md`](CHANGELOG.md)

## Highlights

- **Networked by construction.** Agents interact through an explicit graph
  that every networked environment exposes (`env.adjacency`). Shared
  topologies, Laplacians and `k`-hop neighbourhoods live in
  `env_lib.utils.graphs`, and observations are local to each agent.
- **Continuous control in physical units.** Power-grid swing dynamics,
  vehicle platoons with actuator lag, oscillator networks, single and double
  integrators, and range-bearing target tracking, each with documented
  observation layouts, units and bounds.
- **Fast.** Environments are written batch-first: `env_lib.make_vec(id, 1024)`
  advances 1024 copies with a few array operations, TBD agent-steps per second
  on one CPU core. Single environments are vectorised over agents
  (Kuramoto with 50 oscillators: TBD per step).
- **Convenient.** One catalogue (`env_lib.catalog()`, `env-lib list`), one
  command to run, record, evaluate or benchmark any environment, adapters for
  single-agent libraries and the PettingZoo parallel API, and
  `env_lib.baseline_policy(env)`, a decentralised classical controller for
  every environment.
- **Reproducible.** Randomness comes only from seeded generators, rendering
  works headless in dark and light themes, and episodes export to GIF, MP4 or
  `.npz` trajectories.
- **One install for the whole study.** `toolkit.plotkit` for figures with
  confidence bands, `toolkit.neural_toolkit` for PyTorch policies and critics,
  `toolkit.parakit` for experiment parameters.

## Installation

Python 3.9 or newer. From a clone of the repository:

```bash
git clone https://github.com/EricDMW/My_Tool_Box.git
cd My_Tool_Box
pip install -e ".[all]"
```

The core install needs only NumPy, SciPy, matplotlib, Gymnasium and PyYAML.
Heavier dependencies are extras:

| Extra | Adds | Needed for |
|---|---|---|
| `torch` | PyTorch | `KuramotoOscillatorTorch-*`, `toolkit.neural_toolkit` |
| `pistonball` | pygame, pymunk | `Pistonball-v0` |
| `video` | imageio, imageio-ffmpeg | MP4 export (GIF export works without it) |
| `dev` | pytest, pytest-cov, ruff | tests and linters |
| `all` | all of the above except `dev` | |

## Quick start

```python
import env_lib

env = env_lib.make("PowerGrid-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)              # (n_agents, obs_dim) float32
policy = env_lib.baseline_policy(env)      # droop control, from the observation
obs, reward, terminated, truncated, info = env.step(policy(obs))
info["agent_rewards"]                      # per-agent rewards, shape (n_agents,)
frame = env.render()                       # (H, W, 3) uint8 dashboard

# 1024 copies simulated as one batch, evaluated in parallel
envs = env_lib.make_vec("PowerGrid-v0", num_envs=1024)
print(env_lib.evaluate(envs, policy, n_episodes=1024, seed=0))
```

Plug the same environments into existing learners:

```python
from env_lib.wrappers import FlattenJointSpaces, to_parallel

flat = FlattenJointSpaces(env_lib.make("Platoon-v0"))  # 1-D spaces for single-agent libraries
par = to_parallel("Platoon-v0")                        # PettingZoo parallel API, per-agent dicts
```

Or stay in the shell:

```bash
env-lib list --continuous                          # catalogue of continuous environments
env-lib describe Platoon-v0                        # spaces, observation layout, parameters
env-lib run Formation-v0 --gif renders/run.gif     # roll out the baseline and record it
env-lib evaluate PowerGrid-v0 --num-envs 64        # random versus baseline, in parallel
env-lib bench Platoon-v0 --num-envs 1024           # single, native batch and Sync throughput
```

## Environments

Eighteen registered configurations in eight families. Observations are stacked
per agent as `(n_agents, obs_dim)`; the table lists the default configuration.

**Continuous states and actions**

| Id | System | State and dynamics | Action per agent | Agents | Native batch |
|---|---|---|---|---:|:---:|
| `PowerGrid-v0` | frequency control of a transmission network | bus angles and frequencies, swing equations | bounded power injection | 16 | yes |
| `Platoon-v0` | cooperative adaptive cruise control | gaps, speeds, accelerations with actuator lag | commanded acceleration | 8 | yes |
| `Consensus-v0`, `Formation-v0` | rendezvous and formation over a communication graph | planar positions (and velocities), single or double integrator | velocity or acceleration | 8 | yes |
| `KuramotoOscillator-*` | synchronisation of coupled oscillators | phases, phase-coupled dynamics | control inputs and coupling gains | network | yes |
| `KuramotoOscillatorTorch-*` | the same on PyTorch, batched on CPU or GPU | | | network | tensors |
| `AJLATT-v0` | joint localisation and target tracking by a robot team | poses, EKF beliefs fused by covariance intersection | linear and angular velocity | 4 | |
| `Pistonball-v0` | cooperative rigid-body physics | ball and piston states (pymunk) | piston velocity (or discrete) | 20 | |

**Discrete networked benchmarks**

| Id | System | Action per agent | Agents |
|---|---|---|---:|
| `LineMsg-v0` | relay a message along a line of lossy links | relay or not | 10 |
| `WirelessComm-v0`, `-v1` | deliver packets through shared access points | idle or one of four access points | 36 / 16 |

| Power grid (`PowerGrid-v0`) | Vehicle platoon (`Platoon-v0`) |
|---|---|
| ![PowerGrid](docs/manual/figures/power_grid.png) | ![Platoon](docs/manual/figures/platoon.png) |
| **Formation control (`Formation-v0`)** | **Multi-robot target tracking (`AJLATT-v0`)** |
| ![Formation](docs/manual/figures/formation.png) | ![AJLATT](docs/manual/figures/ajlatt.png) |
| **Kuramoto synchronisation** | **Wireless access grid** |
| ![Kuramoto](docs/manual/figures/kuramoto.png) | ![WirelessComm](docs/manual/figures/wireless.png) |

Every environment ships a decentralised baseline, returned by
`env_lib.baseline_policy(env)`:

| Environment | Baseline controller | Random | Baseline |
|---|---|---:|---:|
| TBD | | | |

Mean return over seeded episodes, higher is better (`env-lib evaluate`).

## Design

<p align="center">
  <img src="docs/images/design.png" width="820" alt="Architecture: access layer, Gymnasium contract, environments, shared services">
</p>

1. **One contract.** Every environment follows the Gymnasium API with joint
   spaces `(n_agents, dim)`, per-agent rewards in `info["agent_rewards"]`,
   `terminated` for task outcomes and `truncated` for its own step limit, and
   randomness from `self.np_random` only.
2. **Batch-first kernels.** Continuous environments implement their dynamics
   once for arrays with a leading batch axis. The single environment is the
   batch of one; `BatchedVectorEnv` runs the same kernels for `B` copies with
   Gymnasium's autoreset modes, and copy 0 reproduces the single environment.
3. **The graph is data.** Interaction topologies are adjacency matrices built
   by `env_lib.utils.graphs` and exposed by the environments, so analysis,
   neighbourhood truncation and rendering use the same object.
4. **Baselines from observations.** Reference controllers are pure functions
   of the observation, which keeps them decentralised and lets one call drive
   a single environment or a batch of thousands.
5. **Light core, lazy extras.** `import env_lib` loads no optional
   dependency; renderers, PyTorch and pymunk are imported on first use.

## Performance

Measured with `benchmarks/benchmark_envs.py` and `benchmarks/benchmark_vector.py`
on one CPU core (`OMP_NUM_THREADS=1`).

TBD

## Documentation

- **Handbook** (`docs/manual/main.pdf`, built by `docs/manual/build.sh`): model
  equations, observation layouts, parameters and rendering of every
  environment, the workflow chapter (vectorised simulation, adapters,
  baselines, evaluation), the toolkit, examples, troubleshooting, an API
  reference and a migration guide.
- **Slides** (`docs/slides/my_tool_box_slides.pdf`, built by
  `docs/slides/build.sh`): a short introduction to the package.
- **Examples** (`examples/`, see `examples/README.md`): one script per
  environment and an end-to-end workflow.

## Development

```bash
pip install -e ".[all,dev]"
pytest -q                                          # test suite
ruff check src tests examples benchmarks           # lint
ruff format src tests examples benchmarks          # format
python benchmarks/benchmark_envs.py                # single-environment timings
python benchmarks/benchmark_vector.py              # batched throughput
```

Tests run headless (`tests/conftest.py` selects matplotlib's Agg backend and
SDL's dummy video driver). Continuous integration runs lint, tests and example
smoke tests on Python 3.9 to 3.12.

## License

MIT License, see `LICENSE`. Copyright (c) 2025 Dongming Wang (EricDMW).
