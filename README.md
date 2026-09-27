# My Tool Box

Multi-agent reinforcement learning environments and research utilities for
Python.

- **`env_lib`**: seven families of Gymnasium environments for networked and
  multi-agent control, from Kuramoto oscillator synchronisation to multi-robot
  target tracking. They share one API, have vectorised NumPy/PyTorch
  implementations, render headless and are seeded reproducibly.
- **`toolkit`**: research utilities:
  - `plotkit` for publication-quality plots;
  - `neural_toolkit` for PyTorch policy, value and Q networks, encoders and
    decoders;
  - `parakit` for managing `argparse` experiment parameters.

| Multi-robot target tracking (`AJLATT-v0`) | Kuramoto synchronisation (`KuramotoOscillator-v0`) |
|---|---|
| ![AJLATT](docs/manual/figures/ajlatt.png) | ![Kuramoto](docs/manual/figures/kuramoto.png) |
| **Formation control (`Formation-v0`)** | **Wireless access grid (`WirelessComm-v0`)** |
| ![Formation](docs/manual/figures/formation.png) | ![WirelessComm](docs/manual/figures/wireless.png) |

## Installation

The package requires Python 3.9 or newer. Install it from a clone of the repository:

```bash
git clone https://github.com/EricDMW/My_Tool_Box.git
cd My_Tool_Box
pip install -e ".[all]"
```

The core install needs only NumPy, SciPy, matplotlib, Gymnasium and PyYAML.
Heavier dependencies come as extras:

| Extra | Adds | Needed for |
|---|---|---|
| `pistonball` | pygame, pymunk | `PistonballEnv` |
| `torch` | PyTorch | `KuramotoOscillatorEnvTorch`, `toolkit.neural_toolkit` |
| `video` | imageio, imageio-ffmpeg | MP4 export (GIF export works without it) |
| `dev` | pytest, pytest-cov, ruff | running the tests and linters |
| `all` | all of the above except `dev` | |

For example, `pip install -e ".[pistonball,dev]"`. For a CPU-only PyTorch,
install `torch` from the PyTorch index before the package.

## Quick start

Every environment follows the Gymnasium API and is registered when `env_lib`
is imported:

```python
import env_lib

print(env_lib.list_envs())                      # all registered ids

env = env_lib.make("KuramotoOscillator-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
frame = env.render()                            # (H, W, 3) uint8 array
env.close()
```

Classes can also be used directly, and episodes recorded as GIF or MP4:

```python
from env_lib import AJLATTEnv
from env_lib.utils import record_episode, set_theme

set_theme("light")                              # or "dark" (default)
env = AJLATTEnv(map_name="obstacles05", num_robots=4, render_mode="rgb_array")
frames = record_episode(env, path="renders/ajlatt.gif", seed=0, max_steps=120)
```

The multi-agent environments use joint spaces:

- Observations are stacked per agent as `(n_agents, obs_dim)`.
- Actions are joint arrays.
- `info["agent_rewards"]` always holds the per-agent rewards.

Publication-quality learning curves take a few lines:

```python
import numpy as np
from toolkit.plotkit import plot_learning_curves, save_figure

runs = {"PPO": np.random.randn(5, 200).cumsum(1), "SAC": np.random.randn(5, 200).cumsum(1)}
ax = plot_learning_curves(runs, band="ci95", smoothing=0.9, xlabel="Episode", ylabel="Return")
save_figure(ax, "renders/learning_curves", formats=("pdf", "png"))
```

## Environments

| Id | Description | Agents | Actions |
|---|---|---|---|
| `KuramotoOscillator-v0` (and variants) | Synchronise a network of coupled oscillators by control inputs and coupling strengths; NumPy backend | oscillator network | continuous |
| `KuramotoOscillatorTorch-v0` (and variants) | Batched PyTorch backend of the Kuramoto environment (CPU or GPU) | batch of networks | continuous |
| `LineMsg-v0` | Relay a message along a line of agents with lossy links | 10 | binary per agent |
| `WirelessComm-v0`, `-v1` | Deliver packets through shared access points without collisions | 6x6 / 4x4 grid | 5 choices per agent |
| `Pistonball-v0` | Cooperatively move a ball to the goal with pistons (pymunk physics) | 20 | continuous or 3 choices |
| `Consensus-v0`, `Formation-v0` | Rendezvous or formation control over a communication graph | 8 | continuous 2-D |
| `AJLATT-v0` | Active joint localisation and target tracking by a robot team with range-bearing sensing and covariance-intersection fusion | 4 | continuous (v, omega) |

The handbook describes every environment in detail: dynamics, observation
layout, reward, parameters and rendering.

## Performance

These are mean step times without rendering, measured with
`benchmarks/benchmark_envs.py` on the development machine, before and after
the 1.0 rewrite.

| Environment | Before | After |
|---|---:|---:|
| Kuramoto, NumPy, 50 oscillators | 2.36 ms | 0.07 ms |
| Kuramoto, NumPy RK4, 50 oscillators | 7.1 ms | 0.1 ms |
| Kuramoto, PyTorch, 50 oscillators x 8 systems | 10.3 ms | 0.4 ms |
| WirelessComm, 12x12 | 0.56 ms | 0.05 ms |
| Pistonball, 20 pistons | 1.11 ms | 0.06 ms |
| AJLATT, `obstacles04`, 4 robots | 90 ms | 5 to 9 ms |

Rendering an `rgb_array` frame takes about 15 to 35 ms. For comparison, the
old AJLATT took about 110 ms per frame, and the old Kuramoto `rgb_array` path
crashed on matplotlib 3.10 and later. Refactors that were not meant to change
results were checked against the previous implementation: seeded trajectories
are bit-identical or agree to numerical precision.

## Project layout

```text
src/env_lib/          environments
  kos_env/            Kuramoto oscillators (NumPy and PyTorch backends)
  linemsg_env/        line message passing
  wireless_comm_env/  wireless access grid
  pistonball_env/     Pistonball (pygame, pymunk)
  consensus_env/      consensus and formation control
  ajlatt_env/         multi-robot localisation and target tracking, maps, map builder
  utils/              rendering themes, figure management, episode recording
src/toolkit/          research utilities
  plotkit/            plotting
  neural_toolkit/     PyTorch network building blocks and tabular RL tools
  parakit/            argparse parameter management (optional Tk editor)
examples/             runnable example scripts (see examples/README.md)
benchmarks/           performance benchmarks
tests/                pytest suite
docs/manual/          LaTeX user manual (handbook)
```

## Documentation

The user manual is in `docs/manual/`. Build it with `docs/manual/build.sh`,
which needs a TeX distribution with `latexmk`; the pre-built PDF is
`docs/manual/main.pdf`. It covers installation, every environment, the
toolkit, examples, troubleshooting, an API reference and a migration guide
from pre-1.0 versions. `CHANGELOG.md` lists every change, including
behavioural fixes.

## Development

```bash
pip install -e ".[all,dev]"
pytest -q                                              # test suite
ruff check src tests examples benchmarks               # lint
ruff format src tests examples benchmarks              # format
python benchmarks/benchmark_envs.py                    # performance
```

Tests run headless: `tests/conftest.py` selects the Agg backend for
matplotlib and SDL's dummy video driver for pygame. Continuous integration
runs the lint and test jobs on Python 3.9 to 3.12.

## License

MIT License, see `LICENSE`. Copyright (c) 2025 Dongming Wang (EricDMW).
