# Examples

Runnable scripts, grouped by topic. Every script has a `--help` flag, and media
files (GIF, PNG, PDF, CSV) are written to `renders/`, which git ignores. Run the
scripts from the repository root.

```
examples/
  getting_started/   first steps: every environment, the vectorised workflow
  environments/      one demo per environment family, with its classical controller
  algorithms/        training the MARL algorithms; comparing your method with baselines
  toolkit/           plotting, parameter management, transformer policies
```

Install the package with the extras the examples need first:

```bash
pip install -e ".[all]"          # or ".[pistonball]" / ".[torch]" for a subset
```

## Getting started (`getting_started/`)

| Script | Shows | Typical invocation |
|---|---|---|
| `quickstart.py` | Every registered environment created through `env_lib.make`, a few random steps each | `python examples/getting_started/quickstart.py --render renders/quickstart` |
| `workflow_demo.py` | The workflow in six steps: catalogue, native vector environment, parallel evaluation of random actions and the baseline, flattened spaces, PettingZoo Parallel API, GIF of the baseline | `python examples/getting_started/workflow_demo.py --env Consensus-v0 --num-envs 128 --episodes 128` |

## Environments (`environments/`)

| Script | Shows | Typical invocation |
|---|---|---|
| `kuramoto_demo.py` | Kuramoto oscillator network driven to synchronisation (NumPy or batched PyTorch backend) | `python examples/environments/kuramoto_demo.py --save renders/kuramoto.gif` |
| `linemsg_demo.py` | Line message passing: random vs duty-cycled vs always-relay policies | `python examples/environments/linemsg_demo.py --policy relay --save renders/linemsg.gif` |
| `wireless_comm_demo.py` | Wireless access grid: random, slotted ALOHA and a collision-free schedule | `python examples/environments/wireless_comm_demo.py --policy schedule --save renders/wireless.gif` |
| `pistonball_demo.py` | Pistonball: heuristic vs random pistons, optional keyboard control (`--manual`) | `python examples/environments/pistonball_demo.py --save renders/pistonball.gif` |
| `consensus_demo.py` | Networked consensus / formation control with the Laplacian baseline | `python examples/environments/consensus_demo.py --task formation --save renders/formation.gif` |
| `power_grid_demo.py` | Power-grid frequency control (`PowerGrid-v0`): no control, random injections and decentralised droop control under the same disturbances; frequency nadir, settling time, trips | `python examples/environments/power_grid_demo.py --buses 32 --topology ring --render renders/power_grid.gif` |
| `platoon_demo.py` | Vehicle platoon (`Platoon-v0`): random actions, sensor-only ACC and cooperative ACC, with the analytic string-stability gain; collisions and spacing errors | `python examples/environments/platoon_demo.py --scenario stop_and_go --render renders/platoon.gif` |
| `ajlatt_demo.py` | Multi-robot localisation and target tracking with the packaged encircling baseline (`env_lib.baseline_policy`) | `python examples/environments/ajlatt_demo.py --save renders/ajlatt.gif` |

Rendering is headless-friendly: `--save` (`--render` in `power_grid_demo.py`
and `platoon_demo.py`) records `rgb_array` frames with
`env_lib.utils.record_episode`, and `--render-mode human` opens a window when an
interactive matplotlib backend (for example TkAgg or QtAgg) or a display for
pygame is available.

### Without writing code: the `env-lib` command line

Installing the package also installs the `env-lib` console script (equivalently
`python -m env_lib`), which lists, describes, runs, records, evaluates and
benchmarks every registered environment with its baseline controller or random
actions:

```bash
env-lib list --continuous                          # catalogue of the environments
env-lib describe PowerGrid-v0                      # spaces, parameters, baseline
env-lib run Platoon-v0 --gif renders/platoon.gif   # one episode, recorded
env-lib evaluate Formation-v0 --episodes 64 --num-envs 32
env-lib bench Consensus-v0 --num-envs 256 --steps 200
```

Constructor arguments are passed as `--kwarg key=value ...`; `env-lib <command> -h`
lists the options. The handbook (`docs/manual`, chapter "Vectorised Simulation,
Adapters, Baselines and Evaluation") describes every command.

## Multi-agent reinforcement learning (`algorithms/`)

| Script | Shows | Typical invocation |
|---|---|---|
| `marl_training_demo.py` | `marl_algorithms` on the environments: MAPPO on PowerGrid, MADDPG on Consensus, QMIX on LineMsg and VDN on WirelessComm, each trained with its tuned preset on batched copies, then compared with random actions and the classical controller; learning curves saved as PNG | `python examples/algorithms/marl_training_demo.py` (a few minutes on one core), `--algo mappo --env PowerGrid-v0 --gif renders/mappo.gif`, `--quick` |
| `baseline_comparison.py` | Your method against the baselines: a distributed droop controller compared with MAPPO and IPPO (trained over two seeds), the classical controller and random actions on the same episodes, with `marl_algorithms.compare`; table, CSV and bar chart | `python examples/algorithms/baseline_comparison.py` (about 6 minutes), `--seeds 0 1 2 --algos mappo ippo maddpg`, `--quick` |

The `marl-train` command line trains and compares the algorithms without
writing code:

```bash
marl-train list                          # the seven algorithms and their papers
marl-train presets                       # tuned (algorithm, environment) pairs
marl-train run mappo PowerGrid-v0        # train with the preset, then evaluate
marl-train run qmix LineMsg-v0 --save runs/qmix.pt --csv runs/qmix.csv
marl-train evaluate runs/qmix.pt LineMsg-v0
marl-train compare PowerGrid-v0 --seeds 0 1 2    # every preset algorithm as a baseline
```

## Toolkit (`toolkit/`)

| Script | Shows | Typical invocation |
|---|---|---|
| `plotkit_gallery.py` | Publication-style figure: learning curves with confidence bands, grouped bars, sweep heatmap | `python examples/toolkit/plotkit_gallery.py --formats pdf png` |
| `parakit_demo.py` | Saving, loading and applying `argparse` parameters; optional Tk editor | `python examples/toolkit/parakit_demo.py --set lr=3e-4` |
| `simple_transformer_example.py` | Transformer policy trained with REINFORCE on a sequential toy task | `python examples/toolkit/simple_transformer_example.py --quick` |
| `transformer_rl_example.py` | Transformer actor-critic with batched environments | `python examples/toolkit/transformer_rl_example.py --quick` |

The two transformer examples require PyTorch. `--quick` runs a few iterations
only and is used as a smoke test.
