# Examples

Runnable scripts demonstrating the environments (`env_lib`) and the research
utilities (`toolkit`). Every script has a `--help` flag. Media files (GIF, PNG,
PDF) are written to `renders/`, which is ignored by git.

Install the package with the extras needed by the examples first:

```bash
pip install -e ".[all]"          # or ".[pistonball]" / ".[torch]" for a subset
```

## Environments

| Script | Shows | Typical invocation |
|---|---|---|
| `quickstart.py` | Every registered environment created through `env_lib.make`, a few random steps each | `python examples/quickstart.py --render renders/quickstart` |
| `kuramoto_demo.py` | Kuramoto oscillator network driven to synchronisation (NumPy or batched PyTorch backend) | `python examples/kuramoto_demo.py --save renders/kuramoto.gif` |
| `linemsg_demo.py` | Line message passing: random vs duty-cycled vs always-relay policies | `python examples/linemsg_demo.py --policy relay --save renders/linemsg.gif` |
| `wireless_comm_demo.py` | Wireless access grid: random, slotted ALOHA and a collision-free schedule | `python examples/wireless_comm_demo.py --policy schedule --save renders/wireless.gif` |
| `pistonball_demo.py` | Pistonball: heuristic vs random pistons, optional keyboard control (`--manual`) | `python examples/pistonball_demo.py --save renders/pistonball.gif` |
| `consensus_demo.py` | Networked consensus / formation control with the Laplacian baseline | `python examples/consensus_demo.py --task formation --save renders/formation.gif` |
| `power_grid_demo.py` | Power-grid frequency control (`PowerGrid-v0`): no control, random injections and decentralised droop control under the same disturbances; frequency nadir, settling time, trips | `python examples/power_grid_demo.py --buses 32 --topology ring --render renders/power_grid.gif` |
| `platoon_demo.py` | Vehicle platoon (`Platoon-v0`): random actions, sensor-only ACC and cooperative ACC, with the analytic string-stability gain; collisions and spacing errors | `python examples/platoon_demo.py --scenario stop_and_go --render renders/platoon.gif` |
| `ajlatt_demo.py` | Multi-robot localisation and target tracking with the packaged encircling baseline (`env_lib.baseline_policy`) | `python examples/ajlatt_demo.py --save renders/ajlatt.gif` |
| `workflow_demo.py` | The 1.1 workflow in six steps: catalogue, native vector environment, parallel evaluation of random actions and the baseline, flattened spaces, PettingZoo Parallel API, GIF of the baseline | `python examples/workflow_demo.py --env Consensus-v0 --num-envs 128 --episodes 128` |

Rendering is headless-friendly: `--save` (`--render` in `power_grid_demo.py`
and `platoon_demo.py`) records `rgb_array` frames with
`env_lib.utils.record_episode`, and `--render-mode human` opens a window when an
interactive matplotlib backend (for example TkAgg or QtAgg) or a display for
pygame is available.

## Without writing code: the `env-lib` command line

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

## Multi-agent reinforcement learning

| Script | Shows | Typical invocation |
|---|---|---|
| `marl_training_demo.py` | `marl_algorithms` on the environments: MAPPO on PowerGrid, MADDPG on Consensus, QMIX on LineMsg and VDN on WirelessComm, each trained with its tuned preset on batched copies, then compared with random actions and the classical controller; learning curves saved as PNG | `python examples/marl_training_demo.py` (a few minutes on one core), `--algo mappo --env PowerGrid-v0 --gif renders/mappo.gif`, `--quick` |

The `marl-train` command line trains any algorithm without writing code:

```bash
marl-train list                          # the seven algorithms and their papers
marl-train presets                       # tuned (algorithm, environment) pairs
marl-train run mappo PowerGrid-v0        # train with the preset, then evaluate
marl-train run qmix LineMsg-v0 --save runs/qmix.pt --csv runs/qmix.csv
marl-train evaluate runs/qmix.pt LineMsg-v0
```

## Toolkit

| Script | Shows | Typical invocation |
|---|---|---|
| `plotkit_gallery.py` | Publication-style figure: learning curves with confidence bands, grouped bars, sweep heatmap | `python examples/plotkit_gallery.py --formats pdf png` |
| `parakit_demo.py` | Saving, loading and applying `argparse` parameters; optional Tk editor | `python examples/parakit_demo.py --set lr=3e-4` |
| `simple_transformer_example.py` | Transformer policy trained with REINFORCE on a sequential toy task | `python examples/simple_transformer_example.py --quick` |
| `transformer_rl_example.py` | Transformer actor-critic with batched environments | `python examples/transformer_rl_example.py --quick` |

The two transformer examples require PyTorch. `--quick` runs a few iterations
only and is used as a smoke test.
