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
| `ajlatt_demo.py` | Multi-robot localisation and target tracking with an encircling heuristic | `python examples/ajlatt_demo.py --save renders/ajlatt.gif` |

Rendering is headless-friendly: `--save` records `rgb_array` frames with
`env_lib.utils.record_episode`, and `--render-mode human` opens a window when an
interactive matplotlib backend (for example TkAgg or QtAgg) or a display for
pygame is available.

## Toolkit

| Script | Shows | Typical invocation |
|---|---|---|
| `plotkit_gallery.py` | Publication-style figure: learning curves with confidence bands, grouped bars, sweep heatmap | `python examples/plotkit_gallery.py --formats pdf png` |
| `parakit_demo.py` | Saving, loading and applying `argparse` parameters; optional Tk editor | `python examples/parakit_demo.py --set lr=3e-4` |
| `simple_transformer_example.py` | Transformer policy trained with REINFORCE on a sequential toy task | `python examples/simple_transformer_example.py --quick` |
| `transformer_rl_example.py` | Transformer actor-critic with batched environments | `python examples/transformer_rl_example.py --quick` |

The two transformer examples require PyTorch. `--quick` runs a few iterations
only and is used as a smoke test.
