# toolkit

Research utilities distributed with the `my-tool-box` package. See the
repository `README.md` for installation and the handbook in `docs/manual/`
(Part II, "Toolkit") for the full documentation.

| Subpackage | Purpose | Extra dependencies |
|---|---|---|
| `toolkit.plotkit` | Publication-quality plots: learning curves with confidence bands, line, bar, scatter, histogram and heatmap plots, style presets and a colour-blind-safe palette | none (matplotlib) |
| `toolkit.neural_toolkit` | PyTorch policy, value and Q networks (MLP, CNN, RNN, Transformer), encoders, decoders, network utilities and tabular RL tools | `torch` extra |
| `toolkit.parakit` | Save, load and validate `argparse` experiment parameters; optional Tk editor | tkinter only for the editor |

Subpackages are imported lazily, so `import toolkit` does not import PyTorch
or tkinter.

```python
import numpy as np
from toolkit.plotkit import plot_learning_curves, save_figure

runs = {"PPO": np.random.randn(5, 200).cumsum(1)}
ax = plot_learning_curves(runs, band="ci95", xlabel="Episode", ylabel="Return")
save_figure(ax, "learning_curves", formats=("pdf", "png"))
```

```python
from toolkit.neural_toolkit import MLPPolicyNetwork

policy = MLPPolicyNetwork(input_dim=8, output_dim=2, hidden_dims=[64, 64])
```

Run the plot gallery with `plotkit-gallery --demo all --save renders/gallery`.
