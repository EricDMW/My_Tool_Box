"""Tests for :mod:`toolkit.plotkit` (headless, Agg backend)."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import BarContainer

import toolkit.plotkit as pk
from toolkit.plotkit import core
from toolkit.plotkit.__main__ import DEMOS, main
from toolkit.plotkit._data import as_series_list, to_numpy
from toolkit.plotkit._stats import _ema_filter, _ema_filter_loop, ema, moving_average

OLD_NAMES = [
    "plot_shadow_curve",
    "plot_heatmap",
    "plot_gray_scale",
    "plot_line",
    "plot_bar",
    "plot_scatter",
    "set_research_style",
    "RESEARCH_COLORS",
    "RESEARCH_COLOR_LIST",
]
Z95 = 1.959963984540054
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_matplotlib():
    """Close figures and restore rcParams after every test."""
    with matplotlib.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def rc_snapshot():
    snapshot = dict(matplotlib.rcParams.copy())
    snapshot.pop("backend", None)
    return snapshot


def bar_containers(ax):
    return [c for c in ax.containers if isinstance(c, BarContainer)]


def band_edges(ax, x, index=0):
    """Lower and upper edge of the ``index``-th fill_between band at each x."""
    paths = ax.collections[index].get_paths()
    verts = np.concatenate([p.vertices for p in paths])
    lower, upper = [], []
    for xi in x:
        ys = verts[np.isclose(verts[:, 0], xi), 1]
        lower.append(ys.min())
        upper.append(ys.max())
    return np.array(lower), np.array(upper)


def ref_moving_average(values, window):
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        chunk = values[max(0, t - window + 1) : t + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            out[t] = chunk.mean()
    return out


def ref_ema(values, weight):
    out = np.full(len(values), np.nan)
    num = den = 0.0
    for t, v in enumerate(values):
        valid = np.isfinite(v)
        num = weight * num + (1 - weight) * (v if valid else 0.0)
        den = weight * den + (1 - weight) * float(valid)
        if den > 0:
            out[t] = num / den
    return out


# ---------------------------------------------------------------------------
# API and backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", OLD_NAMES)
def test_old_names_importable_from_package_and_core(name):
    assert getattr(pk, name) is getattr(core, name)
    assert name in pk.__all__


def test_new_api_exported():
    for name in (
        "plot_learning_curves",
        "plot_histogram",
        "save_figure",
        "style_context",
        "reset_style",
        "get_palette",
        "OKABE_ITO_COLORS",
        "OKABE_ITO_COLOR_LIST",
        "STYLE_PRESETS",
    ):
        assert hasattr(pk, name) and name in pk.__all__


def test_plot_shadow_curve_signature_is_backward_compatible():
    params = inspect.signature(pk.plot_shadow_curve).parameters
    expected = [
        ("y", inspect.Parameter.empty),
        ("x", None),
        ("y_std", None),
        ("labels", None),
        ("colors", None),
        ("alpha", 0.2),
        ("ax", None),
        ("axis", 0),
        ("figsize", (10, 6)),
        ("title", None),
        ("xlabel", None),
        ("ylabel", None),
        ("legend", True),
        ("grid", True),
        ("style", "research"),
        ("legend_labels", None),
        ("x_tick_labels", None),
        ("y_tick_labels", None),
    ]
    assert [(n, p.default) for n, p in list(params.items())[: len(expected)]] == expected


@pytest.mark.parametrize(
    ("func", "names"),
    [
        ("plot_line", ["x", "y", "labels", "colors", "ax", "figsize", "title", "xlabel"]),
        ("plot_scatter", ["x", "y", "labels", "colors", "ax", "figsize", "title", "xlabel"]),
        ("plot_bar", ["x", "height", "labels", "colors", "ax", "figsize", "title", "xlabel"]),
        ("plot_heatmap", ["data", "xlabels", "ylabels", "cmap", "annot", "ax", "figsize"]),
        ("plot_gray_scale", ["data", "xlabels", "ylabels", "ax", "figsize", "title"]),
    ],
)
def test_positional_order_kept(func, names):
    params = list(inspect.signature(getattr(pk, func)).parameters)
    assert params[: len(names)] == names


def test_import_needs_only_matplotlib_and_numpy():
    code = (
        "import sys, toolkit.plotkit; "
        "print(sorted(m for m in ('seaborn', 'pandas', 'torch', 'scipy', 'tensorflow') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def test_list_of_scalars_is_one_series():
    ax = pk.plot_bar(["A", "B", "C"], [10, 20, 15])
    assert len(ax.patches) == 3
    assert [p.get_height() for p in ax.patches] == [10, 20, 15]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    ax = pk.plot_line(None, [1.0, 2.0, 3.0])
    assert len(ax.lines) == 1


def test_series_rules():
    assert len(as_series_list([1, 2, 3])) == 1
    assert len(as_series_list([[1, 2], [3, 4, 5]])) == 2
    assert len(as_series_list(np.zeros((3, 4)))) == 3
    assert len(as_series_list(np.zeros((3, 4)), two_d="samples")) == 1
    assert as_series_list(5.0)[0].shape == (1,)
    with pytest.raises(ValueError, match="mixes scalars"):
        as_series_list([1, [2, 3]])
    with pytest.raises(ValueError, match="1-D or 2-D"):
        as_series_list(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="empty"):
        as_series_list([])


def test_none_input_raises():
    with pytest.raises(ValueError, match="must not be None"):
        pk.plot_shadow_curve(None)


def test_numpy_like_objects_and_masked_arrays():
    class FakeTensor:  # mimics TensorFlow eager tensors
        def __init__(self, data):
            self._data = np.asarray(data)

        def numpy(self):
            return self._data

    np.testing.assert_array_equal(to_numpy(FakeTensor([1, 2])), [1, 2])
    masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
    series = as_series_list(masked)[0]
    assert np.isnan(series[1]) and series[2] == 3.0


def test_torch_tensors(rng):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(0)
    y = torch.randn(5, 20, generator=generator, requires_grad=True)
    ax = pk.plot_shadow_curve(y, x=torch.arange(20))
    expected = y.detach().numpy().mean(axis=0)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected, rtol=1e-5, atol=1e-7)
    assert len(ax.collections) == 1

    ax = pk.plot_line(torch.arange(10), [torch.ones(10), torch.zeros(10)])
    assert len(ax.lines) == 2
    ax = pk.plot_bar(None, [torch.tensor(1.0), torch.tensor(2.0)])  # 0-d tensors -> one series
    assert len(ax.patches) == 2
    ax = pk.plot_heatmap(torch.rand(3, 4), annot=True)
    assert len(ax.texts) == 12
    ax = pk.plot_learning_curves({"a": torch.randn(3, 10, dtype=torch.bfloat16)})
    assert len(ax.lines) == 1


def test_pandas_inputs():
    pd = pytest.importorskip("pandas")
    ax = pk.plot_line(pd.Series(np.arange(5.0)), pd.Series(np.arange(5.0) ** 2))
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), np.arange(5.0) ** 2)
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    ax = pk.plot_line(None, frame)
    assert len(ax.lines) == 3  # one series per column


# ---------------------------------------------------------------------------
# plot_line / plot_scatter
# ---------------------------------------------------------------------------


def test_plot_line_broadcasts_single_x():
    x = np.linspace(0, 1, 30)
    ax = pk.plot_line(x, [np.sin(x), np.cos(x), x])
    assert len(ax.lines) == 3
    for line in ax.lines:
        np.testing.assert_array_equal(line.get_xdata(), x)
    assert ax.get_legend() is not None


def test_plot_line_2d_rows_are_series_and_per_series_x():
    ax = pk.plot_line(np.arange(4), np.arange(12.0).reshape(3, 4))
    assert len(ax.lines) == 3
    np.testing.assert_array_equal(ax.lines[2].get_ydata(), [8, 9, 10, 11])
    ax = pk.plot_line([np.arange(3), np.arange(5)], [np.ones(3), np.ones(5)])
    assert [len(line.get_xdata()) for line in ax.lines] == [3, 5]


def test_plot_line_shape_errors():
    with pytest.raises(ValueError, match=r"series 0 has 50 points but its x has 40"):
        pk.plot_line(np.arange(40), np.zeros(50))
    with pytest.raises(ValueError, match="transpose"):
        pk.plot_line(np.arange(100), np.zeros((100, 3)))
    with pytest.raises(ValueError, match="3 x arrays for 2 series"):
        pk.plot_line([np.arange(3)] * 3, [np.zeros(3), np.zeros(3)])


def test_plot_line_color_and_label_aliases():
    ax = pk.plot_line(np.arange(3), [1, 2, 3], color="red", label="only")
    assert mcolors.same_color(ax.lines[0].get_color(), "red")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["only"]


def test_plot_line_palette_name_and_label_mismatch_warning():
    ax = pk.plot_line(None, [[1, 2], [2, 3]], colors="okabe_ito")
    assert [line.get_color() for line in ax.lines] == pk.OKABE_ITO_COLOR_LIST[:2]
    with pytest.warns(UserWarning, match="1 labels for 2 series"):
        pk.plot_line(None, [[1, 2], [2, 3]], labels=["a"])


def test_plot_scatter_shared_x_and_c_kwarg(rng):
    x = rng.normal(size=20)
    ax = pk.plot_scatter(x, [x, 2 * x])
    assert len(ax.collections) == 2
    np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 1], 2 * x)
    ax = pk.plot_scatter(x, x, c=x, cmap="viridis")
    assert ax.collections[0].get_array() is not None


# ---------------------------------------------------------------------------
# plot_shadow_curve / plot_learning_curves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", ["std", "sem", "ci95", "minmax"])
def test_shadow_band_types_match_numpy(rng, band):
    data = rng.normal(size=(6, 15))
    x = np.arange(15)
    ax = pk.plot_shadow_curve(data, band=band)
    mean = data.mean(axis=0)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), mean)
    if band == "std":
        lo, hi = mean - data.std(axis=0), mean + data.std(axis=0)
    elif band == "minmax":
        lo, hi = data.min(axis=0), data.max(axis=0)
    else:
        half = data.std(axis=0, ddof=1) / np.sqrt(6) * (Z95 if band == "ci95" else 1.0)
        lo, hi = mean - half, mean + half
    lower, upper = band_edges(ax, x)
    np.testing.assert_allclose(lower, lo)
    np.testing.assert_allclose(upper, hi)


def test_shadow_band_none_and_1d_input(rng):
    ax = pk.plot_shadow_curve(rng.normal(size=(4, 10)), band=None)
    assert len(ax.collections) == 0
    ax = pk.plot_shadow_curve(np.arange(10.0))  # 1-D without y_std: no band
    assert len(ax.collections) == 0


def test_shadow_axis_one_is_transpose(rng):
    data = rng.normal(size=(5, 12))
    ax0 = pk.plot_shadow_curve(data, axis=0)
    ax1 = pk.plot_shadow_curve(data.T, axis=1)
    np.testing.assert_allclose(ax0.lines[0].get_ydata(), ax1.lines[0].get_ydata())
    with pytest.raises(ValueError, match="axis"):
        pk.plot_shadow_curve(data, axis=2)


def test_shadow_nan_robust(rng):
    data = rng.normal(size=(5, 10))
    data[0, 2] = np.nan
    data[:, 7] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = pk.plot_shadow_curve(data, band="sem")
    mean = ax.lines[0].get_ydata()
    np.testing.assert_allclose(mean[2], np.nanmean(data[:, 2]))
    assert np.isnan(mean[7])
    assert np.isfinite(np.delete(mean, 7)).all()


@pytest.mark.parametrize("smoothing", [1, 4, 0.3, 0.9])
def test_shadow_smoothing_matches_reference(rng, smoothing):
    data = rng.normal(size=(4, 25))
    x = np.arange(25)
    ax = pk.plot_shadow_curve(data, smoothing=smoothing)
    mean, std = data.mean(axis=0), data.std(axis=0)
    ref = (
        (lambda v: ref_moving_average(v, smoothing))
        if isinstance(smoothing, int)
        else (lambda v: ref_ema(v, smoothing))
    )
    np.testing.assert_allclose(ax.lines[0].get_ydata(), ref(mean))
    lower, upper = band_edges(ax, x)
    np.testing.assert_allclose(lower, ref(mean - std))
    np.testing.assert_allclose(upper, ref(mean + std))
    assert len(ax.lines[0].get_ydata()) == 25  # same length output


@pytest.mark.parametrize(
    ("smoothing", "error"),
    [
        (0, ValueError),
        (-3, ValueError),
        (1.0, ValueError),
        (1.5, ValueError),
        (True, TypeError),
        ("x", TypeError),
    ],
)
def test_shadow_invalid_smoothing(smoothing, error):
    with pytest.raises(error):
        pk.plot_shadow_curve(np.zeros((2, 5)), smoothing=smoothing)


def test_shadow_invalid_band():
    with pytest.raises(ValueError, match="band"):
        pk.plot_shadow_curve(np.zeros((2, 5)), band="iqr")


def test_shadow_explicit_y_std(rng):
    y = np.linspace(0, 1, 8)
    std = np.full(8, 0.1)
    ax = pk.plot_shadow_curve(y, y_std=std)
    lower, upper = band_edges(ax, np.arange(8))
    np.testing.assert_allclose(lower, y - 0.1)
    np.testing.assert_allclose(upper, y + 0.1)
    ax = pk.plot_shadow_curve([y, 2 * y], y_std=[None, 0.5])
    assert len(ax.collections) == 1  # only the second curve has a band
    with pytest.raises(ValueError, match="y_std"):
        pk.plot_shadow_curve(y, y_std=np.ones(5))


def test_shadow_multiple_curves_labels_and_per_series_x(rng):
    a, b = rng.normal(size=(3, 10)), rng.normal(size=(3, 20))
    ax = pk.plot_shadow_curve(
        [a, b], x=[np.arange(10), np.arange(20) * 0.5], legend_labels=["A", "B"]
    )
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    assert ax.lines[1].get_xdata()[-1] == pytest.approx(9.5)
    with pytest.raises(ValueError, match="axis=1"):
        pk.plot_shadow_curve(np.zeros((10, 4)), x=np.arange(10))


def test_shadow_single_curve_legend_only_with_labels(rng):
    assert pk.plot_shadow_curve(rng.normal(size=(3, 5))).get_legend() is None
    assert pk.plot_shadow_curve(rng.normal(size=(3, 5)), labels="Agent").get_legend() is not None


def test_shadow_tick_labels():
    y = np.random.default_rng(1).normal(size=(3, 50))
    ax = pk.plot_shadow_curve(y, x_tick_labels={0: "start", 49: "end"})
    np.testing.assert_allclose(ax.get_xticks(), [0, 49])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["start", "end"]
    ax = pk.plot_shadow_curve(y[:, :5], x_tick_labels=list("abcde"))
    np.testing.assert_allclose(ax.get_xticks(), np.arange(5))
    ax = pk.plot_shadow_curve(
        y, x_tick_labels=["Ep 0", "Ep 10", "Ep 20"], y_tick_labels=["lo", "hi"]
    )
    np.testing.assert_allclose(ax.get_xticks(), [0, 24.5, 49])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["lo", "hi"]
    ax = pk.plot_shadow_curve(y, x_tick_labels="middle")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["middle"]


def test_shadow_mathtext_labels_render():
    ax = pk.plot_shadow_curve(
        [np.ones((2, 5)), np.zeros((2, 5))],
        labels=[r"$\alpha$-DQN", r"$\beta$-PPO"],
        xlabel=r"Steps ($\times 10^3$)",
    )
    ax.figure.canvas.draw()


def test_learning_curves_ragged_runs(rng):
    runs = {"A": [rng.normal(size=10), rng.normal(size=7)], "B": rng.normal(size=(3, 10))}
    ax = pk.plot_learning_curves(runs, x=np.arange(10) * 100, band="std")
    assert len(ax.lines) == 2 and len(ax.collections) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    padded = np.full((2, 10), np.nan)
    padded[0], padded[1, :7] = runs["A"][0], runs["A"][1]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.nanmean(padded, axis=0))
    assert ax.get_xlabel() == "Step" and ax.get_ylabel() == "Return"


def test_learning_curves_default_ci95_and_x_mapping(rng):
    data = rng.normal(size=(5, 8))
    ax = pk.plot_learning_curves({"run": data}, x={"run": np.arange(8) * 2})
    half = Z95 * data.std(axis=0, ddof=1) / np.sqrt(5)
    lower, upper = band_edges(ax, np.arange(8) * 2)
    np.testing.assert_allclose(upper - lower, 2 * half)
    with pytest.raises(ValueError, match="no entry"):
        pk.plot_learning_curves({"run": data}, x={"other": np.arange(8)})
    with pytest.raises(TypeError, match="mapping"):
        pk.plot_learning_curves([data])
    with pytest.raises(ValueError, match="empty"):
        pk.plot_learning_curves({})


# ---------------------------------------------------------------------------
# plot_bar / plot_histogram
# ---------------------------------------------------------------------------


def test_grouped_bar_geometry():
    heights = np.arange(12.0).reshape(3, 4) + 1
    ax = pk.plot_bar(["a", "b", "c", "d"], heights, width=0.9)
    assert len(ax.containers) == 3
    bar_w = 0.9 / 3
    for i, container in enumerate(ax.containers):
        offset = (i - 1) * bar_w
        for j, rect in enumerate(container.patches):
            assert rect.get_width() == pytest.approx(bar_w)
            assert rect.get_x() + rect.get_width() / 2 == pytest.approx(j + offset)
            assert rect.get_height() == pytest.approx(heights[i, j])
    np.testing.assert_allclose(ax.get_xticks(), np.arange(4))
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c", "d"]
    assert ax.get_legend() is not None


def test_bar_error_bars_and_value_labels():
    ax = pk.plot_bar(
        ["a", "b"], [[1.0, 2.0], [3.0, 4.0]], yerr=[[0.1, 0.2], [0.3, 0.4]], value_labels=".1f"
    )
    segments = bar_containers(ax)[1].errorbar.lines[2][0].get_segments()
    lows = [seg[0][1] for seg in segments]
    highs = [seg[1][1] for seg in segments]
    np.testing.assert_allclose(lows, [2.7, 3.6])
    np.testing.assert_allclose(highs, [3.3, 4.4])
    assert sorted(t.get_text() for t in ax.texts) == ["1.0", "2.0", "3.0", "4.0"]


def test_bar_asymmetric_errors_and_scalar_error():
    ax = pk.plot_bar(["a", "b"], [1.0, 2.0], yerr=np.array([[0.1, 0.2], [0.5, 0.6]]))
    segments = bar_containers(ax)[0].errorbar.lines[2][0].get_segments()
    np.testing.assert_allclose([s[0][1] for s in segments], [0.9, 1.8])
    np.testing.assert_allclose([s[1][1] for s in segments], [1.5, 2.6])
    ax = pk.plot_bar(["a", "b"], [[1, 2], [3, 4]], yerr=0.5)
    assert all(c.errorbar is not None for c in bar_containers(ax))


def test_bar_horizontal_and_per_bar_colors():
    ax = pk.plot_bar(["a", "b", "c"], [3, 1, 2], horizontal=True, colors=["r", "g", "b"])
    rects = ax.patches
    assert [r.get_width() for r in rects] == [3, 1, 2]
    assert all(r.get_height() == pytest.approx(0.8) for r in rects)
    assert ax.yaxis_inverted()
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    assert mcolors.same_color(rects[1].get_facecolor()[:3], "g")


def test_bar_shape_errors():
    with pytest.raises(ValueError, match="transpose"):
        pk.plot_bar(["a", "b", "c"], np.ones((3, 2)))
    with pytest.raises(ValueError, match="same number of categories"):
        pk.plot_bar(None, [[1, 2], [1, 2, 3]])
    with pytest.raises(ValueError, match="width"):
        pk.plot_bar(None, [1, 2], width=0)
    with pytest.raises(ValueError, match="yerr"):
        pk.plot_bar(None, [[1, 2], [1, 2]], yerr=[[1, 2], [1, 2], [1, 2]])


def test_histogram_common_bins_and_nan(rng):
    a = np.append(rng.normal(size=200), np.nan)
    b = rng.normal(2, 1, size=100)
    ax = pk.plot_histogram([a, b], bins=10, density=True)
    assert len(ax.patches) == 2
    xs0 = np.unique(ax.patches[0].get_path().vertices[:, 0])
    xs1 = np.unique(ax.patches[1].get_path().vertices[:, 0])
    np.testing.assert_allclose(xs0, xs1)  # shared bin edges
    assert ax.get_ylabel() == "Density"
    assert pk.plot_histogram(b).get_ylabel() == "Count"
    with pytest.raises(ValueError, match="no finite"):
        pk.plot_histogram([np.nan, np.nan])


# ---------------------------------------------------------------------------
# plot_heatmap / plot_gray_scale
# ---------------------------------------------------------------------------


def test_heatmap_annotations_and_text_contrast():
    data = np.array([[0.0, 1.0], [0.5, np.nan]])
    ax = pk.plot_heatmap(data, annot=True, cmap="viridis", fmt=".1f")
    texts = {(t.get_position()): t for t in ax.texts}
    assert len(ax.texts) == 3  # the NaN cell is not annotated
    assert texts[(0, 0)].get_text() == "0.0" and texts[(0, 0)].get_color() == "white"
    assert texts[(1, 0)].get_text() == "1.0" and texts[(1, 0)].get_color() == "black"
    assert ax.images[0].get_array().mask[1, 1]


def test_heatmap_colorbar_and_labels():
    data = np.arange(6.0).reshape(2, 3)
    ax = pk.plot_heatmap(data, xlabels=["a", "b", "c"], ylabels=["r0", "r1"], cbar_label="value")
    colorbar = ax.images[0].colorbar
    assert colorbar is not None and colorbar.ax.get_ylabel() == "value"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["r0", "r1"]
    ax = pk.plot_heatmap(data, cbar=False)
    assert ax.images[0].colorbar is None and len(ax.figure.axes) == 1
    with pytest.raises(ValueError, match="xlabels has 2 entries but data has 3 columns"):
        pk.plot_heatmap(data, xlabels=["a", "b"])


def test_heatmap_norms():
    data = np.array([[-1.0, 0.5], [2.0, 0.0]])
    norm = pk.plot_heatmap(data, center=0, cmap="RdBu_r").images[0].norm
    assert isinstance(norm, mcolors.TwoSlopeNorm)
    assert (norm.vmin, norm.vcenter, norm.vmax) == (-2.0, 0.0, 2.0)
    norm = pk.plot_heatmap(data, vmin=-5, vmax=5).images[0].norm
    assert (norm.vmin, norm.vmax) == (-5, 5)
    with pytest.raises(ValueError, match="center"):
        pk.plot_heatmap(data, center=10, vmin=0, vmax=5)


def test_heatmap_seaborn_kwargs_and_mask():
    data = np.arange(9.0).reshape(3, 3)
    mask = np.eye(3, dtype=bool)
    ax = pk.plot_heatmap(
        data,
        annot=True,
        fmt="{:.2f}",
        annot_kws={"fontsize": 6, "color": "red"},
        square=True,
        linewidths=0,
        mask=mask,
        xticklabels=["x0", "x1", "x2"],
        cbar_kws={"label": "from kws"},
    )
    assert len(ax.texts) == 6 and all(t.get_color() == "red" for t in ax.texts)
    assert ax.texts[0].get_text() == "1.00"
    assert ax.get_aspect() == 1.0
    assert not any(line.get_visible() for line in ax.xaxis.get_gridlines())
    assert [t.get_text() for t in ax.get_xticklabels()] == ["x0", "x1", "x2"]
    assert ax.images[0].colorbar.ax.get_ylabel() == "from kws"
    with pytest.warns(UserWarning, match="shading"):
        pk.plot_heatmap(data, shading="auto")


def test_heatmap_nan_color_does_not_touch_registered_colormap():
    ax = pk.plot_heatmap(np.array([[1.0, np.nan]]), nan_color="red", cmap="viridis")
    assert mcolors.same_color(ax.images[0].get_cmap().get_bad(), "red")
    assert matplotlib.colormaps["viridis"].get_bad()[3] == 0.0


def test_heatmap_integer_default_format_and_errors():
    ax = pk.plot_heatmap(np.array([[85, 15], [10, 190]]), annot=True)
    assert sorted(t.get_text() for t in ax.texts) == ["10", "15", "190", "85"]
    with pytest.raises(ValueError, match="no finite"):
        pk.plot_heatmap(np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="2-D"):
        pk.plot_heatmap(np.zeros((2, 2, 2)))


def test_gray_scale_delegates_to_heatmap():
    ax = pk.plot_gray_scale(np.eye(4), title="eye")
    assert ax.images[0].get_cmap().name == "gray"
    assert ax.get_title() == "eye"
    ax = pk.plot_gray_scale(np.eye(4), cmap="magma")
    assert ax.images[0].get_cmap().name == "magma"


# ---------------------------------------------------------------------------
# Styles and palettes
# ---------------------------------------------------------------------------

PLOT_CALLS = {
    "shadow": lambda **kw: pk.plot_shadow_curve(np.ones((3, 5)), smoothing=2, **kw),
    "learning": lambda **kw: pk.plot_learning_curves({"a": np.ones((3, 5))}, **kw),
    "line": lambda **kw: pk.plot_line(None, [1, 2, 3], **kw),
    "scatter": lambda **kw: pk.plot_scatter([1, 2], [3, 4], **kw),
    "bar": lambda **kw: pk.plot_bar(
        ["a", "b"], [[1, 2], [2, 3]], yerr=0.1, value_labels=True, **kw
    ),
    "histogram": lambda **kw: pk.plot_histogram([1.0, 2.0, 2.5], **kw),
    "heatmap": lambda **kw: pk.plot_heatmap(np.eye(3), annot=True, **kw),
    "grayscale": lambda **kw: pk.plot_gray_scale(np.eye(3), **kw),
}


@pytest.mark.parametrize("style", ["research", "presentation", "minimal", "default"])
@pytest.mark.parametrize("name", list(PLOT_CALLS))
def test_plot_functions_leave_rcparams_unchanged(name, style):
    before = rc_snapshot()
    ax = PLOT_CALLS[name](style=style)
    ax.figure.canvas.draw()
    after = rc_snapshot()
    assert {k for k in before if before[k] != after[k]} == set()


@pytest.mark.parametrize("name", list(PLOT_CALLS))
def test_ax_reuse_creates_no_figure(name):
    fig, ax = plt.subplots()
    n_figures = len(plt.get_fignums())
    returned = PLOT_CALLS[name](ax=ax)
    assert returned is ax
    assert len(plt.get_fignums()) == n_figures


def test_research_style_applied_locally_to_new_figure():
    ax = pk.plot_line(None, [1, 2, 3], title="t", style="research")
    ax.figure.canvas.draw()  # draw outside the style context
    assert matplotlib.rcParams["font.family"] != ["serif"]
    assert ax.title.get_fontfamily() == ["serif"]
    assert ax.get_xticklabels()[0].get_fontfamily() == ["serif"]
    assert not ax.spines["top"].get_visible()
    assert ax.lines[0].get_linewidth() == 2.0


def test_style_restyles_existing_axes_unless_default():
    fig, (ax1, ax2) = plt.subplots(1, 2)  # created with the global (default) style
    pk.plot_line(None, [1, 2, 3], ax=ax1, xlabel="x", style="research")
    pk.plot_line(None, [1, 2, 3], ax=ax2, xlabel="x", style="default")
    fig.canvas.draw()
    assert not ax1.spines["top"].get_visible()
    assert ax1.xaxis.label.get_fontfamily() == ["serif"]
    assert ax1.get_yticklabels()[-1].get_fontfamily() == ["serif"]
    assert ax2.spines["top"].get_visible()
    assert ax2.xaxis.label.get_fontfamily() == matplotlib.rcParams["font.family"]


def test_style_context_presets_and_restore():
    before = rc_snapshot()
    with pk.style_context("presentation"):
        assert matplotlib.rcParams["font.size"] == 16
    with pk.style_context({"lines.linewidth": 7.0}):
        assert matplotlib.rcParams["lines.linewidth"] == 7.0
    with pk.style_context("ggplot"):  # matplotlib style names work too
        pass
    with pk.style_context("default"):
        matplotlib.rcParams["font.size"] = 3  # changes inside are reverted
    assert rc_snapshot() == before
    with pytest.raises(ValueError, match="unknown style"):
        with pk.style_context("no-such-style"):
            pass
    with pytest.raises(TypeError):
        pk.STYLE_PRESETS["research"]["font.size"] = 1


def test_set_research_style_reset_round_trip():
    matplotlib.rcParams["lines.linewidth"] = 3.3  # user customisation survives the round trip
    before = rc_snapshot()
    pk.set_research_style()
    pk.set_research_style()  # idempotent; the first snapshot is kept
    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert matplotlib.rcParams["savefig.dpi"] == 300
    pk.reset_style()
    assert rc_snapshot() == before
    pk.reset_style()  # nothing saved: back to matplotlib defaults
    assert matplotlib.rcParams["lines.linewidth"] == matplotlib.rcParamsDefault["lines.linewidth"]


def test_get_palette():
    assert pk.get_palette() == pk.RESEARCH_COLOR_LIST
    assert pk.get_palette("colorblind", 3) == ["#0072B2", "#E69F00", "#009E73"]
    assert len(pk.get_palette("okabe_ito", 10)) == 10
    assert pk.get_palette("okabe-ito", 9)[8] == pk.get_palette("okabe_ito")[0]
    assert len(pk.get_palette("tab20")) == 20
    viridis = pk.get_palette("viridis", 4)
    assert len(set(viridis)) == 4 and all(c.startswith("#") for c in viridis)
    with pytest.raises(ValueError, match="unknown palette"):
        pk.get_palette("nope")
    with pytest.raises(ValueError, match="positive"):
        pk.get_palette("research", 0)
    assert list(pk.RESEARCH_COLORS.values()) == pk.RESEARCH_COLOR_LIST


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


def test_save_figure_multiple_formats(tmp_path):
    ax = pk.plot_line(None, [1, 2, 3])
    paths = pk.save_figure(ax, tmp_path / "results" / "nested" / "curve", formats=("pdf", "png"))
    assert paths == [tmp_path / "results/nested/curve.pdf", tmp_path / "results/nested/curve.png"]
    assert paths[0].read_bytes().startswith(b"%PDF")
    assert paths[1].read_bytes().startswith(b"\x89PNG")


def test_save_figure_suffix_handling(tmp_path):
    fig, axes = plt.subplots(1, 2)
    assert pk.save_figure(fig, tmp_path / "a.svg") == [tmp_path / "a.svg"]
    assert pk.save_figure(axes, tmp_path / "b.png", formats="pdf") == [tmp_path / "b.pdf"]
    assert pk.save_figure(fig, tmp_path / "run.v2", formats=["png"]) == [tmp_path / "run.v2.png"]
    assert (tmp_path / "run.v2.png").exists()
    with pytest.raises(ValueError, match="unsupported"):
        pk.save_figure(fig, tmp_path / "c", formats="docx")
    with pytest.raises(TypeError):
        pk.save_figure("not a figure", tmp_path / "d.png")


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------


def test_smoothing_helpers_match_references(rng):
    values = rng.normal(size=40)
    values[[3, 4, 20]] = np.nan
    values[0] = np.nan
    np.testing.assert_allclose(moving_average(values, 5), ref_moving_average(values, 5))
    np.testing.assert_allclose(ema(values, 0.7), ref_ema(values, 0.7), equal_nan=True)
    x = rng.normal(size=(3, 30))
    np.testing.assert_allclose(_ema_filter(x, 0.6), _ema_filter_loop(x, 0.6))


def test_smoothing_matches_pandas(rng):
    pd = pytest.importorskip("pandas")
    values = rng.normal(size=30)
    values[[5, 6]] = np.nan
    series = pd.Series(values)
    np.testing.assert_allclose(
        moving_average(values, 4), series.rolling(4, min_periods=1).mean().to_numpy()
    )
    np.testing.assert_allclose(ema(values, 0.8), series.ewm(alpha=0.2).mean().to_numpy())


# ---------------------------------------------------------------------------
# Gallery CLI and example
# ---------------------------------------------------------------------------


def test_gallery_cli_all(tmp_path, capsys):
    assert main(["--demo", "all", "--save", tmp_path, "--dpi", "40"]) == 0
    assert (tmp_path / "plotkit_all.png").stat().st_size > 0
    assert "plotkit_all.png" in capsys.readouterr().out
    assert plt.get_fignums() == []


@pytest.mark.parametrize("demo", list(DEMOS))
def test_gallery_cli_single_demos(tmp_path, demo):
    assert main(["--demo", demo, "--style", "minimal", "--save", str(tmp_path), "--dpi", "40"]) == 0
    assert (tmp_path / f"plotkit_{demo}.png").exists()


def test_gallery_cli_non_interactive_backend_saves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--demo", "line", "--format", "png", "pdf", "--dpi", "40"]) == 0
    assert (tmp_path / "renders" / "plotkit" / "plotkit_line.png").exists()
    assert (tmp_path / "renders" / "plotkit" / "plotkit_line.pdf").exists()


def test_gallery_example_script(tmp_path):
    path = REPO_ROOT / "examples" / "plotkit_gallery.py"
    spec = importlib.util.spec_from_file_location("plotkit_gallery_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.main(
            ["--save", str(tmp_path / "gallery"), "--seeds", "3", "--steps", "40", "--dpi", "40"]
        )
        == 0
    )
    assert (tmp_path / "gallery.pdf").exists() and (tmp_path / "gallery.png").exists()
