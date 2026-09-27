# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-27

First release as a single, standard Python package. The two former projects
(`envlib_project/env_lib` and `toolkit_project/toolkit`) are now one
distribution, `my-tool-box`, that still provides the import packages
`env_lib` and `toolkit`.

### Packaging and repository layout

- Single `pyproject.toml` (setuptools, `src/` layout) replaces the two
  `setup.py` files and the per-environment `setup.py`, `MANIFEST.in`,
  `requirements.txt` and `install.sh` files.
- Heavy dependencies are optional extras: `pistonball` (pygame, pymunk),
  `torch`, `video` (imageio, imageio-ffmpeg), `all`, `dev` (pytest, ruff).
  TensorFlow, seaborn, pandas and scikit-learn are no longer required.
- `import env_lib` and `import toolkit` are lazy: environment classes and
  toolkit subpackages are imported on first use, so importing the package no
  longer pulls in torch, pygame, pymunk or tkinter.
- Console scripts: `ajlatt-build-map`, `ajlatt-plot-map`, `plotkit-gallery`.
- Removed build artefacts and stale files: `*.egg-info`, `build/`,
  `__pycache__`, generated PNGs, a parameter dump, `cleanup_test_images.sh`,
  `OPTIMIZATION_SUMMARY.md` (it described optimisations that did not exist),
  the TkAgg test script and notes, and obsolete map data (a `-backup` header,
  a header without grid, an unused motion-primitive file).
- Tests moved to `tests/` (pytest, 792 tests), examples to `examples/`,
  benchmarks to `benchmarks/`, the LaTeX manual to `docs/manual/`.
- Continuous integration workflow (lint, tests on Python 3.9 to 3.12, example
  smoke tests) in `.github/workflows/ci.yml`.
- Code base formatted and linted with ruff; no emojis or decorative symbols in
  code, output or documentation.

### Environments (`env_lib`)

#### Common

- All environments are registered centrally in `env_lib.registration`;
  `env_lib.make(id)` and `env_lib.list_envs()` were added.
- Registrations no longer add a `TimeLimit` wrapper. Every environment enforces
  its own episode limit (`max_steps`, `max_iter`, `max_cycles`,
  `max_episode_steps`), so a user-supplied limit is no longer silently
  overridden by the registry.
- Gymnasium API everywhere: `reset(*, seed=None, options=None) -> (obs, info)`
  and `step() -> (obs, reward, terminated, truncated, info)`, with episode
  limits reported as `truncated`. Seeding goes through `self.np_random`; no
  environment touches the global NumPy or torch random state any more.
- Every environment reports per-agent rewards in `info["agent_rewards"]`.
- Using an environment before `reset()` raises `env_lib.ResetNeededError`,
  which subclasses both `RuntimeError` and `gymnasium.error.ResetNeeded`.
- Observations are `float32` and contained in `observation_space`.
- New shared rendering layer (`env_lib.utils`): dark and light themes
  (`set_theme`), persistent-artist dashboards with blitting, headless
  `rgb_array` rendering, and `record_episode` / `save_animation` for GIF and
  MP4 export. `"human"` mode renders automatically in `reset()` and `step()`,
  following the Gymnasium convention.
- Rendering never runs unless requested (the old Pistonball drew its screen on
  every step).

#### Kuramoto oscillators (`kos_env`)

- NumPy dynamics vectorised (the O(N^2) Python loop is gone): 20x faster at
  N = 50, 65x faster with RK4.
- Fixed a sign error in the PyTorch backend that made the coupling repulsive
  (`sin(theta_i - theta_j)` instead of `sin(theta_j - theta_i)`). Both backends
  now produce the same trajectories; normalisation by N is opt-in
  (`normalize_coupling=True`) for both.
- The coupling now uses the full matrix; the NumPy loop used to skip entries
  `K_ij <= 0`. Default settings are unaffected.
- Removed `np.random.seed(42)` and `torch.manual_seed(42)` side effects; the
  random topology uses a local generator (`topology_seed`, same topology as
  before).
- `coupling_mode="constant"` without a matrix now uses
  `coupling_strength * adjacency`, which makes the registered `*-Constant-v0`
  ids usable (they raised before).
- Synchronisation sets `terminated`, the step limit sets `truncated`
  (previously both were reported as `terminated`); `sync_threshold` and
  `sync_bonus` are configurable.
- The torch backend accepts `device="auto"`; its phase coherence and bonus
  rules now match the NumPy backend. Tensor actions that require gradients
  are detached before they enter the dynamics.
- Rendering rewritten: phase portrait with coupling chords, fading trails and
  the order-parameter vector, synchronisation time series and phase raster.
  The previous `rgb_array` path crashed on matplotlib 3.10 and later.

#### LineMsg (`linemsg_env`)

- Vectorised action decoding and observations; seeded trajectories are
  bit-identical to the previous version.
- Actions may be an integer in `Discrete(2**N)` or an array of `N` bits;
  `action_space_type="multibinary"` supports more than 62 agents.
- `step()` before `reset()` raises instead of resetting implicitly.
- At least two agents are required (`num_agents=1` used to fail inside
  `step()`).
- The previous text output is available as `render_mode="ansi"`;
  `"human"` and `"rgb_array"` show a new dashboard (agent line, space-time
  raster of states, team reward).

#### WirelessComm (`wireless_comm_env`)

- `step()` fully vectorised (up to 37x faster on large grids) while keeping
  seeded trajectories bit-identical to the previous version.
- Fixed: `self.actions` was never updated, so rendering showed no actions.
  Per-agent outcomes (idle, success, collision, lost) are reported in `info`.
- Argument and action validation (fractional or out-of-range choices raise);
  `"ansi"` text mode; new dashboard (agent
  grid with packet queues, access-point load, transmissions coloured by
  outcome, throughput and return traces).

#### Pistonball (`pistonball_env`)

- 16x to 30x faster without rendering: no pygame work unless rendering is
  requested, vectorised observations and rewards, lazy pygame initialisation.
- Discrete actions use `MultiDiscrete([3] * n)` (same array format as before).
- Actions are validated (shape, range, NaN, and non-integer values in
  discrete mode); `step()` before `reset()` raises.
- Closing one of several open environments no longer shuts down the pygame
  display of the others.
- New vector renderer (shaded pistons, highlighted kappa-hop observers, ball
  with motion trail, heads-up display, progress bar); the PNG sprites were
  removed. Compatible with pymunk 6 and 7.

#### Consensus / Formation (`consensus_env`, new)

- New networked multi-agent control benchmark: agents on a graph (ring, line,
  star, complete, Erdos-Renyi, proximity) must rendezvous or form a shape
  (circle, line, grid, wedge) with single- or double-integrator dynamics.
  Includes a distributed Laplacian baseline controller. Registered as
  `Consensus-v0` and `Formation-v0`.

#### AJLATT (`ajlatt_env`)

- Rewritten as a Gymnasium environment, `env_lib.AJLATTEnv`, configured by the
  `AJLATTConfig` dataclass. The previous implementation called
  `argparse.parse_args()` inside `make()`, so creating the environment from any
  script with its own command-line arguments failed.
- 10x faster (about 9 ms per step instead of 90 ms on `obstacles04`):
  - exact batched Bresenham ray casting (the results match the old loop cell
    for cell; closest-obstacle queries are 7x to 19x faster);
  - covariance intersection solved by an active-set Newton method with
    analytic gradient and Hessian instead of SLSQP with finite differences.
    Its objective is never worse than SLSQP's; `ci_solver="slsqp"` reproduces
    the previous solver, and with it trajectories match the previous
    implementation to about 1e-6.
- Fixed a memory leak: covariance traces were appended to lists that were never
  cleared across episodes. Per-episode traces are available through
  `env.episode_statistics()`.
- Fixed target policies stored as class attributes (shared between
  environments and never reset between episodes) and the `input()` prompt for
  unknown maps; policies are now per-instance (`target_policy` option).
- Fixed dynamic maps (wrong constructor call, wrong obstacle directory, never
  regenerated); they now resample on every `reset()` and no longer need
  scikit-image.
- Three bundled maps were one row short of their header; they are padded with
  a wall row (user maps with the same problem are padded with a warning).
- Actions containing NaN or infinity raise `ValueError`; `render()` before
  `reset()` raises `RuntimeError`.
- A map counts as empty only when its `.cfg` file is empty; a missing `.cfg`
  raises `FileNotFoundError` unless the map name contains `empty`.
- Joint spaces: `action_space` is `(num_robots, 2)` and `observation_space`
  `(num_robots, obs_dim)`; the per-robot spaces are `single_action_space` and
  `single_observation_space`. Rewards and `terminated` are per-robot arrays;
  `TeamRewardWrapper` provides scalar values.
- New renderer: occupancy map, robots with covariance ellipses and sensing
  sectors, trails, communication links, active measurements, every robot's
  target belief, and covariance-trace panels (about 25 ms per frame instead of
  about 110 ms).
- Map tooling consolidated into `env_lib.ajlatt_env.maps` (`builder`,
  `plotting`, `available_maps`, `load_grid_map`). The builder now writes grids
  in the orientation the environment reads (the old builder produced
  transposed maps) and embeds its specification in the generated header.

### Toolkit (`toolkit`)

#### plotkit

- No global side effects: plot functions no longer call
  `plt.style.use("default")` or mutate `rcParams`; `set_research_style()` is
  an explicit opt-in with a matching `reset_style()`.
- Fixed `plot_line(x, [y1, y2])` plotting only the first series, lists of
  scalars being treated as several series, and overlapping bars (bars are now
  grouped side by side, with optional error bars and value labels).
- `plot_heatmap` no longer depends on seaborn.
- `plot_shadow_curve` gained NaN-robust statistics, `band` (`std`, `sem`,
  `ci95`, `minmax`) and `smoothing` (moving average or EMA).
- New: `plot_learning_curves`, `plot_histogram`, `save_figure`, style presets,
  the Okabe-Ito colour-blind-safe palette, and a gallery command-line tool.

#### neural_toolkit

- Fixed MLP stacks that applied no activation between hidden layers, so deep
  "MLPs" were almost linear. This affects every policy, value, Q, encoder and
  decoder network. Checkpoints from earlier versions are not compatible.
- Fixed the positional encoding, which indexed by batch position instead of
  sequence position.
- Fixed non-ReLU CNNs (crashed), the dueling Q-network head, Q-networks
  ignoring the `action` argument, `CNNEncoder` on 64x64 inputs, `CNNDecoder`
  kernel indexing, and bidirectional RNN feature extraction.
- Unknown activation names now raise; the `device` argument is honoured.
- Tabular tools use a per-table random generator; UCB, softmax and
  `PolicyTable.set_value` were fixed.

#### parakit

- `import toolkit.parakit` no longer requires tkinter; the Tk editor is
  imported lazily.
- Headless API: `get_parameters`, `save_parameters`, `load_parameters`,
  `apply_parameters`, `parse_value`.
- Boolean strings, lists, `None` and `store_true` options are parsed
  correctly, and `tune()` now applies the edited values to the parser as
  documented.
- Removed `logging.basicConfig` from library code.

### Migration notes

- Install with `pip install -e ".[all]"` from the repository root instead of
  running `pip install -e .` in the two project folders.
- `env.reset()` returns `(observation, info)` for every environment, including
  AJLATT.
- AJLATT: prefer `env_lib.AJLATTEnv(map_name=..., num_robots=...)`. The old
  factory `env_lib.ajlatt_env(num_Robot=..., T_steps=...)` still works and
  translates the legacy names with a `DeprecationWarning`. Rendering is off by
  default; pass `render_mode="rgb_array"` or `"human"`. Per-robot spaces are
  `env.single_observation_space` / `env.single_action_space`. The
  observation's target-velocity entries now contain the commanded target
  velocity instead of the configured constant. The legacy `render=True`
  keyword still selects `render_mode="human"` (with a `DeprecationWarning`),
  and `env_lib.ajlatt_env.make(...)` remains available as an alias of the
  factory.
- LineMsg needs at least two agents; WirelessComm and discrete Pistonball
  reject fractional action values instead of truncating them.
- Training argparse options that used to live in the AJLATT parameter module
  (PPO, LLM and debug settings) were removed from the environment package; use
  `AJLATTConfig.add_arguments(parser)` to expose the environment options in a
  training script.
- The `AJLATT-v1` id was removed (it was an unusable alias of `AJLATT-v0`).
- Pistonball in discrete mode: the action space is now `MultiDiscrete`.
- neural_toolkit: retrain models; the architectures changed because of the
  activation fix.
