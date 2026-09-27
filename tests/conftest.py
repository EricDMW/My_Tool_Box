"""Shared pytest configuration.

Rendering tests run headless: matplotlib uses the non-interactive Agg backend
and pygame uses SDL's dummy video driver.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib

matplotlib.use("Agg", force=True)
