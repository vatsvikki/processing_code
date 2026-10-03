"""Notebook tools: whole marimo notebooks (in ../notebooks/) offered in the 🛠 Tools menu of the app.

Each opens in its own browser tab, where the app embeds and runs the notebook as it is (all its own buttons, sliders,
state and outputs) - kind="notebook"; run() only returns the notebook's path, the app does the embedding.
Nothing here is used by, or uses, the Flow or any other tool.
"""
from __future__ import annotations

from pathlib import Path

from .registry import register

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


@register("Shot Geometry QC", "Direct-arrival linear-velocity overlay per shot, directional-fix toggles and grid "
          "shifts, whole-survey audit, automatic fix search, geometry maps, and a corrected copy of the SEG-Y "
          "(shot_geometry_qc_marimo.py).", order=20, kind="notebook", category="Tools")
def shot_geometry_qc(path: str = "", **_):
    return str(NOTEBOOK_DIR / "shot_geometry_qc_marimo.py")
