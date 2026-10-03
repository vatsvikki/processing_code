"""The velocity for NMO, shared by the "NMO Correction" and "CDP Stack" steps: from a velocity-model SEG-Y (the trace
nearest each CDP's position, as in the brute-stack notebook), from one velocity function (time / Vrms table), or none
(raw stack). No UI here.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import os

import numpy as np

from . import grid as grid_mod, nmo

SOURCES = ["Velocity model SEG-Y", "Velocity function (table)", "No NMO (raw stack)"]
MODEL, FUNCTION, NONE = SOURCES


def _blank(rows) -> bool:
    return not any(any(str(v).strip() for v in (r.values() if isinstance(r, dict) else r)) for r in (rows or []))


def _transform(rows, what):
    try:
        parsed = grid_mod.parse_corners(rows)
    except ValueError as e:
        return None, f"{what} corner table: {e}"
    tr = nmo.fit_corners(parsed)
    return tr, ("" if tr is not None else f"{what} corner points: enter at least 2 rows with distinct IL values and "
                                          "distinct XL values, each with its X and Y")


@dataclass
class VelocitySetup:
    source: str
    full_t: np.ndarray
    unit: str                                    # output velocity unit: offsets are converted to match
    error: str = ""                              # set: no velocity can be used (the message says why)
    notes: list = field(default_factory=list)
    data_transform: dict | None = None
    model_transform: dict | None = None
    scan: dict | None = None
    model_path: str = ""
    settings: dict = field(default_factory=dict)  # the model conversion settings (model_velocity_for keywords)
    function: np.ndarray | None = None
    function_rows: list = field(default_factory=list)
    signature: tuple = ()                        # what a stack computed with this velocity depends on
    _cache: OrderedDict = field(default_factory=OrderedDict)

    @property
    def nmo(self) -> bool:
        return self.source != NONE and not self.error

    def nearest_traces(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The model trace nearest each position (model source only)."""
        return nmo.nearest_model_traces(self.scan, self.model_transform, xs, ys)

    def for_position(self, x: float, y: float):
        """(Vrms on full_t, interval velocity on full_t, model match info or None) at a real position."""
        if self.source == FUNCTION:
            return self.function, nmo.interval_from_rms(self.full_t, self.function), None
        vel, info = nmo.model_velocity_for(self.model_path, self.scan, self.model_transform, x, y, self.full_t,
                                           **self.settings)
        return vel, info["interval_of_t0"], info

    def model_trace_velocity(self, trace_idx: int, max_cache: int = 20000):
        """(Vrms, interval, last time of the model in s) of one model trace on full_t, cached (neighbouring CDPs share
        the same model trace)."""
        hit = self._cache.get(trace_idx)
        if hit is not None:
            self._cache.move_to_end(trace_idx)
            return hit
        vel, vint, info = nmo.model_trace_velocity(self.model_path, self.scan, int(trace_idx), self.full_t,
                                                   **self.settings)
        hit = (vel.astype(np.float32), vint.astype(np.float32), float(info["model_t"][-1]))
        self._cache[trace_idx] = hit
        if len(self._cache) > max_cache:
            self._cache.popitem(last=False)
        return hit


def build(path: str, full_t: np.ndarray, *, source: str, vel_model_path: str = "", velocity_model_type: str = "RMS velocity",
          model_velocity_units: str = "ft/s", model_domain: str = "TWTT (time)", model_sample_interval: float = 0.0,
          output_velocity_units: str = "ft/s", data_corners=None, model_corners=None, vel_function=None) -> VelocitySetup:
    """Check the settings and prepare the velocity. The data's corner points are prepared for every source (they give
    the CDPs their data IL / XL)."""
    vs = VelocitySetup(source, full_t, output_velocity_units if source != NONE else "ft/s")
    if _blank(data_corners):
        data_corners = grid_mod.corner_table(path)                   # (a table never filled in: this file's grid)
    vs.data_transform, why_d = _transform(data_corners, "Data")
    if vs.data_transform is None:
        vs.notes.append("⚠ **No survey grid** - the data's IL / XL (in the sections, the overlay and the CDP lists) need "
                        "it: fill in the **Survey grid** of the 📁 Data card (at least 2 corners with distinct IL and "
                        "distinct XL, each with X and Y).")
    if source == NONE:
        vs.signature = ("none",)
        return vs
    if source == FUNCTION:
        try:
            vs.function, vs.function_rows = nmo.velocity_function(vel_function or nmo.DEFAULT_VELFN, full_t)
        except ValueError as e:
            vs.error = f"**Velocity function not usable** - {e}."
            return vs
        vs.signature = ("function", tuple(vs.function_rows), output_velocity_units)
        return vs

    path_m = str(vel_model_path or "").strip()
    if not path_m:
        vs.error = "Enter the **velocity model SEG-Y file** to NMO-correct (or choose a velocity function)."
        return vs
    path_m = os.path.expanduser(path_m)
    if not os.path.exists(path_m):
        vs.error = f"**Velocity model not found:** `{path_m}`"
        return vs
    vs.model_path, vs.scan = path_m, nmo.model_scan(path_m)
    rows_m = model_corners
    if _blank(rows_m):
        from_hdr = nmo.model_corners_from_header(path_m)
        rows_m = grid_mod.table_rows(from_hdr)
        if from_hdr:
            vs.notes.append(f"Model corner points read from the model's text header ({len(from_hdr)} corners) - type your "
                            "own in *Model grid corner points* to use others.")
    vs.model_transform, why_m = _transform(rows_m, "Model")
    if vs.model_transform is None:
        vs.error = f"**Model grid not registered** - {why_m}."
        return vs
    dt_s = model_sample_interval
    if dt_s <= 0:
        # 0 = the model file's own sample interval (its dt header word): ms for a time model; for a depth model the same
        # word holds the depth step (e.g. 10000 for 10 ft, as in this survey's velocity model)
        if vs.scan["dt_us"] <= 0:
            vs.error = "The model file's header has no sample interval - enter it in *4. Model sample interval*."
            return vs
        dt_s = vs.scan["dt_us"] / 1000.0
        vs.notes.append(f"Model sample interval taken from the model file's header: {dt_s:g} "
                        + ("ms." if model_domain != "Depth" else
                           f"{'ft' if model_velocity_units == 'ft/s' else 'm'} per sample (its dt word, "
                           f"{vs.scan['dt_us']})."))
    vs.settings = dict(velocity_model_type=velocity_model_type, model_velocity_units=model_velocity_units,
                       model_domain=model_domain, model_sample_interval=dt_s, output_velocity_units=output_velocity_units)
    vs.signature = ("model", path_m, os.path.getmtime(path_m), tuple(sorted(vs.settings.items())),
                    tuple(sorted(vs.model_transform.items())))
    return vs
