"""The pipeline steps (see pipeline.py).  Importing this module registers them.

To add a step, write `fn(state, **params) -> (new_state, note)` with @step - it then shows up
in the Pipeline tool's step list with an auto-generated parameter form.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import correction, decon, detection, filters, stacking
from .params import ANALYSIS, BAD, DEAD, DECON
from .pipeline import PipeState, step
from .registry import Param
from .saving import DEFAULT_DIR
from .segy_io import ShotGather, trace_order
from .segy_write import ProcessedShot, write_shot_segy


def _traces(g: ShotGather, mask: np.ndarray, limit: int = 12) -> str:
    idx = [f"{g.i0 + i + 1}" for i in np.flatnonzero(mask)[:limit]]
    return ", ".join(idx) + (" ..." if int(mask.sum()) > limit else "")


@step("Detect Dead Traces", "Flag flat / zero / near-dead traces (used by the correction step and marked on the plot).",
      params=ANALYSIS + DEAD, order=10, category="QC")
def detect_dead(state: PipeState, **p):
    res = detection.run_qc(state.gather, check_dead=True, check_bad=False, **p)
    n = int(res.dead.sum())
    note = f"{n} dead traces" + (f" (file traces {_traces(state.gather, res.dead)})" if n else "")
    return replace(state, dead=res.dead, dead_checked=True), note


@step("Detect Bad Traces", "Flag noisy / weak / spiky / DC-offset traces (marked on the plot).",
      params=ANALYSIS + BAD, order=20, category="QC")
def detect_bad(state: PipeState, **p):
    res = detection.run_qc(state.gather, check_dead=False, check_bad=True, **p)
    bad = res.bad & ~state.dead                      # a trace already found dead is not also 'bad'
    n = int(bad.sum())
    note = f"{n} bad traces" + (f" (file traces {_traces(state.gather, bad)})" if n else "")
    return replace(state, bad=bad), note


@step("Correct Dead Traces", "Interpolate or remove the flagged dead traces "
      "(if no Detect Dead Traces step ran before, they are found with the default thresholds).",
      params=[Param("mode", "Correct dead traces by", "choice", "interpolate", choices=["interpolate", "remove"],
                    help="interpolate: replace each dead trace from its live neighbours on the same receiver line. "
                         "remove: delete the dead traces from the gather", group="Correction")],
      order=30)
def correct_dead(state: PipeState, mode: str = "interpolate"):
    g, dead, auto = state.gather, state.dead, False
    if not state.dead_checked:
        dead = detection.run_qc(g, check_dead=True, check_bad=False, dead_rel_rms=0.01).dead
        auto = True
    src = " (found with default thresholds - no Detect step before)" if auto else ""
    n = int(dead.sum())
    if n == 0:
        return replace(state, dead_checked=True), "no dead traces - nothing to correct" + src
    fixed, mask = correction.correct_dead(g, dead, mode)
    bad = state.bad[~dead] if mode == "remove" else state.bad
    new = replace(state, gather=fixed, dead=np.zeros(fixed.ntr, bool), bad=bad, dead_checked=True, marks=mask,
                  marks_label="interpolated trace" if mode == "interpolate" else "removal point",
                  src=state.src[~dead] if mode == "remove" else state.src)
    verb = "interpolated" if mode == "interpolate" else f"removed ({g.ntr} -> {fixed.ntr} traces)"
    return new, f"{n} dead traces {verb}{src}"


@step("Spiking Decon", "Spiking (Wiener-Levinson) deconvolution of every trace.", params=DECON, order=40)
def spiking_decon(state: PipeState, operator_ms: float = 160.0, prewhite_pct: float = 0.1,
                  design_start_ms: float = 0.0, design_end_ms: float = 0.0, balance: bool = True):
    g = state.gather
    data = decon.spiking_decon(g.data, g.dt_ms, operator_ms=operator_ms, prewhite_pct=prewhite_pct,
                               t_start_ms=design_start_ms, t_end_ms=design_end_ms, balance=balance)
    end = design_end_ms if design_end_ms > 0 else g.ns * g.dt_ms
    note = (f"operator {operator_ms:g} ms ({int(round(operator_ms / g.dt_ms))} samples), pre-whitening "
            f"{prewhite_pct:g} %, design window {design_start_ms:g}-{end:g} ms"
            + (", RMS restored" if balance else ", RMS not restored"))
    return replace(state, gather=ShotGather(g.ffid, g.i0, data, g.headers, g.dt_ms)), note


# ---------------------------------------------------------------------------
# processing steps (corrections, gain, filters, moveout, stack) - none of them QC
# ---------------------------------------------------------------------------
def _with_data(state: PipeState, data: np.ndarray) -> PipeState:
    g = state.gather
    return replace(state, gather=ShotGather(g.ffid, g.i0, data, g.headers, g.dt_ms))


@step("Geometric Spreading", "Gain that compensates the amplitude decay with time: every sample times (t / 1 s)^power.",
      params=[Param("power", "Time power", "float", 2.0, min=0, max=4, step=0.5, group="Gain",
                    help="0 = no gain; 1 - 2 is usual for a spreading correction")], order=35)
def geometric_spreading(state: PipeState, power: float = 2.0):
    g = state.gather
    return _with_data(state, filters.spreading_gain(g.data, g.dt_ms, power)), f"t^{power:g} gain"


@step("Bandpass Filter", "Zero-phase trapezoidal band-pass filter of every trace (Ormsby corners, Hz).",
      params=[Param("f1", "Low cut, Hz", "float", 3.0, min=0, step=1, group="Band-pass"),
              Param("f2", "Low pass, Hz", "float", 6.0, min=0, step=1, group="Band-pass"),
              Param("f3", "High pass, Hz", "float", 60.0, min=0, step=1, group="Band-pass"),
              Param("f4", "High cut, Hz", "float", 80.0, min=0, step=1, group="Band-pass",
                    help="Must stay below the Nyquist frequency (250 Hz for 2 ms samples)")], order=45)
def bandpass_filter(state: PipeState, f1: float = 3.0, f2: float = 6.0, f3: float = 60.0, f4: float = 80.0):
    g = state.gather
    return _with_data(state, filters.bandpass(g.data, g.dt_ms, f1, f2, f3, f4)), f"{f1:g}-{f2:g}-{f3:g}-{f4:g} Hz"


@step("AGC", "Automatic gain control: every sample divided by the RMS amplitude of a window around it.",
      params=[Param("window_ms", "AGC window, ms", "float", 500.0, min=50, step=50, group="Gain")], order=50)
def agc_gain(state: PipeState, window_ms: float = 500.0):
    g = state.gather
    return _with_data(state, filters.agc(g.data, g.dt_ms, window_ms)), f"{window_ms:g} ms window"


@step("Top Mute", "Zero everything above a line t = t0 + |offset| / velocity (noise ahead of the first arrivals), with a taper.",
      params=[Param("velocity", "Mute velocity (length unit per s)", "float", 12000.0, min=100, step=100, auto="mute_velocity",
                    group="Mute", help="Slightly faster than the first arrivals, so only what is ahead of them is muted"),
              Param("t0_ms", "Time at zero offset, ms", "float", 0.0, min=0, step=50, group="Mute"),
              Param("taper_ms", "Taper length, ms", "float", 100.0, min=0, step=10, group="Mute")], order=55)
def top_mute(state: PipeState, velocity: float = 12000.0, t0_ms: float = 0.0, taper_ms: float = 100.0):
    g = state.gather
    out = filters.top_mute(g.data, g.dt_ms, g.headers["offset"], velocity, t0_ms, taper_ms)
    return _with_data(state, out), f"line {t0_ms:g} ms + offset / {velocity:g}, taper {taper_ms:g} ms"


_SORT_KEYS = {"offset": "offset", "channel": "channel", "receiver": "receiver", "cdp": "cdp", "cdp, offset": "cdp_offset"}


@step("Sort Traces", "Reorder the traces of the gather (by offset, channel or receiver line / station).",
      params=[Param("by", "Sort by", "choice", "offset", choices=list(_SORT_KEYS), group="Sort",
                    help="offset = absolute source-receiver offset; receiver = receiver line, then station; cdp = the CDP "
                         "(ensemble) number of the trace header; cdp, offset = by CDP, and by offset within a CDP"),
              Param("descending", "Descending", "bool", False, group="Sort")], order=60)
def sort_traces(state: PipeState, by: str = "offset", descending: bool = False):
    g = state.gather
    order = trace_order(g, _SORT_KEYS[by])
    if descending:
        order = order[::-1]
    new = ShotGather(g.ffid, g.i0, g.data[order], {k: v[order] for k, v in g.headers.items()}, g.dt_ms)
    return (replace(state, gather=new, dead=state.dead[order], bad=state.bad[order],
                    marks=None if state.marks is None else state.marks[order], src=state.src[order]),
            f"by {by}{', descending' if descending else ''}")


@step("NMO Correction", "Normal-moveout correction of every trace with a 1-D velocity function (time, velocity picks) and "
      "a stretch mute. Sort by offset first to see the flattened events in order.",
      params=[Param("velocity", "NMO velocity function  (table: time ms, velocity per s)", "table",
                    stacking.default_velocity_rows("ft"), auto="velocity_table", group="NMO",
                    help="Time (ms) and velocity (length unit of the file per second) picks, linear between them; the "
                         "starting function is only a guess - replace it"),
              Param("stretch", "NMO stretch mute, %  (0 = off)", "float", 30.0, min=0, max=200, step=5, group="NMO")],
      order=65)
def nmo_step(state: PipeState, velocity=None, stretch: float = 30.0):
    g = state.gather
    vt, vv = stacking.parse_velocity(velocity if velocity else stacking.default_velocity_rows("ft"))
    out, live = stacking.nmo_correct(g.data, g.headers["offset"].astype(float), g.dt_ms, vt, vv, stretch)
    return (_with_data(state, out),
            f"{len(vt)}-pick velocity {vv.min():,.0f}-{vv.max():,.0f}, stretch mute {stretch:g} % "
            f"({100 * (1 - live.mean()):.0f} % of the samples muted)")


@step("Stack Traces", "Stack (mean or sum) all traces of the gather into one trace - normally after an NMO correction. "
      "Muted (zero) samples do not count for the mean.",
      params=[Param("method", "Stack", "choice", "mean", choices=["mean", "sum"], group="Stack")], order=70)
def stack_traces(state: PipeState, method: str = "mean"):
    g = state.gather
    total = g.data.sum(axis=0)
    if method == "mean":
        n = (g.data != 0).sum(axis=0)
        total = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    mid = g.ntr // 2                                            # the header of the middle trace stands for the stack
    new = ShotGather(g.ffid, g.i0, total[None, :].astype(np.float32), {k: v[[mid]] for k, v in g.headers.items()}, g.dt_ms)
    return (replace(state, gather=new, dead=np.zeros(1, bool), bad=np.zeros(1, bool), dead_checked=True, marks=None,
                    marks_label="", src=state.src[[mid]]),
            f"{g.ntr} traces -> 1 ({method})")


# ---------------------------------------------------------------------------
# Save Data: write the gather as it is at this point of the flow
# ---------------------------------------------------------------------------
def save_target(path: str, flow_before: list[dict], params: dict, ffid: int | None = None) -> str:
    """File a Save Data step writes: `<folder>/<name or source name>[_FFID<n>]_<code>.sgy`.  <code> identifies the steps
    before it (and their settings), so the same flow always gives the same name and another flow never overwrites it.
    `ffid` None = the whole-data file (all shots), else the single-shot file."""
    code = hashlib.md5(json.dumps(flow_before, sort_keys=True, default=str).encode()).hexdigest()[:6]
    base = str(params.get("name") or "").strip() or Path(path).stem
    tail = f"_FFID{ffid}" if ffid is not None else ""
    return str(Path(str(params.get("folder") or DEFAULT_DIR).strip()).expanduser() / f"{base}{tail}_{code}.sgy")


@step("Save Data", "Write the data as it is at this point of the flow to a SEG-Y file (headers of the source traces kept). "
      "For one shot when 'Write file' is ticked; in 'Apply to whole data' the data of EVERY shot at this point goes into one "
      "file next to the final output - a checkpoint before the later steps.",
      params=[Param("folder", "Folder", "text", DEFAULT_DIR, group="Save"),
              Param("name", "File name  (empty = the source file's name)", "text", "", group="Save",
                    help="A code of the steps before this one is added, so other flows never overwrite it"),
              Param("fmt", "Sample format", "choice", "ibm", choices=["ibm", "ieee"], group="Save"),
              Param("write", "Write file", "bool", False, group="Save",
                    help="Off by default: the pipe re-runs every time you change a setting or move to another shot, and "
                         "each run would write a file. Tick it when you want this shot saved. (A whole-data run always "
                         "writes its checkpoint file.)")], order=90)
def save_data_step(state: PipeState, folder: str = DEFAULT_DIR, name: str = "", fmt: str = "ibm", write: bool = False):
    g = state.gather
    params = {"folder": folder, "name": name}
    if state.batch:
        return state, f"whole-data run: the data of every shot at this point goes to {save_target(state.path, state.flow, params)}"
    if not write:
        return state, "not written - tick 'Write file' to save this shot here"
    if not state.path:
        raise ValueError("no source file is known, so the trace headers cannot be copied")
    out = save_target(state.path, state.flow, params, g.ffid)
    notes = state.log[-3:]
    write_shot_segy(ProcessedShot(state.path, g, state.src, "Save Data", notes, flow=list(state.flow)), out, fmt)
    return state, f"saved {g.ntr} trace(s) to {out}"
