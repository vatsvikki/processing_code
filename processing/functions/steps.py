"""The pipeline steps (see pipeline.py).  Importing this module registers them.

To add a step, write `fn(state, **params) -> (new_state, note)` with @step - it then shows up
as a function of the Processing tab, with an auto-generated parameter form.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np

from . import cdp_sort, correction, decon, detection, filters, fold as fold_mod, geometry, nmo, noise, plotting
from . import cdp_flow, segy_io, stack as stack_mod, velocity_setup
from .pipeline import get_step
from .saving import DEFAULT_DIR
from . import grid as grid_mod
from .params import ANALYSIS, BAD, DEAD, DECON, DISPLAY
from .pipeline import PipeState, step
from .registry import Output, Param, figure, flip, markdown, table
from .segy_io import ShotGather, trace_order


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
# processing steps (gain, filter, mute) - none of them QC
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


# ---------------------------------------------------------------------------
# coherent-noise removal: F-K filter and Radon (functions/noise.py)
# ---------------------------------------------------------------------------
_OUTPUTS = ["Filtered data", "Removed noise"]


def _gather_flip(g: ShotGather, filtered: np.ndarray, what: str):
    """Flip-flop of the shot gather before / after a noise filter and the noise it removed - all at the clip of the
    input, so what changes between the frames is the data, not the display scaling."""
    ref = np.abs(g.data[:, ::2])
    clip = float(np.percentile(ref, 98)) if ref.size else 1.0
    frames = []
    for label, data in (("Before", g.data), ("After", filtered), ("Removed noise", g.data - filtered)):
        sg = ShotGather(g.ffid, g.i0, np.asarray(data, np.float32), g.headers, g.dt_ms)
        frames.append((label, plotting.plot_gather(sg, clip=clip or 1.0, fig_height=7.0,
                                                   title=f"FFID {g.ffid} - {what} - {label.lower()} (same clip)")))
    return flip(f"{what}: gather flip-flop (before / after / removed)", frames)


FK_MODES = ["Velocity fan (from the inputs)", "Manual polygon (drawn on the F-K plot)"]
_FK_POLY_DEFAULT = [{noise.Z_COL: "", noise.K_COL: "", noise.F_COL: ""}]     # none: the user clicks the points


@step("F-K Filter", "Removes ground roll and other slow linear noise in the frequency-wavenumber (F-K) domain, one "
      "receiver line at a time: a velocity fan from the inputs, or a reject polygon drawn on the F-K plot.",
      params=[Param("fk_mode", "Reject zone", "choice", FK_MODES[0], choices=FK_MODES, group="F-K filter",
                    help="Velocity fan: everything slower than the reject velocity. Manual polygon: drag a box or "
                         "Shift + drag a lasso on the 'F-K domain' plot (or type the corners) - that zone is rejected"),
              Param("v_reject", "Reject apparent velocities below (length unit per s)", "float", 3000.0, min=100, step=100,
                    group="F-K filter", show_if={"fk_mode": FK_MODES[0]},
                    help="Ground roll and other slow noise travel slower than this; reflections "
                         "(apparent velocity far higher) are kept"),
              Param("v_pass", "Pass apparent velocities above (length unit per s)", "float", 4500.0, min=100, step=100,
                    group="F-K filter", show_if={"fk_mode": FK_MODES[0]},
                    help="Fully kept above this; a cosine taper between the two velocities avoids "
                         "ringing. Must be above the reject velocity"),
              Param("fk_f_max", "Filter only below, Hz  (0 = all frequencies)", "float", 0.0, min=0, step=5,
                    group="F-K filter", show_if={"fk_mode": FK_MODES[0]},
                    help="Ground roll is low-frequency: e.g. 25 leaves everything above 25 Hz untouched"),
              Param("fk_polygon", "Reject zone corners  (zone, k cycles per 1000 length units, f Hz)", "table",
                    _FK_POLY_DEFAULT, group="F-K filter", show_if={"fk_mode": FK_MODES[1]},
                    help="Filled in by the polygon editor on the F-K domain plot (click points, double-click to "
                         "finish; drag a point to move it, right-click to delete it; applied at once). Rows with the same 'zone' number are "
                         "one polygon (at least 3 corners); every zone is rejected. You can also type the corners here"),
              Param("fk_mirror", "Mirror the polygon to negative k", "bool", True, group="F-K filter",
                    show_if={"fk_mode": FK_MODES[1]},
                    help="Ground roll goes both ways from the source (split spread): the same zone at -k is rejected too"),
              Param("fk_plot_fmax", "F-K plots up to, Hz  (0 = auto: 50 Hz)", "float", 0.0, min=0,
                    step=10, group="F-K filter",
                    help="Frequency range of the F-K plots - lower = the low-frequency ground roll is bigger to draw on"),
              Param("fk_output", "Output", "choice", _OUTPUTS[0], choices=_OUTPUTS, group="F-K filter",
                    help="Removed noise = what the filter takes away (input minus filtered) - to check that no "
                         "reflection energy is removed")],
      order=37)
def fk_filter_step(state: PipeState, fk_mode: str = FK_MODES[0], v_reject: float = 3000.0, v_pass: float = 4500.0,
                   fk_f_max: float = 0.0, fk_polygon=None, fk_mirror: bool = True,
                   fk_plot_fmax: float = 0.0,
                   fk_output: str = _OUTPUTS[0]):
    g = state.gather
    manual = fk_mode == FK_MODES[1]
    poly = noise.polygon_rows(fk_polygon if fk_polygon is not None else _FK_POLY_DEFAULT) if manual else None
    if manual and not any(len(p) >= 3 for p in poly):
        weight = lambda f, k: np.ones((len(f), len(k)))         # no polygon yet: nothing rejected
        zone = "no reject polygon yet (click points on the F-K domain, double-click to finish) - nothing removed"
    elif manual:
        weight = noise.polygon_weight(poly, fk_mirror)
        zone = (f"reject zone of {sum(len(p) >= 3 for p in poly)} polygon(s), {sum(len(p) for p in poly)} corners"
                + (" (mirrored to -k)" if fk_mirror else ""))
    else:
        if v_pass <= v_reject:
            raise ValueError(f"the pass velocity ({v_pass:g}) must be above the reject velocity ({v_reject:g})")
        weight = noise.fan_weight(v_reject, v_pass, fk_f_max)
        zone = (f"rejects apparent velocities below {v_reject:g}, passes above {v_pass:g}"
                + (f", only below {fk_f_max:g} Hz" if fk_f_max > 0 else ""))
    out, shown, note = noise.fk_filter(g, weight, spectra=not state.batch)
    data = out if fk_output == _OUTPUTS[0] else (g.data - out).astype(g.data.dtype)
    figs = None
    if shown is not None:
        nyq = 500.0 / g.dt_ms
        # one fixed range for both reject-zone modes: the axes must not move under the points being drawn
        f_lim = min(fk_plot_fmax if fk_plot_fmax > 0 else 50.0, nyq)
        draw = dict(fan=None if manual else (v_reject, v_pass), poly=poly, mirror=fk_mirror)
        domain = noise.plot_fk(shown, f_lim, "before", figsize=(8.0, 6.5), **draw)
        live = plotting.plot_gather(ShotGather(g.ffid, g.i0, np.asarray(out, np.float32), g.headers, g.dt_ms),
                                    clip=float(np.percentile(np.abs(g.data[:, ::2]), 98)) or 1.0, figsize=(8.0, 6.5),
                                    title=f"FFID {g.ffid} after the F-K filter (input's clip)")
        figs = [Output("image", "F-K domain",
                       plotting.figure_to_png(domain), figure=domain, pick="fk"),
                Output("image", "Shot gather with this reject zone", plotting.figure_to_png(live), figure=live,
                       pick="fk_live"),
                flip("F-K spectrum flip-flop (before / after / removed noise)",
                     [(lab, noise.plot_fk(shown, f_lim, which, **draw)) for lab, which in
                      (("Before", "before"), ("After", "after"), ("Removed noise", "removed"))]),
                _gather_flip(g, out, "F-K Filter")]
    note = f"{zone} · {note}" + ("" if fk_output == _OUTPUTS[0] else " · showing the removed noise")
    return replace(_with_data(state, data), figs=figs), note


_RADON = ["Linear (ground roll)", "Parabolic (multiples)"]


@step("Radon Filter", "Least-squares Radon transform: Linear (tau-p) removes ground roll / slow linear noise, also when "
      "it is spatially aliased; Parabolic removes multiples (after NMO with a velocity function they keep a residual "
      "moveout). The noise is modelled and subtracted, the rest of the data is left as it was.",
      params=[Param("radon_type", "Radon type", "choice", _RADON[0], choices=_RADON, group="Radon",
                    help="Linear: ground roll and linear noise (t = offset / velocity). Parabolic: multiples"),
              Param("radon_output", "Output", "choice", _OUTPUTS[0], choices=_OUTPUTS, group="Radon",
                    help="Removed noise = the modelled noise that is subtracted - to check that no primary energy is removed"),
              Param("v_cut", "Remove events slower than (length unit per s)", "float", 3000.0, min=100, step=100,
                    group="Linear Radon", show_if={"radon_type": _RADON[0]},
                    help="Ground roll / slow noise travel slower than this; reflections are faster and kept"),
              Param("v_min", "Slowest velocity modelled (length unit per s)", "float", 1000.0, min=100, step=100,
                    group="Linear Radon", show_if={"radon_type": _RADON[0]},
                    help="Must be below the slowest noise. Lower = more time padding = slower"),
              Param("lin_f_max", "Highest frequency, Hz", "float", 30.0, min=1, step=5, group="Linear Radon",
                    show_if={"radon_type": _RADON[0]}, help="Ground roll is low-frequency: above this nothing is removed"),
              Param("radon_vel", "Primary velocity function  (table: time ms, Vrms per row)", "table", nmo.DEFAULT_VELFN,
                    group="Parabolic Radon", show_if={"radon_type": _RADON[1]},
                    help="RMS velocity of the PRIMARIES: NMO with it flattens them, the multiples (slower) stay curved"),
              Param("q_min_ms", "Smallest moveout modelled, ms", "float", -100.0, step=10, group="Parabolic Radon",
                    show_if={"radon_type": _RADON[1]}, help="Moveout at the farthest offset after NMO (negative = over-corrected)"),
              Param("q_max_ms", "Largest moveout modelled, ms", "float", 600.0, step=10, group="Parabolic Radon",
                    show_if={"radon_type": _RADON[1]}),
              Param("q_cut_ms", "Remove moveout above, ms", "float", 40.0, step=5, group="Parabolic Radon",
                    show_if={"radon_type": _RADON[1]},
                    help="Primaries are flat (about 0 ms) after NMO; events with more moveout than this at the far "
                         "offset are taken as multiples and removed"),
              Param("par_f_max", "Highest frequency, Hz", "float", 80.0, min=1, step=5, group="Parabolic Radon",
                    show_if={"radon_type": _RADON[1]}),
              Param("n_p", "Number of slownesses / moveouts", "int", 150, min=20, max=500, step=10, group="Radon",
                    help="More = finer separation, slower"),
              Param("damping_pct", "Damping (pre-whitening), %", "float", 1.0, min=0.01, step=0.5, group="Radon",
                    help="Stabilises the least-squares inversion; more = smoother, less sharp Radon panel")],
      order=38)
def radon_step(state: PipeState, radon_type: str = _RADON[0], radon_output: str = _OUTPUTS[0], v_cut: float = 3000.0,
               v_min: float = 1000.0, lin_f_max: float = 30.0, radon_vel=None, q_min_ms: float = -100.0,
               q_max_ms: float = 600.0, q_cut_ms: float = 40.0, par_f_max: float = 80.0, n_p: int = 150,
               damping_pct: float = 1.0):
    g = state.gather
    if radon_type == _RADON[0]:
        if not v_min < v_cut:
            raise ValueError(f"the slowest velocity modelled ({v_min:g}) must be below the cut ({v_cut:g})")
        clean, removed, panel, note = noise.radon_linear(g, v_cut, v_min, n_p, lin_f_max, damping_pct)
        kind = "linear"
    else:
        if not q_min_ms < q_cut_ms < q_max_ms:
            raise ValueError("the moveouts must be: smallest < remove above < largest")
        vel, _ = nmo.velocity_function(radon_vel if radon_vel is not None else nmo.DEFAULT_VELFN,
                                       np.arange(g.ns) * (g.dt_ms / 1000.0))
        clean, removed, panel, note = noise.radon_parabolic(g, vel, q_min_ms, q_max_ms, q_cut_ms, n_p, par_f_max,
                                                            damping_pct)
        kind = "parabolic"
    data = clean if radon_output == _OUTPUTS[0] else removed
    figs = None if state.batch else [_gather_flip(g, clean, "Radon Filter"),
                                     figure("Radon panel", noise.plot_radon(*panel, g.dt_ms, kind))]
    return replace(_with_data(state, data), figs=figs), note + ("" if radon_output == _OUTPUTS[0] else " · showing the removed noise")


# ---------------------------------------------------------------------------
# "Display" functions of the Flow: they do not change the gather - they set state.view_extra, a map figure shown
# for this stage instead of the usual gather plot (see pipeline.run_steps / tools._flow_outputs). Chainable so a
# map can be checked at any chosen point of a flow.
# ---------------------------------------------------------------------------
_FIG_HEIGHT = next(p for p in DISPLAY if p.key == "fig_height")

GEOMETRY = [
    Param("show_sources", "Show source positions", "bool", True, live=True, group="Display"),
    Param("show_receivers", "Show receiver positions", "bool", True, live=True, group="Display"),
    Param("show_selected", "Highlight the selected shot", "bool", True, live=True, group="Display",
          help="Its source (star) and the receivers that recorded it, plus their spread relative to the source"),
    Param("show_grid", "Show the IL / XL grid", "bool", True, live=True, group="Display",
          help="Inline / crossline lines and the survey outline, when a grid is known (see the corner points)"),
    Param("ilxl_ticks", "Tick labels in IL / XL  (instead of X / Y)", "bool", False, live=True, group="Display",
          help="Axis ticks at round inline / crossline numbers; needs a grid (headers or corner points)"),
    Param("rec_shots", "Shots read for the receiver positions  (0 = all)", "int", 60, min=0, step=10, live=True,
          help="Receiver positions come from this many evenly spaced shots. Sources are always taken from every "
               "shot. 60 is enough for a fixed receiver patch; 0 reads every trace header of the file (~20 s "
               "for 950 000 traces) - use it when the receivers move", group="Geometry"),
]
_GEO = {p.key: p for p in GEOMETRY}
FOLD = [
    Param("show_sources", "Show source positions", "bool", False, live=True, group="Display"),
    Param("show_receivers", "Show receiver positions", "bool", False, live=True, group="Display"),
    _GEO["show_grid"], _GEO["ilxl_ticks"],
    Param("cmap", "Colour map", "choice", "viridis", live=True, choices=["viridis", "plasma", "turbo", "cividis", "magma"],
          group="Fold"),
    Param("fold_max", "Colour scale maximum  (0 = automatic)", "int", 0, min=0, step=5, live=True, group="Fold",
          help="Automatic = the 99.5th percentile of the fold, so a few very high bins do not wash out the rest"),
    Param("bin_align", "Bin alignment", "choice", "auto", live=True, choices=fold_mod.ALIGN_CHOICES, group="Fold",
          help="Where the bins sit relative to whole IL / XL numbers. Auto tries the four half-bin shifts and keeps "
               "the smoothest fold (midpoints at the bin centres); midpoints lying exactly on bin edges show up "
               "as stripes"),
]


def _resolve_grid(path: str, geom, corners, forget_saved: bool, rec_shots: int = 60):
    """The IL / XL grid for a map: the corner table given wins over the IL / XL of the headers.
    Returns (grid or None, [notes for the user]); saves / forgets the table as needed."""
    header_grid, why = grid_mod.grid_from_headers(geom)
    # the table is filled in from the header grid of the default read; an untouched table equals it
    base_grid = header_grid if rec_shots == 60 else grid_mod.grid_from_headers(geometry.read_geometry(path, 60))[0]
    notes, grid, typed = [], None, []
    if forget_saved:
        notes.append("🗑 Saved table for this file removed - back to the header IL / XL." if grid_mod.forget_saved(path)
                     else "No table was saved for this file.")
        corners = None
    try:
        typed = grid_mod.parse_corners(corners)
    except ValueError as e:
        notes.append(f"⚠ **Table not used:** {e}")
    if typed and any(g is not None and grid_mod.same_corners(typed, g) for g in (header_grid, base_grid)):
        typed = []                                    # still the header values: nothing was changed
        if grid_mod.forget_saved(path):
            notes.append("The table equals the header IL / XL again - the table saved earlier was removed.")
    if typed:
        try:
            grid = grid_mod.grid_from_corners(typed, header_grid, why)
            fresh = grid_mod.save_corners(path, grid_mod.corners_to_text(typed))
            notes.append(f"IL / XL taken from **your table** ({len(typed)} corner points)"
                         + (" - it replaces the IL / XL of the headers. " if header_grid is not None else ". ")
                         + ("💾 Saved for this file" if fresh else "Saved for this file")
                         + f" ({grid_mod.store_path()}) - change it in the **Survey grid** table of the 📁 Data card.")
            if grid.rms > 0.05 * min(grid.il_spacing, grid.xl_spacing):
                notes.append(f"⚠ **The corner points do not fit a regular grid** (best fit is off by {grid.rms:,.1f} "
                             f"{geom.unit or 'units'} on average): check the IL, XL, X and Y values of every row.")
        except ValueError as e:
            notes.append(f"⚠ **Table not used:** {e}")
    if grid is None and header_grid is not None:
        grid = header_grid
        what = ("**IL = receiver line** (RECLN, byte 173) and **XL = receiver station** (RECSTN, byte 181) of the receivers"
                if grid.source == "receiver headers" else "the inline / crossline words of the trace headers")
        notes.append(f"IL / XL taken from the headers: {what}. To use your own IL / XL, change the **Survey grid** "
                     "table of the 📁 Data card.")
    elif grid is None:
        notes.append(f"⚠ **No inline / crossline grid in the headers:** {why}. Fill in the **Survey grid** table of the "
                     "📁 Data card (one corner per row: IL, XL, X, Y - e.g. 1001, 1001, 1971388.1, 428622).")
    return grid, notes


@step("Plot Shot Gather", "Plain shot gather image - the same view the flow already shows for this stage (marks any "
      "dead / bad traces flagged so far); does not change the gather. Useful as a labelled checkpoint in the flow.",
      order=69, category="Display")
def plot_shot_gather_step(state: PipeState):
    return state, "gather shown"


@step("Acquisition Geometry", "Plan view of the acquisition: every source position and the receiver positions, with "
      "this shot's spread highlighted (coordinates from the trace headers). Does not change the gather.",
      params=[_FIG_HEIGHT] + GEOMETRY, order=70, category="Display")
def acquisition_geometry_step(state: PipeState, fig_height: float = 7.0, show_sources: bool = True,
                              show_receivers: bool = True, show_selected: bool = True, show_grid: bool = True,
                              ilxl_ticks: bool = False, rec_shots: int = 60):
    if state.batch:            # nothing displays this map inside a whole-data run - skip building it for every shot
        return state, "map not built (whole-data run)"
    g = state.gather
    geom = geometry.read_geometry(state.path, rec_shots)
    spread = geometry.read_spread(state.path, g.ffid) if show_selected else None
    grid, notes = _resolve_grid(state.path, geom, grid_mod.corner_table(state.path), False, rec_shots)
    if ilxl_ticks and grid is None:
        notes.append("IL / XL tick labels need a grid - the axes stay in X / Y until corner points are entered.")
    out = [markdown("Inline / crossline grid", "  \n".join(notes))]
    out.append(figure("Acquisition geometry", plotting.plot_geometry(
        geom, spread, grid=grid, show_grid=show_grid, ilxl_ticks=ilxl_ticks, show_sources=show_sources,
        show_receivers=show_receivers, fig_height=fig_height)))
    if spread is not None:
        out.append(figure(f"Spread of FFID {spread.ffid}", plotting.plot_spread(spread, geom.unit)))
    out.append(table("Survey geometry", ["Item", "Value"], geometry.summary_rows(geom)))
    if grid is not None:
        out.append(table("Inline / crossline grid", ["Item", "Value"], grid_mod.grid_rows(grid, geom.unit, spread)))
    return replace(state, view_extra=out), f"map shown ({len(geom.ffids):,} shots)"


@step("Acquisition Fold", "Fold map: traces per bin from the source-receiver midpoints of EVERY trace, coloured on "
      "the same canvas as Acquisition Geometry (colour bar inside the figure). The first run reads all trace "
      "headers (~20 s for 950 000 traces); after that changes are instant. Does not change the gather.",
      params=[_FIG_HEIGHT] + FOLD, order=71, category="Display")
def acquisition_fold_step(state: PipeState, fig_height: float = 7.0, show_sources: bool = False,
                          show_receivers: bool = False, show_grid: bool = True, ilxl_ticks: bool = False,
                          cmap: str = "viridis", fold_max: int = 0, bin_align: str = "auto"):
    if state.batch:            # nothing displays this map inside a whole-data run - skip building it for every shot
        return state, "fold map not built (whole-data run)"
    geom = geometry.read_geometry(state.path, 60)
    grid, notes = _resolve_grid(state.path, geom, grid_mod.corner_table(state.path), False, 60)
    if ilxl_ticks and grid is None:
        notes.append("IL / XL tick labels need a grid - the axes stay in X / Y until corner points are entered.")
    bin_size = fold_mod.auto_bin_size(geom)
    if grid is None:
        notes.append(f"No IL / XL grid, so the fold is counted in square X / Y bins of {bin_size:g} {geom.unit} "
                     "(half the receiver spacing). Fill in the Survey grid of the 📁 Data card to count it per IL / XL "
                     "cell instead.")
    fold_map = fold_mod.compute_fold(state.path, grid, bin_size, bin_align)
    out = [markdown("Inline / crossline grid", "  \n".join(notes))]
    out.append(figure("Fold map", plotting.plot_fold(
        fold_map, geom, grid=grid, show_grid=show_grid, ilxl_ticks=ilxl_ticks, show_sources=show_sources,
        show_receivers=show_receivers, cmap=cmap, vmax=fold_max, fig_height=fig_height)))
    out.append(table("Fold statistics", ["Item", "Value"], fold_mod.summary_rows(fold_map)))
    if grid is not None:
        out.append(table("Inline / crossline grid", ["Item", "Value"], grid_mod.grid_rows(grid, geom.unit)))
    return replace(state, view_extra=out), f"fold map shown ({fold_map.traces:,} traces)"


# ---------------------------------------------------------------------------------------------------------------------
# NMO correction and CDP stacking (the brute-stack notebook): parameters shared by both steps
# ---------------------------------------------------------------------------------------------------------------------
def _cdp_params(what: str):
    return [
        Param("cdp_pick", f"CDP to {what}  (top 200 by fold)", "choice", "", choices=[], choices_auto="top_cdps",
              live=True, group="CDP", help="The 200 CDPs with the highest fold, with their data IL / XL (this file's grid) "
              "and fold, highest first. The list is read from all trace headers the first time"),
        Param("cdp_number", "Other CDP  (0 = use the list above)", "int", 0, min=0, step=1, live=True, group="CDP",
              help="Any CDP number (see the table of 🛠 Tools -> CDP Sort) - it overrides the list; a number not in the file shows the "
                   "nearest CDP"),
    ]


def _velocity_params(sources):
    return [
        Param("vel_source", "Velocity for NMO", "choice", sources[0], choices=list(sources), live=True, group="Velocity",
              help="First choose where the velocity comes from. Velocity model SEG-Y: the model trace nearest each CDP "
                   "(then its file, settings and the model's corner points are asked for). Velocity function: one "
                   "time / RMS-velocity table used for every CDP of the survey"
                   + (". No NMO: the plain mean of the traces (raw stack)" if velocity_setup.NONE in sources else "")),
        Param("vel_function", "Velocity function  (table: time ms, Vrms per row)", "table", nmo.DEFAULT_VELFN, live=True, show_if={"vel_source": velocity_setup.FUNCTION},
              group="Velocity function",
              help="Used with 'Velocity function': RMS velocity at zero-offset times, in the output velocity unit (5.). "
                   "Linear between rows, held flat before the first and after the last; empty rows are ignored"),
        Param("vel_model_path", "Velocity model SEG-Y file", "text", "", live=True, show_if={"vel_source": velocity_setup.MODEL}, group="Velocity model",
              help="Full path of the velocity model (on the machine the app runs on). Its traces are placed by their "
                   "INLINE_3D / CROSSLINE_3D words (bytes 189 / 193)"),
        Param("velocity_model_type", "1. This model's samples are", "choice", nmo.MODEL_TYPES[0], choices=nmo.MODEL_TYPES,
              live=True, show_if={"vel_source": velocity_setup.MODEL}, group="Velocity model"),
        Param("model_velocity_units", "2. This model's input velocity values are in", "choice", "ft/s", choices=nmo.UNITS,
              live=True, show_if={"vel_source": velocity_setup.MODEL}, group="Velocity model"),
        Param("model_domain", "3. Model's vertical axis is", "choice", nmo.DOMAINS[0], choices=nmo.DOMAINS, live=True, show_if={"vel_source": velocity_setup.MODEL},
              group="Velocity model", help="A depth model is always taken as interval velocity vs depth"),
        Param("model_sample_interval", "4. Model sample interval  (ms for time, length unit for depth; 0 = the model "
              "file's own)", "float", 0.0, min=0, step=0.5, live=True, show_if={"vel_source": velocity_setup.MODEL}, group="Velocity model",
              help="0 = taken from the model file's sample-interval header word: ms for a time model, the depth step for a "
                   "depth model (10000 -> 10 ft). A wrong interval mis-times every velocity, which shows up as NMO not "
                   "flattening events - type the right one here if the header is wrong"),
        Param("output_velocity_units", "5. Output velocity (RMS vs TWTT) in", "choice", "ft/s", choices=nmo.UNITS,
              live=True, group="Velocity model",
              help="NMO uses offsets in this unit too (the headers' offsets are taken as feet); also the unit of the "
                   "velocity function"),
        Param("model_corners", "Model grid corner points  (INLINE_3D = IL, CROSSLINE_3D = XL <-> X, Y)", "table",
              grid_mod.table_rows([], 4), live=True, show_if={"vel_source": velocity_setup.MODEL}, group="Model corner points",
              help="The velocity model's own grid, in its INLINE_3D / CROSSLINE_3D numbers. Left empty, they are read from "
                   "the model's text header (a GRID CORNERS block), when it has one. At least 2 rows with distinct IL and "
                   "distinct XL, with X and Y"),
        Param("stretch_mute_pct", "NMO stretch mute, %  (0 = off)", "float", 30.0, min=0, max=100, step=5, live=True,
              group="NMO"),
        Param("nmo_clip_pct", "Amplitude clip percentile  (all image plots)", "float", 98.0, min=80, max=100, step=0.5,
              live=True, group="NMO"),
    ]


_VEL_KEYS = ("vel_model_path", "velocity_model_type", "model_velocity_units", "model_domain", "model_sample_interval",
             "output_velocity_units", "model_corners", "vel_function")


def _cdp_and_velocity(state: PipeState, cdp_pick, cdp_number, vel_source, kw):
    """The CDP chosen, its gather (offset-sorted, with the steps above applied), the velocity setup and the source."""
    src = cdp_flow.CdpSource(state.path, state.flow)
    idx = src.idx
    k = idx.position(int(cdp_number) or cdp_sort.cdp_from_label(cdp_pick))
    cg = src.gather(k)
    full_t = np.arange(cg.ns) * (cg.dt_ms / 1000.0)
    vs = velocity_setup.build(state.path, full_t, source=vel_source, **{k_: kw[k_] for k_ in _VEL_KEYS if k_ in kw})
    return idx, k, cg, full_t, vs, src


def nmo_settings_of(flow: list[dict]):
    """(velocity source, velocity keywords, stretch mute %) of the last NMO Correction step in `flow`, or None."""
    spec = next((f for f in reversed(flow or []) if f["step"] == "nmo_correction_step"), None)
    if spec is None:
        return None
    st = get_step("nmo_correction_step")
    p = {q.key: (spec.get("params") or {}).get(q.key, q.default) for q in st.params}
    kw = {k_: p[k_] for k_ in _VEL_KEYS}
    return p["vel_source"], kw, float(p["stretch_mute_pct"])


def _cdp_velocity(vs, idx, k):
    """(Vrms, model match info or None, the summary text) for CDP k."""
    x, y = float(idx.mid_x[k]), float(idx.mid_y[k])
    vel, _, info = vs.for_position(x, y)
    if info is not None:
        txt = (f"matched model trace INLINE_3D = {info['inline3d']}, CROSSLINE_3D = {info['crossline3d']} "
               f"({info['distance_ft']:,.1f} ft away). Output velocity range (RMS, TWTT): {vel.min():,.0f} - "
               f"{vel.max():,.0f} {vs.unit}.")
    else:
        pts = ", ".join(f"{t:g} ms {v:,.0f}" for t, v in vs.function_rows[:6]) + (" ..." if len(vs.function_rows) > 6 else "")
        txt = f"velocity function: {pts} {vs.unit} (RMS, TWTT)."
    return vel, info, txt


def _head(idx, k, vs):
    x, y = float(idx.mid_x[k]), float(idx.mid_y[k])
    ilxl = ""
    if vs.data_transform is not None:
        c_xl, c_il = nmo.to_data_ilxl(vs.data_transform, x, y)
        ilxl = f" (data IL {c_il:,.0f} / XL {c_xl:,.0f})"
    return f"**CDP {int(idx.cdps[k]):,}** · {int(idx.fold[k])} traces · real position X = {x:,.1f}, Y = {y:,.1f}{ilxl}"


def _model_overlay(vs, idx, k, info, fig_height):
    """The notebook's geometry overlay (data vs. model grid coverage), for a model source with both grids registered."""
    if vs.scan is None or vs.model_transform is None or vs.data_transform is None or info is None:
        return None
    x, y = float(idx.mid_x[k]), float(idx.mid_y[k])
    (m_xl, m_il), (d_xl, d_il) = nmo.overlay_points(vs.scan, vs.model_transform, vs.data_transform, idx.mid_x, idx.mid_y)
    c_xl, c_il = nmo.to_data_ilxl(vs.data_transform, x, y)
    hx = vs.model_transform["mx"] * info["crossline3d"] + vs.model_transform["cx"]
    hy = vs.model_transform["my"] * info["inline3d"] + vs.model_transform["cy"]
    t_xl, t_il = nmo.to_data_ilxl(vs.data_transform, hx, hy)
    return plotting.plot_model_overlay(m_xl, m_il, d_xl, d_il, cdp_xl=c_xl, cdp_il=c_il, match_xl=t_xl, match_il=t_il,
                                       cdp=int(idx.cdps[k]), fig_height=fig_height)




@step("NMO Correction", "NMO correction of a CDP gather as in the brute-stack notebook: velocity from a velocity-model "
      "SEG-Y (the model trace nearest the CDP's position, converted to RMS vs two-way time) or from a velocity function "
      "(time / Vrms table); every sample moved to t0 (t = sqrt(t0^2 + (offset / V)^2)) with an NMO-stretch mute. Shows "
      "the CDP gather, the NMO stretch section and the velocity used. Does not change the shot gather.",
      params=[_FIG_HEIGHT] + _cdp_params("NMO-correct") + _velocity_params(velocity_setup.SOURCES[:2]),
      order=73, category="Display")
def nmo_correction_step(state: PipeState, fig_height: float = 6.0, cdp_pick: str = "", cdp_number: int = 0,
                        vel_source: str = velocity_setup.MODEL, stretch_mute_pct: float = 30.0, nmo_clip_pct: float = 98.0,
                        **kw):
    if state.batch:            # nothing displays this inside a whole-data run - skip it for every shot
        return state, "NMO preview not built (whole-data run)"
    idx, k, cg, full_t, vs, src = _cdp_and_velocity(state, cdp_pick, cdp_number, vel_source, kw)
    cdp = int(idx.cdps[k])
    traces = cg.data.astype(np.float64)
    head = _head(idx, k, vs)

    def _show(md, vel=None, info=None, nmo_traces=None, overlay=None):
        fig = plotting.plot_nmo(traces, nmo_traces if nmo_traces is not None else traces, full_t, vel, info,
                                vs.unit, clip_pct=nmo_clip_pct, fig_height=fig_height,
                                title=f"CDP {cdp} - NMO correction",
                                no_velocity_text="No velocity" if vs.error else "No velocity model")
        out = [markdown("NMO correction", md), figure(f"CDP {cdp} NMO", fig)]
        if overlay is not None:
            out.append(figure("Data vs. model grid coverage", overlay))
        return replace(state, view_extra=out)

    if vs.error:
        return _show(head + f"\n\n⚠ {vs.error} (the gather is shown uncorrected).\n\n{src.describe()}"
                     + "".join(f"\n\n{n}" for n in vs.notes)), f"CDP {cdp}: no velocity"
    vel, info, vtxt = _cdp_velocity(vs, idx, k)
    nmo_traces, _ = nmo.nmo_stack(traces, nmo.cdp_offsets(cg.headers["offset"], vs.unit), full_t, vel,
                                  stretch_mute_pct / 100.0)
    warns = nmo.sanity_warnings(vel, info, vs.unit) if info is not None else []
    md = (head + f" -- {vtxt} Stretch mute {stretch_mute_pct:g} %."
          + "".join(f"\n\n⚠ {w}" for w in warns) + "\n\n" + src.describe() + "".join(f"\n\n{n}" for n in vs.notes))
    out = _show(md, vel, info, nmo_traces, _model_overlay(vs, idx, k, info, fig_height))
    return out, f"CDP {cdp} NMO-corrected (Vrms {vel.min():,.0f}-{vel.max():,.0f} {vs.unit})"


@step("CDP Stack", "Stacks the CDP gathers the steps above produced - nothing else: with NMO Correction above, the mean "
      "of the live (not stretch-muted) samples of its NMO-corrected gathers at every time (its velocity and mute; NMO is "
      "not done again); without it, the plain mean of the CDP gathers. 'Run flow on whole data' stacks every CDP; the "
      "IL / XL stacked sections (and, with NMO above, the velocity sections and overlay) are shown here.",
      params=[_FIG_HEIGHT] + _cdp_params("stack") + [
          Param("section_il", "Fixed IL (inline section)  (0 = the most common)", "int", 0, min=0, step=1, live=True,
                group="Sections"),
          Param("section_xl", "Fixed XL (crossline section)  (0 = the most common)", "int", 0, min=0, step=1, live=True,
                group="Sections"),
          Param("section_velocity_type", "Velocity sections / overlay show  (with NMO above)", "choice", "RMS",
                choices=["RMS", "Interval"], live=True, group="Sections"),
          Param("nmo_clip_pct", "Amplitude clip percentile  (all image plots)", "float", 98.0, min=80, max=100,
                step=0.5, live=True, group="Sections"),
      ],
      order=74, category="Display")
def cdp_stack_step(state: PipeState, fig_height: float = 6.0, cdp_pick: str = "", cdp_number: int = 0,
                   section_il: int = 0, section_xl: int = 0, section_velocity_type: str = "RMS",
                   nmo_clip_pct: float = 98.0):
    if state.batch:            # nothing displays this inside a whole-data run - skip it for every shot
        return state, "stack not built (whole-data run)"
    # the input: the NMO-corrected gathers of an NMO Correction step above (its velocity and mute), or the CDP gathers
    chained = nmo_settings_of(state.flow)
    if chained is not None:
        vel_source, kw, mute = chained
    else:
        vel_source, kw, mute = velocity_setup.NONE, {}, 0.0
    idx, k, cg, full_t, vs, src = _cdp_and_velocity(state, cdp_pick, cdp_number, vel_source, kw)
    cdp = int(idx.cdps[k])
    traces = cg.data.astype(np.float64)
    head = _head(idx, k, vs)
    notes = list(vs.notes)

    if chained is not None and vs.error:
        md = (head + f"\n\n⚠ The **NMO Correction** step above has no usable velocity: {vs.error} Fix it there - "
              "this step stacks what it produces.")
        fig = plotting.plot_nmo(traces, traces, full_t, None, None, vs.unit, clip_pct=nmo_clip_pct, fig_height=fig_height,
                                title=f"CDP {cdp}", no_velocity_text="No velocity")
        return replace(state, view_extra=[markdown("CDP stack", md + "\n\n" + src.describe()),
                                          figure(f"CDP {cdp}", fig)]), f"CDP {cdp}: NMO above has no velocity"

    # ---- the chosen CDP: the previous step's output, stacked
    if vs.nmo:
        vel, info, vtxt = _cdp_velocity(vs, idx, k)
        gathers, stack_trace = nmo.nmo_stack(traces, nmo.cdp_offsets(cg.headers["offset"], vs.unit), full_t, vel,
                                             mute / 100.0)
        md = (head + " -- stack of the **NMO-corrected** gathers of the NMO Correction step above (" + vtxt
              + f" Stretch mute {mute:g} %): at every time the mean of the live (not muted) samples.")
    else:
        vel = info = None
        gathers, stack_trace = traces, np.mean(traces, axis=0)
        md = head + f" -- stack of the CDP gathers (no NMO Correction above): the plain mean of its {traces.shape[0]} traces."
    fig = plotting.plot_nmo(traces, gathers, full_t, vel, info, vs.unit, clip_pct=nmo_clip_pct, fig_height=fig_height,
                            title=f"CDP {cdp} - stack", stack_trace=stack_trace,
                            nmo_title="NMO-corrected gathers (input of the stack)" if vs.nmo else "CDP gathers (input of the stack)",
                            no_velocity_text="No NMO above")
    out = [figure(f"CDP {cdp} stack", fig)]

    # ---- the full stack, made by "Run flow on whole data" for this flow
    # only a stack of exactly the data the steps above produce: with processing steps above, the whole-data run's
    # processed file - never a stack of the raw file (that would show raw-data sections under processed gathers)
    stacked_file = src.processed if src.specs else src.file
    fs = (stack_mod.cached(stacked_file, idx, vs, mute, None, {"raw": False, "post_nmo": False, "stack": False})
          if stacked_file else None)
    if fs is None:
        md += ("\n\n**Run flow on whole data** (the 🌐 Whole data card) to stack every CDP of "
               + (f"the data processed by {' → '.join(src.labels)}" if src.specs else "the raw data")
               + " - the IL / XL stacked sections" + (", velocity sections and overlay" if vs.nmo else "")
               + " then show here.")
    elif vs.data_transform is None:
        md += "\n\nFull stack ready, but the IL / XL sections need the **Survey grid** of the 📁 Data card."
    else:
        il, xl = stack_mod.cdp_ilxl(idx, vs.data_transform)
        il_v = int(section_il) or stack_mod.most_common(il)
        xl_v = int(section_xl) or stack_mod.most_common(xl)
        data = fs.data
        il_rows, xl_rows = stack_mod.section_rows(il, xl, il_v), stack_mod.section_rows(xl, il, xl_v)
        il_res = (il_rows[1], np.asarray(data[il_rows[0]], np.float64)) if il_rows else None
        xl_res = (xl_rows[1], np.asarray(data[xl_rows[0]], np.float64)) if xl_rows else None
        md += (f"\n\n**Full stack:** {len(fs.cdps):,} CDPs of "
               + (f"the data processed by {' → '.join(src.labels)}" if src.specs else "the raw data")
               + f" stacked ({stack_mod.describe(vs, mute, None, {})}; made "
               f"{fs.meta.get('created', '')}, {fs.meta.get('seconds', 0):,.0f} s). Sections at IL {il_v} and XL {xl_v}"
               + ("" if il_res and xl_res else " - one of them has no CDPs: pick another IL / XL") + ".")
        out.append(figure(f"Stacked sections IL {il_v} / XL {xl_v}",
                          plotting.plot_stack_sections(il_res, xl_res, il_v, xl_v, full_t, clip_pct=nmo_clip_pct,
                                                       fig_height=fig_height + 0.5)))
        if vs.nmo:
            def _vel(rows_res):
                return None if rows_res is None else (rows_res[1], stack_mod.velocity_rows(vs, idx, rows_res[0],
                                                                                           section_velocity_type))
            il_vel, xl_vel = _vel(il_rows), _vel(xl_rows)
            out.append(figure(f"Velocity sections IL {il_v} / XL {xl_v}",
                              plotting.plot_velocity_sections(il_vel, xl_vel, il_v, xl_v, full_t, section_velocity_type,
                                                              fig_height=fig_height + 0.5)))
            ov = lambda s_, v_: None if s_ is None else (s_[0], s_[1], v_[1])
            out.append(figure(f"Stack + velocity IL {il_v} / XL {xl_v}",
                              plotting.plot_overlay_sections(ov(il_res, il_vel), ov(xl_res, xl_vel), il_v, xl_v, full_t,
                                                             section_velocity_type, clip_pct=nmo_clip_pct,
                                                             fig_height=fig_height + 0.5)))
    md += "\n\n" + src.describe() + "".join(f"\n\n{n}" for n in notes)
    out.insert(0, markdown("CDP stack", md))
    return replace(state, view_extra=out), (f"CDP {cdp} stacked" + (f"; full stack {len(fs.cdps):,} CDPs" if fs else ""))
