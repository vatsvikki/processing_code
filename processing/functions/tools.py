"""The functions that appear as buttons in the GUI.

Each one is `fn(path, **params) -> list[Output]`, registered with @register.
They only orchestrate segy_io / detection / plotting -- nothing here knows about
marimo (or any other GUI).  Add a new tool by writing another decorated function.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import ebcdic, pipeline as pipe, plotting, segy_io, steps  # noqa: F401  (steps registers itself)
from .registry import Param, data, figure, get_functions, image, markdown, register, table  # noqa: F401
from .segy_write import ProcessedShot

from .params import DISPLAY, SHOT

_DISPLAY_KEYS = {p.key for p in DISPLAY} | {"sort_by"}


def _pick(kw: dict, keys: set) -> dict:
    return {k: v for k, v in kw.items() if k in keys}


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def _file_summary_md(sgy) -> str:
    s = ebcdic.file_summary(sgy)
    return (
        f"**{s['file']}**  \n"
        f"{s['size_gb']:.2f} GB · **{s['traces']:,}** traces · {s['samples']} samples × {s['dt_ms']:g} ms "
        f"(= {s['record_ms']:g} ms) · format {s['format']} · {s['byte_order']}  \n"
        f"FFID range **{s['first_ffid']} → {s['last_ffid']}** · extended text headers: {s['ext_headers']}"
    )


@register("Load Data", "Read the SEG-Y file (headers + shot index) with a progress bar. Press Load first: "
          "the other tools open the loaded file.", order=10, category="Data", kind="loader")
def load_data(path: str, progress=None):
    """kind="loader": the GUI shows a Load button and calls this once per press.
    progress(fraction 0..1, text) is optional; a GUI passes it to draw a progress bar."""
    def report(frac: float, text: str = ""):
        if progress is not None:
            progress(min(max(frac, 0.0), 1.0), text)

    t0 = time.perf_counter()
    report(0.0, "Opening the file ...")
    segy_io.clear_caches()                                   # a real read, also when the file was opened before
    sgy = segy_io.open_segy(path)                            # text header, binary header, trace count
    report(0.05, "Headers read - indexing the shots ...")
    idx = segy_io.shot_index(path, progress=lambda f, t="": report(0.05 + 0.90 * f, t), force=True)
    report(0.95, "Reading the first shot ...")
    segy_io.read_shot(path, int(idx.ffids[0]))               # warms the cache so the first plot is instant
    report(1.0, "Done")
    secs = time.perf_counter() - t0

    ntr = idx.count
    md = (
        f"### ✅ Data loaded in {secs:.1f} s\n\n" + _file_summary_md(sgy) + "  \n"
        f"**{len(idx):,} shots** · {int(ntr.min())} – {int(ntr.max())} traces per shot "
        f"(median {int(np.median(ntr))})\n\n"
        "Pick a function from **＋ Add a function** (Flow card, left). **EBCDIC & Headers** shows the text header, "
        "binary header and trace headers."
    )
    rows = [{"FFID": int(f), "First trace": int(a) + 1, "Traces": int(n)} for f, a, n in zip(idx.ffids, idx.first, ntr)]
    return [
        markdown("Load summary", md),
        table("Shots in the file", ["FFID", "First trace", "Traces"], rows, paginate=True),
    ]


@register("EBCDIC & Headers", "EBCDIC textual header, binary header and every trace-header word of the chosen trace "
          "(plus min / max of each word over its shot).",
          params=[SHOT[0], Param("trace", "Trace in shot", "int", 1, min=1, step=1, live=True, group="Shot",
                                 help="Trace position within the selected shot (1 = first)")],
          order=11, category="Data")
def file_headers(path: str, ffid: int = 0, trace: int = 1):
    sgy = segy_io.open_segy(path)
    idx = segy_io.shot_index(path)
    k = idx.position(idx.nearest(int(ffid)))
    ffid, i0, n = int(idx.ffids[k]), int(idx.first[k]), int(idx.count[k])
    trace = min(max(int(trace), 1), n)
    raw, _, _ = sgy.read_block(i0, i0 + n)                   # all 240-byte headers of the shot

    bin_rows = ebcdic.binary_header_rows(sgy) + [ebcdic.unassigned_binary_row(sgy)]
    out = [
        markdown("File summary", _file_summary_md(sgy)),
        table("EBCDIC textual header (40 × 80 characters)", ["Line", "Text"], ebcdic.ebcdic_rows(sgy.ebcdic), mono=("Text",)),
        table("Binary header (400 bytes)", ["Byte", "Field", "Value", "Description"], bin_rows),
    ]
    for j, lines in enumerate(sgy.ext_headers, 1):
        out.append(table(f"Extended textual header {j}", ["Line", "Text"], ebcdic.ebcdic_rows(lines), mono=("Text",)))
    out += [
        table(f"Trace header (240 bytes) - trace {trace} of {n} in FFID {ffid}  (file trace {i0 + trace:,})",
              ["Bytes", "Field", "Value", "Description"], ebcdic.decode_trace_header(raw[trace - 1], sgy.order),
              paginate=True),
        table(f"Trace headers of the whole shot - FFID {ffid}, {n} traces: range of every word",
              ["Bytes", "Field", "Min", "Max", "Distinct", "Varies", "Description"],
              ebcdic.trace_header_stats(raw, sgy.order), paginate=True),
    ]
    return out


# Acquisition Geometry, Acquisition Fold and Plot Shot Gather moved to functions/steps.py: they are functions of the
# Flow now (see "display function in chain" - chainable), not standalone tools here.
# GEOMETRY / FOLD / _resolve_grid live there too.


# ---------------------------------------------------------------------------
# Flow: the processing functions chained, added / removed / reordered in the UI
# ---------------------------------------------------------------------------
def _flow_outputs(path: str, stages, k: int, kw: dict, flow: list[dict]):
    """Result of a flow: the list of steps with their notes, then either the gather after stage k with the spectrum /
    autocorrelation against the input (k > 0), or - for a "display" step (Acquisition Geometry / Fold) - that
    step's own map figure instead (state.view_extra; the gather itself is unchanged by these steps)."""
    disp = _pick(kw, _DISPLAY_KEYS)
    st = stages[k]
    state, g = st.state, st.state.gather
    lines = []
    for i, s in enumerate(stages):
        label = s.label.replace(". ", "\\. ", 1)               # "1\. name": keep markdown from making a nested list
        lines.append(f"- {'**▶ ' if i == k else ''}{label}{'**' if i == k else ''} — {s.note}"
                     + (f" · {s.seconds * 1000:.0f} ms" if i else ""))
    md = f"**FFID {g.ffid}** · {stages[0].state.gather.ntr} traces in · {len(stages) - 1} step(s)\n\n" + "\n".join(lines)
    from . import cdp_flow
    _, _, problems = cdp_flow.split_flow(flow)
    if problems:                                   # the whole-data run refuses such a flow: say so here already
        md += "\n\n" + "\n\n".join(f"⚠ {p_}" for p_ in problems)
    out = [markdown("Flow", md)]
    if state.view_extra is not None:
        out += state.view_extra
    else:
        fig = plotting.plot_gather(
            g, pipe.flag_result(state, disp.get("sort_by", "file")), highlight=state.marks,
            highlight_label=state.marks_label or "marked trace",
            title=f"Shot gather  FFID {g.ffid}  -  {st.label}  -  {g.ntr} traces", **disp)
        out.append(figure(f"Shot gather - {st.label}", fig, zoom=True))
        if k > 0:
            out.append(figure(f"Spectrum and autocorrelation - {st.label} vs input", plotting.plot_spectrum_acorr(
                stages[0].state.gather.data, g.data, g.dt_ms, ("input", st.label))))
        out += state.figs or []                                # a step's own QC figures (F-K spectrum, Radon panel)
    notes = [f"{s.label}: {s.note}" for s in stages[1:k + 1]]
    out.append(data(f"Processed data - {st.label}", ProcessedShot(path, g, state.src, st.label, notes, flow=flow[:k])))
    return out


_FLOW_CACHE: dict = {}             # (file, FFID) -> [(signature of the steps that made it, Stage), ...], stage 0 first


def _signature(specs: list[dict]) -> str:
    return json.dumps(specs, sort_keys=True, default=str)


def flow_status(path: str, ffid: int, steps: list | None) -> int:
    """How many of the flow's steps have a result for this shot that is still valid (their settings unchanged)."""
    key = (segy_io.norm_path(path), int(ffid))
    cached = _FLOW_CACHE.get(key) or []
    n = 0
    for i in range(1, len(steps or []) + 1):
        if i < len(cached) and cached[i][0] == _signature((steps or [])[:i]):
            n = i
        else:
            break
    return n


@register("Flow", "Processing functions chained: click the functions to add them; ▶ beside a function runs the flow up "
          "to it on the selected shot (the steps above, already run with the same settings, are reused).",
          params=SHOT + DISPLAY, order=15, kind="pipeline", category="Processing")
def flow(path: str, ffid: int = 0, steps: list | None = None, view: int = -1, run_to: int | None = None, **kw):
    """steps = [{"step": key, "params": {...}}, ...]; view = the stage to plot (0 = the input, -1 = the last one).
    Nothing is computed unless run_to is given: run_to = k runs the flow up to stage k (-1 = all), reusing the stages
    already computed for this shot whose steps and settings are unchanged. Otherwise the stages computed earlier are
    shown (a stage not run yet - or whose settings changed since - shows the last valid one, with a note)."""
    steps = list(steps or [])
    g = segy_io.read_shot(path, ffid)
    key = (segy_io.norm_path(path), int(g.ffid))
    cached = _FLOW_CACHE.get(key) or []
    valid = [pipe.Stage("0. Input (raw)", pipe.PipeState.initial(g, path), f"{g.ntr} traces")] if not cached else [cached[0][1]]
    for i in range(1, len(steps) + 1):                          # still valid: same steps and settings above
        if i < len(cached) and cached[i][0] == _signature(steps[:i]):
            valid.append(cached[i][1])
        else:
            break
    if run_to is not None:
        target = len(steps) if int(run_to) < 0 else min(int(run_to), len(steps))
        # ▶ always runs the function it belongs to again (it may show data from outside the flow - e.g. the whole-data
        # run's stack - which can have changed); the functions above it are reused while their settings are unchanged
        valid = pipe.run_steps(g, steps[:target], path=path, start=valid[:max(1, target)])
    _FLOW_CACHE.clear() if len(_FLOW_CACHE) > 8 else None    # (a few shots kept; each stage holds a gather)
    _FLOW_CACHE[key] = [("", valid[0])] + [(_signature(steps[:i]), valid[i]) for i in range(1, len(valid))]
    want = len(steps) if view is None or not 0 <= int(view) <= len(steps) else int(view)
    k = min(want, len(valid) - 1)
    out = _flow_outputs(path, valid, k, kw, steps)
    if want > k:                                               # the stage asked for is not computed (yet / any more)
        names = ", ".join(f"{i}. {pipe.get_step(steps[i - 1]['step']).label}" for i in range(k + 1, want + 1))
        out.insert(0, markdown("Not run yet", f"⏸ **{names}** - not run yet for FFID {g.ffid} (or its settings "
                                              f"changed). Press **▶** beside it (or **▶ Run all**). Shown: stage {k}."))
    return out


# the standalone QC tools (Dead Traces, Bad Traces, Dead + Bad QC, Whole-Survey QC) were removed: the QC steps
# Detect Dead Traces / Detect Bad Traces (functions/steps.py) cover the same tests as functions of the Flow now.
# See removed_code/ for the earlier code (detection.run_qc / flagged_table, survey.scan_survey, the QC plots).
