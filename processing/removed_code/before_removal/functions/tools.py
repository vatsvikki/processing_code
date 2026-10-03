"""The functions that appear as buttons in the GUI.

Each one is `fn(path, **params) -> list[Output]`, registered with @register.
They only orchestrate segy_io / detection / plotting -- nothing here knows about
marimo (or any other GUI).  Add a new tool by writing another decorated function.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import correction as dead_correction
from . import detection, ebcdic, fold as fold_mod, geometry, grid as grid_mod, pipeline as pipe, sortqc, stacking, plotting, segy_io, steps, survey  # noqa: F401  (steps registers itself)
from .registry import Param, data, get_functions, image, markdown, register, table  # noqa: F401
from .saving import DEFAULT_DIR
from .segy_write import ProcessedShot

from .params import (ANALYSIS, BAD, CORRECT, DEAD, DECON, DECON_PRE, DECON_VIEW, DISPLAY, SHOT, SURVEY)  # noqa: F401

_DISPLAY_KEYS = {p.key for p in DISPLAY} | {"sort_by"}
_QC_KEYS = {p.key for p in ANALYSIS + DEAD + BAD} | {"sort_by"}
_SURVEY_KEYS = {p.key for p in SURVEY + ANALYSIS + DEAD + BAD}


def figure(title: str, fig, zoom: bool = False):
    """Image output that keeps its Figure (so Save can write PDF / SVG / high-dpi PNG).

    zoom=True marks a shot gather that a GUI may let the user zoom by dragging a box on it."""
    return image(title, plotting.figure_to_png(fig), figure=fig, zoom=zoom)


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
        "Pick a tool above. **EBCDIC & Headers** shows the text header, binary header and trace headers."
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
    Param("corners", "Survey corner points  (table: IL, XL, X, Y per row)", "table", grid_mod.table_rows([], 4),
          auto="grid_corners", group="Inline / crossline grid",
          help="Filled in from the IL / XL of the headers when they describe a grid, or from the table saved for this "
               "file. Change any value - or type your own corners - and press Apply to use your IL / XL instead of the "
               "headers'; a table you changed is saved for this file and offered again next time. 3-4 corners; X, Y may "
               "be left empty when the headers give a grid; empty rows are ignored."),
    Param("forget_saved", "Back to the header IL / XL  (forget the saved table)", "bool", False, group="Inline / crossline grid"),
    Param("rec_shots", "Shots read for the receiver positions  (0 = all)", "int", 60, min=0, step=10,
          help="Receiver positions come from this many evenly spaced shots. Sources are always taken from every "
               "shot. 60 is enough for a fixed receiver patch; 0 reads every trace header of the file (~20 s "
               "for 950 000 traces) - use it when the receivers move", group="Geometry"),
]


def _resolve_grid(path: str, geom, corners, forget_saved: bool, rec_shots: int = 60):
    """The IL / XL grid for a map: the corner table the user filled in wins over the IL / XL of the headers.
    Returns (grid or None, [notes for the user]); saves / forgets the table as needed."""
    # ---- inline / crossline grid: the table the user filled in wins over the IL / XL of the headers
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
                         + ("💾 Saved for this file" if fresh else "Saved earlier for this file")
                         + f" ({grid_mod.store_path()}) and offered again next time; tick *Back to the header IL / XL* and "
                         "press Apply to drop it.")
            if grid.rms > 0.05 * min(grid.il_spacing, grid.xl_spacing):
                notes.append(f"⚠ **The corner points do not fit a regular grid** (best fit is off by {grid.rms:,.1f} "
                             f"{geom.unit or 'units'} on average): check the IL, XL, X and Y values of every row.")
        except ValueError as e:
            notes.append(f"⚠ **Table not used:** {e}")
    if grid is None and header_grid is not None:
        grid = header_grid
        what = ("**IL = receiver line** (RECLN, byte 173) and **XL = receiver station** (RECSTN, byte 181) of the receivers"
                if grid.source == "receiver headers" else "the inline / crossline words of the trace headers")
        notes.append(f"IL / XL taken from the headers: {what}. Change values in the table and press Apply to use your "
                     "own IL / XL instead.")
    elif grid is None:
        notes.append(f"⚠ **No inline / crossline grid in the headers:** {why}. Fill in the **corner points table** on the "
                     "left (one corner per row: IL, XL, X, Y - e.g. 1001, 1001, 1971388.1, 428622) and press Apply.")

    return grid, notes


@register("Acquisition Geometry", "Plan view of the acquisition: every source position and the receiver positions, "
          "with the selected shot's spread highlighted (coordinates from the trace headers).",
          params=[SHOT[0], _FIG_HEIGHT] + GEOMETRY, order=25, category="Display")
def acquisition_geometry(path: str, ffid: int = 0, fig_height: float = 7.0, show_sources: bool = True,
                         show_receivers: bool = True, show_selected: bool = True, show_grid: bool = True,
                         ilxl_ticks: bool = False, corners=None, forget_saved: bool = False, rec_shots: int = 60):
    """`corners`: the corner table (rows keyed IL / XL / X / Y) or text with one `IL, XL, X, Y` per line."""
    geom = geometry.read_geometry(path, rec_shots)
    spread = geometry.read_spread(path, ffid) if show_selected else None
    grid, notes = _resolve_grid(path, geom, corners, forget_saved, rec_shots)

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
    return out


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
    _GEO["corners"], _GEO["forget_saved"],
]


@register("Acquisition Fold", "Fold map: traces per bin from the source-receiver midpoints of EVERY trace, coloured on the "
          "same canvas as the geometry plot (colour bar inside the figure). Bins = the IL / XL grid cells. The first run "
          "reads all trace headers (~20 s for 950 000 traces); after that changes are instant.",
          params=[_FIG_HEIGHT] + FOLD, order=26, autorun=False, category="Display")
def acquisition_fold(path: str, fig_height: float = 7.0, show_sources: bool = False, show_receivers: bool = False,
                     show_grid: bool = True, ilxl_ticks: bool = False, cmap: str = "viridis", fold_max: int = 0,
                     bin_align: str = "auto", corners=None, forget_saved: bool = False):
    geom = geometry.read_geometry(path, 60)
    grid, notes = _resolve_grid(path, geom, corners, forget_saved, 60)
    if ilxl_ticks and grid is None:
        notes.append("IL / XL tick labels need a grid - the axes stay in X / Y until corner points are entered.")
    bin_size = fold_mod.auto_bin_size(geom)
    if grid is None:
        notes.append(f"No IL / XL grid, so the fold is counted in square X / Y bins of {bin_size:g} {geom.unit} (half the "
                     "receiver spacing). Fill in the corner points table to count it per IL / XL cell instead.")
    fold_map = fold_mod.compute_fold(path, grid, bin_size, bin_align)
    out = [markdown("Inline / crossline grid", "  \n".join(notes))]
    out.append(figure("Fold map", plotting.plot_fold(
        fold_map, geom, grid=grid, show_grid=show_grid, ilxl_ticks=ilxl_ticks, show_sources=show_sources,
        show_receivers=show_receivers, cmap=cmap, vmax=fold_max, fig_height=fig_height)))
    out.append(table("Fold statistics", ["Item", "Value"], fold_mod.summary_rows(fold_map)))
    if grid is not None:
        out.append(table("Inline / crossline grid", ["Item", "Value"], grid_mod.grid_rows(grid, geom.unit)))
    return out


# ---------------------------------------------------------------------------
# CMP Sort (Processing class): the CMP bins of the IL / XL grid, the sort by CDP / offset, and its QC
# ---------------------------------------------------------------------------
_BINS_CACHE: dict = {}


def _bins_for(path: str, grid, align: str):
    """CMP bins of the grid (kept for the last two grids / alignments)."""
    p = segy_io.norm_path(path)
    key = (p, os.path.getmtime(p), tuple(np.round(grid.a, 6).ravel()), grid.il_range, grid.xl_range, align)
    if key not in _BINS_CACHE:
        if len(_BINS_CACHE) >= 2:
            _BINS_CACHE.clear()
        _BINS_CACHE[key] = fold_mod.assign_bins(path, grid, align)
    return key, _BINS_CACHE[key]


def _cmp_setup(path: str, corners, forget_saved: bool, bin_align: str):
    """(geometry, grid, notes, bins) - grid / bins are None when no IL / XL grid is known."""
    geom = geometry.read_geometry(path, 60)
    grid, notes = _resolve_grid(path, geom, corners, forget_saved, 60)
    if grid is None:
        return geom, None, notes, None
    return geom, grid, notes, _bins_for(path, grid, bin_align)[1]


_CLIP = next(p for p in DISPLAY if p.key == "clip_pct")
_BIN_ALIGN = Param("bin_align", "CMP bin alignment", "choice", "auto", choices=fold_mod.ALIGN_CHOICES, group="CMP bins",
                   help="As in the fold map: auto puts the midpoints at the bin centres")
_CMP_BIN = [
    Param("il", "IL of the CMP bin to show  (0 = the highest fold)", "float", 0.0, min=0, step=0.5, live=True, group="CMP bin"),
    Param("xl", "XL of the CMP bin to show  (0 = the highest fold)", "float", 0.0, min=0, step=0.5, live=True, group="CMP bin",
          help="IL and XL of the bin centre; the bin containing them is used"),
]
_SORT_BY = Param("cmp_sort_by", "Sort traces by", "choice", "CMP bin, then offset", choices=list(sortqc.SORT_KEYS),
                 group="CMP sort", help="The CMP sort of the brute stack: CMP bin, then offset inside a bin (the default)")
_QC_PARAMS = [
    Param("low_fold_pct", "QC: low-fold limit, % of the median fold", "float", 25.0, min=0, max=100, step=5, group="QC"),
    Param("offset_tol_pct", "QC: offset header tolerance, %", "float", 2.0, min=0, max=50, step=0.5, group="QC"),
]
_GRID_PARAMS = [_GEO["corners"], _GEO["forget_saved"]]


def _cmp_view(cg, key: str):
    """The CMP gather in the order asked for: by offset (cmp_gather already gives that), or - for the bin only ("CMP bin (cdp)"
    puts no order inside a bin) - as recorded in the file, labelled by the header CDP."""
    order = np.argsort(cg.trace_index, kind="stable") if key == "cdp" else np.arange(cg.ntr)
    g = segy_io.ShotGather(ffid=int(cg.ffid[0]), i0=int(cg.trace_index[0]), data=cg.data[order],
                           headers={h: v[order] for h, v in cg.headers.items()}, dt_ms=cg.dt_ms)
    return g, order


def _cmp_figure(cg, clip_pct: float, fig_height: float, key: str, label: str):
    g, _ = _cmp_view(cg, key)
    title = f"CMP gather  IL {cg.il:g} / XL {cg.xl:g}  -  sorted by {label}  -  {cg.ntr} traces"
    kw = dict(sort_by="file", label_by="cdp") if key == "cdp" else dict(sort_by="offset")
    return figure("CMP gather", plotting.plot_gather(g, None, clip_pct=clip_pct, fig_height=fig_height, title=title, **kw))


def _cmp_table(cg, key: str):
    g, order = _cmp_view(cg, key)
    rows = [{"Trace": k + 1, "FFID": int(cg.ffid[j]), "File trace": int(cg.trace_index[j]) + 1, "Header CDP": int(cg.headers["cdp"][j]),
             "Offset": int(abs(cg.offsets[j]))} for k, j in enumerate(order)]
    return table("Traces of the CMP bin (in sort order)", ["Trace", "FFID", "File trace", "Header CDP", "Offset"], rows, paginate=True)


@register("CMP Sort", "Sort the traces into CMP bins - the IL / XL grid cells of the fold map, as in the brute stack (takes the "
          "corner-points table) - and by offset inside a bin, by bin only, or by offset only. Shows the CMP gather of one bin "
          "and QC that the sort is proper: bins in order, offsets ascending in every bin, fold, holes, offset coverage, "
          "duplicates, offset header against the coordinates, and whether traces that neighbour in offset look alike. The first "
          "run reads the headers of every trace (~20 s), then it is instant.",
          params=[_SORT_BY] + _CMP_BIN + _QC_PARAMS + [_FIG_HEIGHT, _CLIP, _BIN_ALIGN] + _GRID_PARAMS, order=70,
          autorun=False, category="Processing")
def cmp_sort(path: str, cmp_sort_by: str = "CMP bin, then offset", il: float = 0.0, xl: float = 0.0, low_fold_pct: float = 25.0,
             offset_tol_pct: float = 2.0, fig_height: float = 7.0, clip_pct: float = 98.0, bin_align: str = "auto",
             corners=None, forget_saved: bool = False):
    geom, grid, notes, bins = _cmp_setup(path, corners, forget_saved, bin_align)
    if grid is None:
        return [markdown("Inline / crossline grid", "  \n".join(notes + ["The CMP sort puts the traces into the bins of an "
                                                                          "IL / XL grid: fill in the corner points table."]))]
    key = sortqc.SORT_KEYS[cmp_sort_by]
    head = markdown("Inline / crossline grid", "  \n".join(notes))
    qc = sortqc.run_sort_qc(path, bins, key, low_fold_pct, offset_tol_pct)
    verdict = (f"QC: **{len(qc.rows)} checks, {qc.n_warn} to look at**" if qc.n_warn else f"QC: **{len(qc.rows)} checks, all fine**")
    try:
        cg = stacking.cmp_gather(path, bins, il, xl)
    except LookupError as e:
        return [head, markdown("CMP bin", f"⚠ **{e}**  \n{verdict}")]
    off = np.abs(cg.offsets)
    live = bins.counts[bins.counts > 0]
    u = f" {geom.unit}" if geom.unit else ""
    md = (f"**CMP bin IL {cg.il:g} / XL {cg.xl:g}** · fold **{cg.ntr}** (from {len(np.unique(cg.ffid))} shots) · offsets "
          f"{off.min():,.0f} - {off.max():,.0f}{u}  \n"
          f"{live.size:,} bins with data · mean fold {live.mean():.1f} · maximum {live.max()} · {bins.alignment}  \n{verdict}")
    # the sorted order of the whole file, shown from the selected bin on (the bins follow each other in the sorted file)
    tab = geometry.read_trace_table(path)
    bid = sortqc.bin_ids(bins)
    start = 0
    if key in ("cdp", "cdp_offset"):
        si, sj = bins.locate(cg.il, cg.xl)
        start = int(np.searchsorted(bid[qc.order], si * bins.shape[1] + sj))
    order_rows = []
    for k, i in enumerate(qc.order[start:start + 300]):
        inb = bins.ii[i] >= 0
        order_rows.append({"Order": start + k + 1, "File trace": int(i) + 1,
                           "CMP bin IL": f"{float(bins.il_of(bins.ii[i])):g}" if inb else "-",
                           "CMP bin XL": f"{float(bins.xl_of(bins.jj[i])):g}" if inb else "-",
                           "Header CDP": int(tab["cdp"][i]), "FFID": int(tab["key"][i] // 100000),
                           "Channel": int(tab["key"][i] % 100000), "Offset": int(abs(tab["offset"][i]))})
    where = "from the selected bin" if start else "first"
    return [head, markdown("CMP bin", md), _cmp_figure(cg, clip_pct, fig_height, key, cmp_sort_by), _cmp_table(cg, key),
            figure("CMP sort QC", plotting.plot_sort_qc(qc)),
            table("CMP sort QC checks", ["Status", "Check", "Result", "Meaning"], qc.rows),
            table(f"Sorted order of the whole file by {cmp_sort_by} ({where} 300 of {len(qc.order):,} traces)",
                  ["Order", "File trace", "CMP bin IL", "CMP bin XL", "Header CDP", "FFID", "Channel", "Offset"], order_rows,
                  paginate=True)]


# ---------------------------------------------------------------------------
# the processing steps of the pipeline as tools of their own (Processing class): one shot, before / after
# ---------------------------------------------------------------------------
def _step_tool(step_key: str, label: str, order: int, autorun: bool = True, defaults: dict | None = None):
    """Register pipeline step `step_key` as a tool: it runs that one step on the selected shot and shows the result the way
    the Pipeline does (gather, spectrum / autocorrelation vs the input, Save data, Apply to whole data)."""
    st = pipe.get_step(step_key)

    def tool(path: str, ffid: int = 0, **kw):
        g = segy_io.read_shot(path, ffid)
        flow = [{"step": step_key, "params": {p.key: kw[p.key] for p in st.params if kw.get(p.key) is not None}}]
        return _pipeline_outputs(path, pipe.run_steps(g, flow, path=path), 1, kw, flow)

    tool.__name__ = f"{step_key}_tool"
    params = [replace(p, default=defaults[p.key]) if defaults and p.key in defaults else p for p in st.params]
    register(label, st.description, params=SHOT + params + DISPLAY, order=order, autorun=autorun, kind="flowstep",
             category="Processing", step_key=step_key)(tool)


# every processing step is a function of the Processing tab; clicking it adds it to the flow (see the top bar)
for _key, _label, _order in (("correct_dead", "Correct Dead Traces", 35), ("geometric_spreading", "Geometric Spreading", 38),
                             ("spiking_decon", "Spiking Decon", 45), ("bandpass_filter", "Bandpass Filter", 46),
                             ("agc_gain", "AGC", 50), ("top_mute", "Top Mute", 55), ("sort_traces", "Sort Traces", 60),
                             ("nmo_step", "NMO Correction", 65), ("stack_traces", "Stack Traces", 68),
                             ("save_data_step", "Save Data", 90)):
    _step_tool(_key, _label, _order, defaults={"write": True} if _key == "save_data_step" else None)


@register("Plot Shot Gather", "Plain shot gather image (no QC).", params=SHOT + DISPLAY, order=20, category="Display")
def plot_shot_gather(path: str, ffid: int = 0, **kw):
    g = segy_io.read_shot(path, ffid)
    fig = plotting.plot_gather(g, None, **_pick(kw, _DISPLAY_KEYS))
    return [figure(f"Shot gather - FFID {g.ffid}", fig, zoom=True)]


def _qc(path, ffid, check_dead, check_bad, kw):
    g = segy_io.read_shot(path, ffid)
    res = detection.run_qc(g, check_dead=check_dead, check_bad=check_bad, **_pick(kw, _QC_KEYS))
    n_dead, n_bad = int(res.dead.sum()), int(res.bad.sum())
    good = g.ntr - n_dead - n_bad
    md = (f"**FFID {g.ffid}** · {g.ntr} traces · "
          f"**dead: {n_dead}** · **bad: {n_bad}** · good: {good} ({good / g.ntr:.1%})")
    rows = detection.flagged_table(g, res)
    cols = ["position", "file_trace", "channel", "rec_line", "rec_stn", "offset", "class", "rms", "rms/ref", "reason"]
    out = [
        markdown("Result", md),
        figure("Shot gather - flagged traces marked", plotting.plot_gather(g, res, **_pick(kw, _DISPLAY_KEYS)),
               zoom=True),
        figure("Trace RMS vs. thresholds", plotting.plot_qc_metrics(g, res)),
    ]
    if "corr" in res.metrics:
        out.append(figure("Neighbour correlation vs. limit", plotting.plot_correlation_metrics(g, res)))
    if rows:
        out.append(table("Flagged traces", cols, rows, paginate=True))
    return out


@register("Dead Traces", "Find dead (zero / flat / near-zero) traces and mark them on the gather.",
          params=SHOT + ANALYSIS + DEAD + DISPLAY, order=30, category="QC")
def detect_dead_traces(path: str, ffid: int = 0, **kw):
    return _qc(path, ffid, True, False, kw)


@register("Bad Traces", "Find noisy, weak, spiky or DC-offset traces and mark them red on the gather.",
          params=SHOT + ANALYSIS + BAD + DISPLAY, order=40, category="QC")
def detect_bad_traces(path: str, ffid: int = 0, **kw):
    return _qc(path, ffid, False, True, kw)


@register("Dead + Bad QC", "Run all dead- and bad-trace tests together (bad = red, dead = blue).",
          params=SHOT + ANALYSIS + DEAD + BAD + DISPLAY, order=50, category="QC")
def trace_qc(path: str, ffid: int = 0, **kw):
    return _qc(path, ffid, True, True, kw)


@register("Whole-Survey QC", "Count the dead and bad traces in every shot of the whole file (takes a while).",
          params=SURVEY + ANALYSIS + DEAD + BAD, order=60, autorun=False, category="QC")
def survey_qc(path: str, **kw):
    r = survey.scan_survey(path, **_pick(kw, _SURVEY_KEYS))
    n, nd, nb = r.n_traces, r.n_dead, r.n_bad
    lo, hi = r.per_shot[0]["ffid"], r.per_shot[-1]["ffid"]
    md = (f"**FFID {lo} to {hi}** · {len(r.per_shot):,} shots · {n:,} traces scanned  \n"
          f"**dead: {nd:,}** ({nd / n:.3%}) · **bad: {nb:,}** ({nb / n:.3%}) · "
          f"good: {n - nd - nb:,} ({(n - nd - nb) / n:.3%})  \n"
          f"shots with at least one flagged trace: {sum(1 for x in r.per_shot if x['dead'] or x['bad']):,}")
    shot_cols = ["ffid", "traces", "dead", "bad", "good"]
    trace_cols = ["ffid", "position", "file_trace", "channel", "rec_line", "rec_stn", "offset", "class",
                  "rms", "rms/ref", "reason"]
    out = [
        markdown("Whole-survey result", md),
        figure("Dead and bad traces per shot", plotting.plot_survey_counts(r.per_shot)),
        table("Per-shot counts", shot_cols, r.per_shot, paginate=True),
    ]
    if r.flagged:
        out.append(table("All flagged traces", trace_cols, r.flagged, paginate=True))
    return out


# ---------------------------------------------------------------------------
# pipeline: steps chained in an order the user chooses
# ---------------------------------------------------------------------------
_DECON_KEYS = {p.key for p in DECON}


def _pipeline_outputs(path: str, stages, k: int, kw: dict, flow: list[dict]):
    """Result of a flow: the flow summary, the gather after stage k, and (k > 0) the spectrum / autocorrelation."""
    disp = _pick(kw, _DISPLAY_KEYS)
    sorts = [s for s in flow[:k] if s["step"] == "sort_traces"]
    if sorts and disp.get("sort_by", "file") == "file":      # the gather is in the order of a Sort step: label it by that key
        disp["label_by"] = steps._SORT_KEYS.get((sorts[-1].get("params") or {}).get("by", "offset"), "offset")
    st = stages[k]
    state, g = st.state, st.state.gather
    lines = []
    for i, s in enumerate(stages):
        label = s.label.replace(". ", "\\. ", 1)               # "1\. name": keep markdown from making a nested list
        lines.append(f"- {'**▶ ' if i == k else ''}{label}{'**' if i == k else ''} — {s.note}"
                     + (f" · {s.seconds * 1000:.0f} ms" if i else ""))
    md = f"**FFID {g.ffid}** · {stages[0].state.gather.ntr} traces in · {len(stages) - 1} step(s)\n\n" + "\n".join(lines)
    fig = plotting.plot_gather(
        g, pipe.flag_result(state, disp.get("sort_by", "file")), highlight=state.marks,
        highlight_label=state.marks_label or "marked trace",
        title=f"Shot gather  FFID {g.ffid}  -  {st.label}  -  {g.ntr} traces", **disp)
    out = [markdown("Flow", md), figure(f"Shot gather - {st.label}", fig, zoom=True)]
    if k > 0:
        out.append(figure(f"Spectrum and autocorrelation - {st.label} vs input", plotting.plot_spectrum_acorr(
            stages[0].state.gather.data, g.data, g.dt_ms, ("input", st.label))))
    notes = [f"{s.label}: {s.note}" for s in stages[1:k + 1]]
    out.append(data(f"Processed data - {st.label}", ProcessedShot(path, g, state.src, st.label, notes, flow=flow[:k])))
    return out


@register("Flow", "The functions clicked in the Processing tab, run from top to bottom on the selected shot.",
          params=SHOT + DISPLAY, order=15, kind="pipeline", category="Flow")
def pipeline(path: str, ffid: int = 0, steps: list | None = None, view: int = -1, **kw):
    """steps = [{"step": key, "params": {...}}, ...] (see pipeline.get_steps); view = stage to plot (-1 = last)."""
    g = segy_io.read_shot(path, ffid)
    stages = pipe.run_steps(g, steps or [], path=path)
    k = len(stages) - 1 if view is None or not 0 <= int(view) < len(stages) else int(view)
    return _pipeline_outputs(path, stages, k, kw, list(steps or []))


