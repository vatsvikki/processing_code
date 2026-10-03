"""CDP Recalculate: a standalone tool in the 🛠 Tools menu of the top bar (not a Flow step) that rebuilds the CDP number of every trace from its
source-receiver midpoint and an IL / XL grid typed in by the user, and can write a copy of the file with it.

    CalcCDPX = (SX + GX) / 2                 CalcCDPY = (SY + GY) / 2      (coordinates scaled by byte 71)
    IL       = round(IL min + (CalcCDPY - Y0) / IL bin)
    XL       = round(XL min + (CalcCDPX - X0) / XL bin)
    CDP      = IL * N_XL + XL                N_XL = XL max - XL min + 1

The grid is axis-aligned (IL along Y, XL along X). Only the trace headers are read. The grid of every run is saved
(~/.trace_qc/cdp_recalc_grids.json, or $TRACE_QC_HOME) and becomes the starting value of the inputs next time.
Nothing here is used by, or uses, the Flow or any other tool.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from . import segy_io
from .registry import Output, Param, markdown, register, table
from .segy_io import header_word, set_header_word

_SCALAR, _SRC_X, _SRC_Y, _REC_X, _REC_Y = (71, "i2"), (73, "i4"), (77, "i4"), (81, "i4"), (85, "i4")
_CDP, _INLINE, _XLINE = (21, "i4"), (189, "i4"), (193, "i4")
_CHUNK = 100_000                                  # traces per header read / per write block (headers only: ~24 MB)
_MAX_POINTS = 20_000                              # interactive scatter plots: every n-th trace above this
FIG_WIDTH, FIG_HEIGHT = 900, 600                  # all four plots the same size (as in the notebook)
_MAX_ROWS = 5_000                                 # per-trace table: first traces only

_BUILTIN = {"cdp_il_min": 953, "cdp_il_max": 1304, "cdp_xl_min": 950, "cdp_xl_max": 1585,
            "cdp_x0": 1971388.1, "cdp_y0": 428622.0, "cdp_il_bin": 150.0, "cdp_xl_bin": 150.0}


# ---------------------------------------------------------------------------
# saved grid: the inputs of the last run are the starting values next time
# ---------------------------------------------------------------------------
def store_path() -> Path:
    return Path(os.environ.get("TRACE_QC_HOME") or Path.home() / ".trace_qc") / "cdp_recalc_grids.json"


def _read_store() -> dict:
    try:
        return json.loads(store_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_grid(path: str, grid: dict) -> None:
    store = _read_store()
    store.setdefault("files", {})[segy_io.norm_path(path)] = grid
    store["last"] = grid
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.replace(p)


class _SavedParam(Param):
    """A Param whose default is the value saved by the last run (read each time a widget is built)."""

    @property
    def default(self):
        return _read_store().get("last", {}).get(self.key, self._fallback)

    @default.setter
    def default(self, value):
        self._fallback = value


def _grid_param(key: str, label: str, kind: str, step: float, help: str) -> Param:
    return _SavedParam(key, label, kind, _BUILTIN[key], step=step, group="IL / XL grid", help=help)


GRID = [
    _grid_param("cdp_il_min", "IL min", "int", 1, "First inline of the grid"),
    _grid_param("cdp_il_max", "IL max", "int", 1, "Last inline of the grid"),
    _grid_param("cdp_xl_min", "XL min", "int", 1, "First crossline of the grid"),
    _grid_param("cdp_xl_max", "XL max", "int", 1, "Last crossline of the grid"),
    _grid_param("cdp_x0", "X at (IL min, XL min)", "float", 0.1, "X of the first corner"),
    _grid_param("cdp_y0", "Y at (IL min, XL min)", "float", 0.1, "Y of the first corner"),
    _grid_param("cdp_il_bin", "IL bin size (along Y)", "float", 0.01, "Distance between neighbouring inlines"),
    _grid_param("cdp_xl_bin", "XL bin size (along X)", "float", 0.01, "Distance between neighbouring crosslines"),
]
OUTPUT = [
    Param("cdp_write", "Write a SEG-Y with the recalculated CDP", "bool", False, group="Output file",
          help="Copy of the input with only the CDP word (bytes 21-24) replaced. Takes a while on a large file."),
    Param("cdp_out", "Output SEG-Y file", "text", "", group="Output file",
          help="Full path; empty = <input name>_recalc_cdp.sgy next to the input"),
]


# ---------------------------------------------------------------------------
# headers of every trace (cached per file version, so a new grid re-runs at once)
# ---------------------------------------------------------------------------
def _scale(raw: np.ndarray, order: str) -> np.ndarray:
    sc = header_word(raw, *_SCALAR, order).astype(float)
    f = np.ones(len(sc))
    f[sc > 0] = sc[sc > 0]
    f[sc < 0] = 1.0 / -sc[sc < 0]
    return f


@lru_cache(maxsize=2)
def _headers(path: str, mtime: float) -> dict[str, np.ndarray]:
    sgy = segy_io.open_segy(path)
    n = sgy.ntraces
    out = {k: np.empty(n, np.float64) for k in ("sx", "sy", "gx", "gy")}
    out.update({k: np.empty(n, np.int64) for k in ("cdp", "il", "xl")})
    for i0 in range(0, n, _CHUNK):
        raw = sgy.read_headers(i0, i0 + _CHUNK)
        s = slice(i0, i0 + len(raw))
        f = _scale(raw, sgy.order)
        for k, w in (("sx", _SRC_X), ("sy", _SRC_Y), ("gx", _REC_X), ("gy", _REC_Y)):
            out[k][s] = header_word(raw, *w, sgy.order) * f
        for k, w in (("cdp", _CDP), ("il", _INLINE), ("xl", _XLINE)):
            out[k][s] = header_word(raw, *w, sgy.order)
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _cell_mean(il, xl, values):
    """Mean of `values` per (IL, XL) cell -> (grid [n_il, n_xl] with NaN where empty, il range, xl range)."""
    i0, i1, j0, j1 = int(il.min()), int(il.max()), int(xl.min()), int(xl.max())
    ni, nj = i1 - i0 + 1, j1 - j0 + 1
    key = (il - i0) * nj + (xl - j0)
    cnt = np.bincount(key, minlength=ni * nj)
    tot = np.bincount(key, weights=values.astype(np.float64), minlength=ni * nj)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).reshape(ni, nj)
    return g, (i0, i1), (j0, j1)


def _inset_colorbar() -> dict:
    """Colour bar drawn inside the plot area, so it never changes the figure's width."""
    return dict(title=dict(text="CDP", font=dict(size=11)), x=0.98, xanchor="right", y=0.98, yanchor="top",
                len=0.4, thickness=14, outlinewidth=0, bgcolor="rgba(255,255,255,0.65)")


def _plot(title: str, fig_fn, png_fn, filename: str, label: str, note: str = "") -> Output:
    """kind "plotly": an interactive plotly figure (content["fig"]) plus a lazy full-resolution PNG for download.
    Drawn by the Tools-window page of the app (the main window's renderer does not know this kind)."""
    return Output("plotly", title, {"fig": fig_fn, "png": png_fn, "filename": filename, "label": label, "note": note})


def _scatter(x, y, colour, custom, hover: str, title: str, xtitle: str, ytitle: str):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scattergl(
        x=x, y=y, mode="markers",
        marker=dict(size=3, color=colour, colorscale="Viridis", colorbar=_inset_colorbar()),
        customdata=custom, hovertemplate=hover,
    ))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, yaxis=dict(scaleanchor="x", scaleratio=1),
                      width=FIG_WIDTH, height=FIG_HEIGHT, margin=dict(l=70, r=40, t=70, b=60))
    return fig


def _heatmap(il, xl, values, title: str, hover: str):
    import plotly.graph_objects as go

    g, (i0, i1), (j0, j1) = _cell_mean(il, xl, values)
    fig = go.Figure(go.Heatmap(x=np.arange(j0, j1 + 1), y=np.arange(i0, i1 + 1), z=g.astype(np.float32),
                               colorscale="Viridis",
                               colorbar=_inset_colorbar(), hovertemplate=hover))
    fig.update_layout(
        title=title, xaxis_title="XL (calculated)", yaxis_title="IL (calculated)",
        xaxis=dict(range=[j0 - 2, j1 + 2]), yaxis=dict(scaleanchor="x", scaleratio=1, range=[i0 - 2, i1 + 2]),
        width=FIG_WIDTH, height=FIG_HEIGHT, margin=dict(l=70, r=40, t=70, b=60),
    )
    return fig


def _png(fig) -> bytes:
    return fig.to_image(format="png", width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)


def _figures(h: dict, x, y, il, xl, cdp) -> list[Output]:
    """The four plots of plot_cdp_from_header_marimo.py: header CDP by midpoint / on the calculated IL-XL grid, and
    recalculated CDP by midpoint / on the grid. Scatter plots show every n-th trace (max 20 000) interactively; their
    PNG download uses every trace."""
    n = len(cdp)
    stride = max(1, n // _MAX_POINTS)
    idx = np.arange(0, n, stride)
    note = (f"*Showing {len(idx):,} of {n:,} traces (every {stride}th) for interactive display — the downloaded PNG "
            "uses every trace.*") if stride > 1 else ""

    # hover columns as int32 (coordinates rounded): half the size of float64 - marimo refuses a cell output > 8 MB,
    # and plot 1 with 20 000 points x 7 float64 columns was just over it
    hdr_custom = lambda k: np.rint(np.stack([h["il"][k], h["xl"][k], h["cdp"][k], h["sx"][k], h["sy"][k], h["gx"][k],
                                             h["gy"][k]], axis=1)).astype(np.int32)
    hdr_hover = ("X=%{x:,.1f}  Y=%{y:,.1f}<br>header IL=%{customdata[0]}  header XL=%{customdata[1]}<br>"
                 "header CDP=%{customdata[2]}<br>SX=%{customdata[3]:,}  SY=%{customdata[4]:,}<br>"
                 "GX=%{customdata[5]:,}  GY=%{customdata[6]:,}<extra></extra>")
    calc_custom = lambda k: np.stack([il[k], xl[k], cdp[k]], axis=1).astype(np.int32)
    calc_hover = ("X=%{x:,.1f}  Y=%{y:,.1f}<br>calc IL=%{customdata[0]}  calc XL=%{customdata[1]}<br>"
                  "calc CDP=%{customdata[2]}<extra></extra>")
    all_ = slice(None)

    def plot1(k=idx, title=f"1. CDP from header (showing {len(idx):,} of {n:,} traces, every {stride}th)"):
        return _scatter(x[k], y[k], h["cdp"][k].astype(np.int32), hdr_custom(k), hdr_hover, title,
                        "X (SX/GX midpoint)", "Y (SY/GY midpoint)")

    def plot3(k=idx, title=f"3. CDP recalculated (showing {len(idx):,} of {n:,} traces, every {stride}th)"):
        return _scatter(x[k], y[k], cdp[k].astype(np.int32), calc_custom(k), calc_hover, title, "CalcCDPX", "CalcCDPY")

    def plot2():
        return _heatmap(il, xl, h["cdp"], "2. CDP from header — positioned by calculated IL/XL",
                        "XL=%{x}<br>IL=%{y}<br>header CDP=%{z:.0f}<extra></extra>")

    def plot4():
        return _heatmap(il, xl, cdp, "4. CDP recalculated — positioned by calculated IL/XL",
                        "XL=%{x}<br>IL=%{y}<br>CDP=%{z:.0f}<extra></extra>")

    return [
        _plot("1. CDP from header — scatter, positioned by header coordinates", plot1,
              lambda: _png(plot1(all_, "1. CDP from header — positioned by header coordinates")),
              "cdp_from_header_scatter.png", "💾 Save full-resolution plot 1", note),
        _plot("2. CDP from header — map, positioned by calculated IL / XL", plot2, lambda: _png(plot2()),
              "cdp_header_by_calc_ilxl.png", "💾 Save plot 2"),
        _plot("3. CDP recalculated — scatter, positioned by CalcCDPX / CalcCDPY", plot3,
              lambda: _png(plot3(all_, "3. CDP recalculated — positioned by CalcCDPX/CalcCDPY")),
              "cdp_recalculated_scatter.png", "💾 Save full-resolution plot 3", note),
        _plot("4. CDP recalculated — map, positioned by calculated IL / XL", plot4, lambda: _png(plot4()),
              "cdp_recalculated_grid.png", "💾 Save plot 4"),
    ]


# ---------------------------------------------------------------------------
# writing the copy with the new CDP
# ---------------------------------------------------------------------------
def _write_copy(path: str, out: str, cdp: np.ndarray) -> tuple[str, float]:
    sgy = segy_io.open_segy(path)
    if not out.strip():
        p = Path(sgy.path)
        out = str(p.with_name(f"{p.stem}_recalc_cdp{p.suffix or '.sgy'}"))
    out = os.path.expanduser(out.strip())
    if not Path(out).suffix:
        out += ".sgy"
    if os.path.exists(out) and os.path.samefile(out, sgy.path):
        raise ValueError("the output file is the input file - choose another name")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vals = cdp.astype(np.int32)
    with open(sgy.path, "rb") as fin, open(out, "wb") as fout:
        fout.write(fin.read(sgy.data_start))                    # text + binary (+ extended) headers, unchanged
        for i0 in range(0, sgy.ntraces, _CHUNK):
            n = min(_CHUNK, sgy.ntraces - i0)
            blk = np.fromfile(fin, dtype=np.uint8, count=n * sgy.trace_bytes)
            if len(blk) != n * sgy.trace_bytes:
                raise IOError("short read while copying the traces")
            blk = blk.reshape(n, sgy.trace_bytes)
            set_header_word(blk[:, :segy_io.TRACE_HEADER_BYTES], _CDP[0], vals[i0:i0 + n], _CDP[1], sgy.order)
            blk.tofile(fout)
        fout.write(fin.read())                                  # any bytes after the last full trace
    return out, os.path.getsize(out) / 1e9


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------
@register("CDP Recalculate", "Recalculate the CDP of every trace from its midpoint and an IL / XL grid you enter; "
          "corner points, maps, and an optional copy of the file with the new CDP. Independent of the Flow.",
          params=GRID + OUTPUT, order=12, autorun=False, category="Tools")
def cdp_recalculate(path: str, cdp_il_min: int = 953, cdp_il_max: int = 1304, cdp_xl_min: int = 950,
                    cdp_xl_max: int = 1585, cdp_x0: float = 1971388.1, cdp_y0: float = 428622.0,
                    cdp_il_bin: float = 150.0, cdp_xl_bin: float = 150.0, cdp_write: bool = False, cdp_out: str = ""):
    bad = [m for ok, m in [(cdp_il_max >= cdp_il_min, "IL max must be ≥ IL min"),
                           (cdp_xl_max >= cdp_xl_min, "XL max must be ≥ XL min"),
                           (cdp_il_bin > 0 and cdp_xl_bin > 0, "bin sizes must be > 0")] if not ok]
    if bad:
        raise ValueError("grid not valid: " + "; ".join(bad))
    grid = {"cdp_il_min": int(cdp_il_min), "cdp_il_max": int(cdp_il_max), "cdp_xl_min": int(cdp_xl_min),
            "cdp_xl_max": int(cdp_xl_max), "cdp_x0": float(cdp_x0), "cdp_y0": float(cdp_y0),
            "cdp_il_bin": float(cdp_il_bin), "cdp_xl_bin": float(cdp_xl_bin)}
    _save_grid(path, grid)

    sgy = segy_io.open_segy(path)
    h = _headers(sgy.path, os.path.getmtime(sgy.path))
    n_xl = cdp_xl_max - cdp_xl_min + 1
    x, y = (h["sx"] + h["gx"]) / 2.0, (h["sy"] + h["gy"]) / 2.0
    il = np.round(cdp_il_min + (y - cdp_y0) / cdp_il_bin).astype(np.int64)
    xl = np.round(cdp_xl_min + (x - cdp_x0) / cdp_xl_bin).astype(np.int64)
    cdp = il * n_xl + xl

    ntr = len(cdp)
    outside = int(np.count_nonzero((il < cdp_il_min) | (il > cdp_il_max) | (xl < cdp_xl_min) | (xl > cdp_xl_max)))
    same = int(np.count_nonzero(cdp == h["cdp"]))
    x1 = cdp_x0 + (cdp_xl_max - cdp_xl_min) * cdp_xl_bin
    y1 = cdp_y0 + (cdp_il_max - cdp_il_min) * cdp_il_bin

    eq = (f"CalcCDPX = (SX + GX) / 2,   CalcCDPY = (SY + GY) / 2\n"
          f"IL       = round({cdp_il_min} + (CalcCDPY - {cdp_y0:.10g}) / {cdp_il_bin:.10g})\n"
          f"XL       = round({cdp_xl_min} + (CalcCDPX - {cdp_x0:.10g}) / {cdp_xl_bin:.10g})\n"
          f"N_XL     = {cdp_xl_max} - {cdp_xl_min} + 1 = {n_xl}\n"
          f"CDP      = IL * N_XL + XL")
    # summary: key numbers as tiles (styles: .qc-tiles in the app's custom.css), the grid warning, the equations
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tile = lambda name, value, warn=False: (f'<div class="qc-tile{" qc-tile-warn" if warn else ""}">'
                                            f'<span>{esc(name)}</span><b>{esc(value)}</b></div>')
    md = ('<div class="qc-tiles">'
          + tile("Traces", f"{ntr:,}")
          + tile("Grid", f"{cdp_il_max - cdp_il_min + 1:,} IL × {n_xl:,} XL")
          + tile("Recalculated CDP", f"{int(cdp.min()):,} – {int(cdp.max()):,}")
          + tile("Outside the grid", f"{outside:,}", warn=outside > 0)
          + tile("Header CDP = recalculated", f"{same:,} ({100.0 * same / max(ntr, 1):.1f} %)")
          + "</div>\n\n"
          + (f"⚠️ **{outside:,} traces** fall outside IL {cdp_il_min}–{cdp_il_max} / XL {cdp_xl_min}–{cdp_xl_max} "
             "— check the grid.\n\n" if outside else "✅ All traces fall inside the grid.\n\n")
          # one equation per line (a ``` block loses its line breaks once the page embeds it)
          + '<div class="qc-eq">' + "<br>".join(esc(line) for line in eq.splitlines()) + "</div>\n\n"
          + f'<span style="opacity:.65;font-size:.85rem">The grid is saved and filled in again next time '
            f"({esc(store_path())}).</span>")

    corners = [
        {"Corner": 1, "IL": cdp_il_min, "XL": cdp_xl_min, "X": f"{cdp_x0:,.2f}", "Y": f"{cdp_y0:,.2f}"},
        {"Corner": 2, "IL": cdp_il_min, "XL": cdp_xl_max, "X": f"{x1:,.2f}", "Y": f"{cdp_y0:,.2f}"},
        {"Corner": 3, "IL": cdp_il_max, "XL": cdp_xl_max, "X": f"{x1:,.2f}", "Y": f"{y1:,.2f}"},
        {"Corner": 4, "IL": cdp_il_max, "XL": cdp_xl_min, "X": f"{cdp_x0:,.2f}", "Y": f"{y1:,.2f}"},
    ]
    k = min(ntr, _MAX_ROWS)
    cols = ["Trace", "Header IL", "Header XL", "Header CDP", "X", "Y", "Calc IL", "Calc XL", "Calc CDP"]
    rows = [{"Trace": i + 1, "Header IL": int(h["il"][i]), "Header XL": int(h["xl"][i]), "Header CDP": int(h["cdp"][i]),
             "X": round(float(x[i]), 1), "Y": round(float(y[i]), 1), "Calc IL": int(il[i]), "Calc XL": int(xl[i]),
             "Calc CDP": int(cdp[i])} for i in range(k)]

    out = [markdown("Summary", md)]
    if cdp_write:
        dest, gb = _write_copy(path, cdp_out, cdp)
        out.append(markdown("Output file", f"✅ Wrote **{ntr:,} traces** ({gb:.2f} GB) to `{dest}` - CDP (bytes 21-24) "
                                           "replaced with the recalculated value, everything else unchanged."))
    out += [
        table("Corner points", ["Corner", "IL", "XL", "X", "Y"], corners),
        *_figures(h, x, y, il, xl, cdp),
        table(f"Every trace - header vs recalculated (first {k:,} of {ntr:,})", cols, rows, paginate=True),
    ]
    return out
