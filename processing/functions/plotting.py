"""Matplotlib figures for shot-gather QC (no UI code, no pyplot global state).

Every function returns a `matplotlib.figure.Figure`; use `figure_to_png()` to
get bytes for any GUI (marimo, Qt, Streamlit ...) or `fig.savefig()` for a file.
"""
from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from . import decon
from .detection import QCResult
from .filters import agc
from .fold import FoldMap
from .geometry import Geometry, Spread
from .grid import Grid
from .segy_io import ShotGather, trace_order

# bad = red, dead = blue (slots 8 / 1 of the reference palette); marker SHAPE
# differs as well so the two classes never depend on colour alone.
BAD_COLOR = "#e34948"
DEAD_COLOR = "#2a78d6"
FIXED_COLOR = "#2b9a66"       # traces replaced by interpolation
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"


def figure_to_png(fig: Figure, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    FigureCanvasAgg(fig)
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return buf.getvalue()


def _sort_key_labels(gather: ShotGather, order: np.ndarray, sort_by: str, positions: np.ndarray) -> list[str]:
    h = gather.headers
    out = []
    for p in positions:
        i = order[p - 1]
        if sort_by == "offset":
            sec = f"{abs(int(h['offset'][i]))}"
        elif sort_by == "receiver":
            sec = f"L{int(h['rec_line'][i])}/{int(h['rec_stn'][i])}"
        elif sort_by in ("cdp", "cdp_offset"):
            sec = f"{int(h['cdp'][i])}"
        else:
            sec = f"ch{int(h['chan'][i])}"
        out.append(f"{p}\n{sec}")
    return out


def plot_gather(
    gather: ShotGather,
    result: QCResult | None = None,
    *,
    sort_by: str = "file",
    clip_pct: float = 98.0,
    agc_ms: float = 0.0,
    t_min_ms: float = 0.0,
    t_max_ms: float = 0.0,
    trace_min: int = 0,
    trace_max: int = 0,
    label_by: str | None = None,
    show_bad: bool = True,
    show_dead: bool = True,
    highlight: np.ndarray | None = None,
    highlight_label: str = "interpolated trace",
    title: str | None = None,
    fig_height: float = 8.0,
    figsize: tuple[float, float] | None = None,
    clip: float | None = None,
) -> Figure:
    """Shot gather as a grey-scale image; bad traces = red lines, dead = blue lines.
    clip = a fixed amplitude clip (instead of the clip_pct percentile) - the same for gathers to be compared.

    trace_min/trace_max are 1-based display positions (0 = whole gather);
    t_max_ms <= 0 means 'to the end of the record'.
    """
    order = result.order if result is not None else trace_order(gather, sort_by)
    ntr, ns, dt = gather.ntr, gather.ns, gather.dt_ms
    p0 = max(int(trace_min), 1) if trace_min else 1
    p1 = min(int(trace_max), ntr) if trace_max else ntr
    if p0 > ntr:
        raise ValueError(f"first trace position {p0} is beyond the last trace ({ntr})")
    if p1 < p0:
        raise ValueError(f"last trace position ({p1}) is before the first ({p0})")
    t0 = max(float(t_min_ms), 0.0)
    t1 = min(float(t_max_ms), ns * dt) if t_max_ms and t_max_ms > 0 else ns * dt
    if t0 >= t1 - dt:
        raise ValueError(f"plot time range {t0:g}-{t1:g} ms is empty (record is {ns * dt:g} ms long)")
    s0, s1 = int(round(t0 / dt)), int(round(t1 / dt))

    sel = order[p0 - 1:p1]
    img = gather.data[sel]
    if agc_ms and agc_ms > 0:
        img = agc(img, dt, agc_ms)
    img = img[:, s0:s1]

    flagged = np.zeros(len(sel), bool)
    if result is not None:
        flagged = result.flagged[sel]
    ref = np.abs(img[~flagged]) if (~flagged).any() else np.abs(img)
    if clip is None:
        clip = float(np.percentile(ref[:, ::2], clip_pct)) if ref.size else 1.0
    if not clip > 0:
        clip = float(ref.max()) if ref.size and ref.max() > 0 else 1.0

    fig = Figure(figsize=figsize or (14.0, float(fig_height)), facecolor="white")
    ax = fig.add_axes([0.065, 0.10, 0.905, 0.80])
    ax.imshow(
        img.T, aspect="auto", cmap="gray", vmin=-clip, vmax=clip,
        extent=[p0 - 0.5, p1 + 0.5, s1 * dt, s0 * dt], interpolation="antialiased",
    )
    ax.set_xlim(p0 - 0.5, p1 + 0.5)

    handles = []
    pos = np.arange(p0, p1 + 1)
    if highlight is not None and highlight[sel].any():
        m = highlight[sel]
        ax.vlines(pos[m], s0 * dt, s1 * dt, colors=FIXED_COLOR, linewidth=1.0, alpha=0.75, zorder=3)
        ax.plot(pos[m], np.ones(m.sum()), linestyle="none", marker="^", markersize=6, color=FIXED_COLOR,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=4)
        handles.append(Line2D([0], [0], color=FIXED_COLOR, marker="^", markersize=6, linestyle="none",
                              label=f"{highlight_label} ({int(m.sum())})"))
    if result is not None:
        for show, mask, color, marker, label in (
            (show_dead, result.dead, DEAD_COLOR, "D", "dead trace"),
            (show_bad, result.bad, BAD_COLOR, "v", "bad trace"),
        ):
            m = mask[sel]
            if not show or not m.any():
                continue
            ax.vlines(pos[m], s0 * dt, s1 * dt, colors=color, linewidth=1.0, alpha=0.85, zorder=3)
            ax.plot(pos[m], np.ones(m.sum()), linestyle="none", marker=marker, markersize=5,
                    color=color, transform=ax.get_xaxis_transform(), clip_on=False, zorder=4)
            handles.append(Line2D([0], [0], color=color, marker=marker, markersize=5, linewidth=1.5,
                                  label=f"{label} ({int(m.sum())})"))

    n_ticks = 12
    ticks = np.unique(np.linspace(p0, p1, min(n_ticks, p1 - p0 + 1)).round().astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels(_sort_key_labels(gather, order, label_by or sort_by, ticks), fontsize=8, color=INK_2)
    ax.tick_params(axis="y", labelsize=9, colors=INK_2)
    ax.set_ylabel("Time (ms)", color=INK_2)
    xl = {"offset": "offset", "channel": "channel", "receiver": "receiver line/station", "file": "shot order",
          "cdp": "CDP", "cdp_offset": "CDP, then offset"}
    if label_by and sort_by == "file":                 # a gather already sorted (e.g. a CDP gather): label by its key
        ax.set_xlabel(f"Trace position (sorted by {xl.get(label_by, label_by)}; 2nd row = {xl.get(label_by, label_by)} value)",
                      color=INK_2)
    else:
        ax.set_xlabel(f"Trace position (sorted by {xl.get(sort_by, sort_by)}; 2nd row = {xl.get(sort_by, sort_by)} value)"
                      if sort_by != "file" else "Trace position (common-shot order as recorded; 2nd row = channel)", color=INK_2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)

    ax.set_title(title or f"Shot gather  FFID {gather.ffid}  -  {ntr} traces", loc="left",
                 fontsize=12, color=INK, pad=20)
    if handles:
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles),
                  frameon=False, fontsize=10, labelcolor=INK_2, borderaxespad=0.3)
    return fig


def plot_qc_metrics(gather: ShotGather, result: QCResult, *, figsize: tuple[float, float] = (14.0, 4.0)) -> Figure:
    """Per-trace RMS (log scale) against the reference and the noisy/weak limits.

    Helps to tune thresholds: points outside the dashed limits are the flagged ones.
    """
    order = result.order
    ntr = gather.ntr
    pos = np.arange(1, ntr + 1)
    rms = result.metrics["rms"][order]
    ref = result.ref_rms[order]
    dead = result.dead[order]
    bad = result.bad[order]
    live = rms > 0
    floor = (rms[live].min() / 3.0) if live.any() else 1e-12

    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.065, 0.16, 0.92, 0.66])
    ok = ~dead & ~bad
    ax.plot(pos[ok], np.where(live, rms, floor)[ok], linestyle="none", marker="o", markersize=2.5,
            color=MUTED, alpha=0.8, label="trace RMS")
    ax.plot(pos, ref, color=INK, linewidth=1.4, label="reference (running median)")

    p = result.params
    if p.get("noisy_ratio", 0) > 0:
        ax.plot(pos, ref * p["noisy_ratio"], color=INK_2, linewidth=1.0, linestyle="--",
                label=f"noisy limit ({p['noisy_ratio']:g}x)")
    if p.get("weak_ratio", 0) > 0:
        ax.plot(pos, ref * p["weak_ratio"], color=INK_2, linewidth=1.0, linestyle=":",
                label=f"weak limit ({p['weak_ratio']:g}x)")
    if bad.any():
        ax.plot(pos[bad], np.where(live, rms, floor)[bad], linestyle="none", marker="v", markersize=6,
                color=BAD_COLOR, label=f"bad ({int(bad.sum())})")
    if dead.any():
        ax.plot(pos[dead], np.where(live, rms, floor)[dead], linestyle="none", marker="D", markersize=5,
                color=DEAD_COLOR, label=f"dead ({int(dead.sum())})")

    ax.set_yscale("log")
    ax.set_xlim(0.5, ntr + 0.5)
    ax.set_xlabel("Trace position (same order as the gather plot)", color=INK_2)
    ax.set_ylabel("RMS amplitude", color=INK_2)
    ax.tick_params(labelsize=9, colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.set_title(f"Trace RMS  -  FFID {gather.ffid}", loc="left", fontsize=12, color=INK, pad=28)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=6, frameon=False, fontsize=9,
              labelcolor=INK_2, borderaxespad=0.3, columnspacing=1.2, handlelength=1.6)
    return fig


def plot_survey_counts(per_shot: list[dict], *, figsize: tuple[float, float] = (14.0, 6.0)) -> Figure:
    """Dead and bad trace count of every scanned shot, against FFID."""
    ffid = np.array([r["ffid"] for r in per_shot])
    fig = Figure(figsize=figsize, facecolor="white")
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"left": 0.065, "right": 0.985, "bottom": 0.11,
                                                        "top": 0.86, "hspace": 0.12})
    for ax, key, color, label in ((axes[0], "dead", DEAD_COLOR, "dead"), (axes[1], "bad", BAD_COLOR, "bad")):
        n = np.array([r[key] for r in per_shot])
        ax.vlines(ffid, 0, n, colors=color, linewidth=1.4)
        ax.set_ylabel(f"{label} traces / shot", color=INK_2)
        ax.set_ylim(0, max(n.max(), 1) * 1.15)
        ax.tick_params(labelsize=9, colors=INK_2)
        ax.grid(True, axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.text(0.995, 0.93, f"total {int(n.sum()):,}", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=color)
    axes[1].set_xlabel("FFID", color=INK_2)
    fig.suptitle(f"Dead and bad traces per shot  -  {len(per_shot)} shots, "
                 f"{sum(r['traces'] for r in per_shot):,} traces", x=0.065, ha="left", fontsize=12, color=INK)
    return fig


def plot_spectrum_acorr(before: np.ndarray, after: np.ndarray, dt_ms: float, labels: tuple[str, str],
                        *, figsize: tuple[float, float] = (14.0, 4.2)) -> Figure:
    """Mean amplitude spectrum (dB) and mean autocorrelation of two versions of a gather."""
    fig = Figure(figsize=figsize, facecolor="white")
    ax1 = fig.add_axes([0.065, 0.16, 0.41, 0.66])
    ax2 = fig.add_axes([0.565, 0.16, 0.41, 0.66])
    colors = (MUTED, INK)
    for data, color, lab in ((before, colors[0], labels[0]), (after, colors[1], labels[1])):
        f, amp = decon.mean_amplitude_spectrum(data, dt_ms)
        ax1.plot(f, 20 * np.log10(np.maximum(amp / max(amp.max(), 1e-30), 1e-6)), color=color, linewidth=1.5, label=lab)
        lag, ac = decon.mean_autocorrelation(data, dt_ms)
        ax2.plot(lag, ac, color=color, linewidth=1.3, label=lab)
    ax1.set_xlim(0, min(125.0, 500.0 / dt_ms))
    ax1.set_ylim(-60, 3)
    ax1.set_xlabel("Frequency (Hz)", color=INK_2)
    ax1.set_ylabel("Mean amplitude spectrum (dB)", color=INK_2)
    ax2.set_xlabel("Lag (ms)", color=INK_2)
    ax2.set_ylabel("Mean autocorrelation (normalised)", color=INK_2)
    ax2.set_ylim(-0.7, 1.05)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=9, colors=INK_2)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=10,
               labelcolor=INK_2, borderaxespad=0.3)
    fig.suptitle("Spectrum and autocorrelation of the live traces", x=0.065, y=0.97, ha="left", fontsize=12, color=INK)
    return fig


def plot_correlation_metrics(gather: ShotGather, result: QCResult, *, figsize: tuple[float, float] = (14.0, 4.0)) -> Figure:
    """Neighbour-correlation score of every trace against its local reference and the flagging limit."""
    order, ntr = result.order, gather.ntr
    pos = np.arange(1, ntr + 1)
    score, ref = result.metrics["corr"][order], result.metrics["corr_ref"][order]
    bad = result.bad[order]
    limit = result.params.get("corr_ratio_min", 0.0)
    low = (result.metrics["corr_ratio"][order] < limit) if limit > 0 else np.zeros(ntr, bool)
    tested = np.isfinite(score) & (ref > 0)

    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.065, 0.16, 0.92, 0.66])
    ok = tested & ~low
    ax.plot(pos[ok], score[ok], linestyle="none", marker="o", markersize=2.5, color=MUTED, alpha=0.8, label="trace score")
    ax.plot(pos[tested], ref[tested], color=INK, linewidth=1.2, label="local reference (median)")
    if limit > 0:
        ax.plot(pos[tested], ref[tested] * limit, color=INK_2, linewidth=1.0, linestyle="--", label=f"limit ({limit:g}x)")
    if (low & tested).any():
        ax.plot(pos[low & tested], score[low & tested], linestyle="none", marker="v", markersize=6, color=BAD_COLOR,
                label=f"low correlation ({int((low & tested).sum())})")
    ax.set_xlim(0.5, ntr + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Trace position (same order as the gather plot)", color=INK_2)
    ax.set_ylabel("Neighbour correlation", color=INK_2)
    ax.tick_params(labelsize=9, colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.set_title(f"Correlation with neighbours on the receiver line  -  FFID {gather.ffid}", loc="left",
                 fontsize=12, color=INK, pad=28)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, frameon=False, fontsize=9, labelcolor=INK_2,
              borderaxespad=0.3, columnspacing=1.2, handlelength=1.6)
    return fig


def _style_axes(ax) -> None:
    """The axes look shared by all figures: light grid, no top / right spine, muted left / bottom spine."""
    ax.tick_params(labelsize=9, colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)


def _pad_limits(ax, fx: float = 0.03, fy: float = 0.07) -> None:
    """Room around everything drawn, so outline labels and corner marks do not touch the legend or the tick labels.
    Done with two invisible points, so the equal-scale axes can still pick whichever limits they need."""
    (x0, x1), (y0, y1) = ax.dataLim.intervalx, ax.dataLim.intervaly
    ax.plot([x0 - fx * (x1 - x0), x1 + fx * (x1 - x0)], [y0 - fy * (y1 - y0), y1 + fy * (y1 - y0)],
            linestyle="none", marker="none", zorder=0)


def _coord_axes(ax, unit: str, xname: str = "X", yname: str = "Y") -> None:
    """Equal-scale map axes with thousands separators."""
    ax.set_aspect("equal", adjustable="datalim")
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if unit != "deg" else f"{v:.3f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    u = f" ({unit})" if unit else ""
    ax.set_xlabel(f"{xname}{u}", color=INK_2)
    ax.set_ylabel(f"{yname}{u}", color=INK_2)


def _nice_ticks(lo: float, hi: float, target: int = 8) -> np.ndarray:
    """Round values (1, 2 or 5 x 10^k apart) covering lo..hi, about `target` of them."""
    if hi <= lo:
        return np.array([lo])
    raw = (hi - lo) / target
    mag = 10.0 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    return np.arange(np.ceil(lo / step) * step, hi + step * 0.001, step)


def _ends(grid: Grid, il, xl):
    """((x_start, x_end), (y_start, y_end)) of the grid line through the two (il, xl) points."""
    x, y = grid.xy(il, xl)
    return (x[0], x[1]), (y[0], y[1])


def _draw_grid(ax, grid: Grid) -> Line2D:
    """Inline / crossline lines, the survey outline and its corner labels."""
    il0, il1 = grid.il_range
    xl0, xl1 = grid.xl_range
    kw = dict(color=INK_2, linewidth=0.7, alpha=0.35, zorder=1)

    def line(a, b, text):
        """Draw the grid line a -> b and write `text` just outside its start, along the line."""
        (xa, xb), (ya, yb) = a, b
        ax.plot([xa, xb], [ya, yb], **kw)
        d = np.hypot(xb - xa, yb - ya) or 1.0
        ax.annotate(text, (xa, ya), xytext=(-14 * (xb - xa) / d, -14 * (yb - ya) / d), textcoords="offset points",
                    ha="center", va="center", fontsize=8, color=MUTED, annotation_clip=False)

    for v in _nice_ticks(il0, il1):
        line(*_ends(grid, [v, v], [xl0, xl1]), f"IL {v:g}")
    for v in _nice_ticks(xl0, xl1):
        line(*_ends(grid, [il0, il1], [v, v]), f"XL {v:g}")
    cx, cy = [c[2] for c in grid.corners], [c[3] for c in grid.corners]
    ax.plot(cx + cx[:1], cy + cy[:1], color=INK_2, linewidth=1.4, linestyle="--", zorder=6)
    ax.plot(cx, cy, linestyle="none", marker="s", markersize=6, color=INK_2, zorder=7)
    mx, my = np.mean(cx), np.mean(cy)
    for il, xl, x, y in grid.corners:
        ax.annotate(f"IL {il:g} / XL {xl:g}", (x, y), xytext=(6 if x >= mx else -6, 6 if y >= my else -6),
                    textcoords="offset points", ha="left" if x >= mx else "right", va="bottom" if y >= my else "top",
                    fontsize=9, color=INK, zorder=8)
    return Line2D([0], [0], color=INK_2, linewidth=1.4, linestyle="--", marker="s", markersize=5,
                  label=f"survey grid ({grid.source})")


def _ilxl_axes(ax, grid: Grid) -> None:
    """Replace the X / Y tick labels by IL / XL: ticks at round IL / XL numbers, each at its position on the axis.
    The x axis takes the grid direction that lies closer to it (for a grid rotated against the plot, the values
    are read along the middle of the other axis and the axis label says so)."""
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    m = grid.a[:, 1:]                                # columns: (dx, dy) per IL step, per XL step
    ang = lambda col: float(np.degrees(np.arctan2(abs(m[1, col]), abs(m[0, col]))))   # from the x axis
    x_is_xl = ang(1) <= ang(0)
    tilt = max(min(ang(0), ang(1)), 90.0 - max(ang(0), ang(1)))
    pick_x = (lambda il, xl: xl) if x_is_xl else (lambda il, xl: il)
    pick_y = (lambda il, xl: il) if x_is_xl else (lambda il, xl: xl)
    what = ({"IL": "receiver line", "XL": "receiver station"} if grid.source == "receiver headers"
            else {"IL": "inline", "XL": "crossline"})
    for axis, lo, hi, other, pick, name in (
        ("x", x0, x1, (y0 + y1) / 2, pick_x, "XL" if x_is_xl else "IL"),
        ("y", y0, y1, (x0 + x1) / 2, pick_y, "IL" if x_is_xl else "XL"),
    ):
        pts = ((lo, other), (hi, other)) if axis == "x" else ((other, lo), (other, hi))
        v0, v1 = (float(pick(*grid.ilxl(*p))) for p in pts)
        if v0 == v1:
            continue
        ticks = _nice_ticks(min(v0, v1), max(v0, v1), 9)
        pos = lo + (ticks - v0) * (hi - lo) / (v1 - v0)
        getattr(ax, f"set_{axis}ticks")(pos)
        getattr(ax, f"set_{axis}ticklabels")([f"{t:g}" for t in ticks])
        note = f", grid turned {tilt:.0f}° against the plot axes: read along the middle" if tilt > 2 else ""
        getattr(ax, f"set_{axis}label")(f"{name} ({what[name]}){note}", color=INK_2)


def plot_geometry(geom: Geometry, spread: Spread | None = None, *, grid: Grid | None = None, show_grid: bool = True,
                  ilxl_ticks: bool = False, show_sources: bool = True, show_receivers: bool = True,
                  fig_height: float = 8.0, title: str | None = None,
                  figsize: tuple[float, float] | None = None) -> Figure:
    """Plan view of the acquisition: receivers = grey dots, sources = blue triangles; the selected shot's
    receivers are green and its source a black-edged star.  With a `grid`, the IL / XL lines and the survey
    outline (corner points) are drawn too."""
    fig = Figure(figsize=figsize or (14.0, float(fig_height)), facecolor="white")
    ax = fig.add_axes([0.065, 0.10, 0.905, 0.80])
    handles = []
    if show_receivers and len(geom.rec_x):
        ax.plot(geom.rec_x, geom.rec_y, linestyle="none", marker="o", markersize=2.5, color=MUTED, alpha=0.8, zorder=2)
        part = "" if geom.shots_read >= len(geom.ffids) else f", from {geom.shots_read} shots"
        handles.append(Line2D([0], [0], color=MUTED, marker="o", markersize=5, linestyle="none",
                              label=f"receivers ({len(geom.rec_x):,}{part})"))
    if show_sources:
        ax.plot(geom.src_x, geom.src_y, linestyle="none", marker="^", markersize=4, color=DEAD_COLOR, alpha=0.85, zorder=3)
        handles.append(Line2D([0], [0], color=DEAD_COLOR, marker="^", markersize=6, linestyle="none",
                              label=f"sources ({len(geom.src_x):,})"))
    if spread is not None:
        ax.plot(spread.rec_x, spread.rec_y, linestyle="none", marker="o", markersize=3.5, color=FIXED_COLOR, zorder=4)
        ax.plot([spread.src_x], [spread.src_y], linestyle="none", marker="*", markersize=15, markerfacecolor=FIXED_COLOR,
                markeredgecolor=INK, markeredgewidth=1.0, zorder=5)
        handles.append(Line2D([0], [0], color=FIXED_COLOR, marker="o", markersize=5, linestyle="none",
                              label=f"FFID {spread.ffid} receivers ({len(spread.rec_x)})"))
        handles.append(Line2D([0], [0], color=FIXED_COLOR, marker="*", markersize=11, markeredgecolor=INK,
                              linestyle="none", label=f"FFID {spread.ffid} source"))
    if grid is not None and show_grid:
        handles.append(_draw_grid(ax, grid))
    _pad_limits(ax)
    _style_axes(ax)
    _coord_axes(ax, geom.unit, "X (easting)", "Y (northing)")
    if grid is not None and ilxl_ticks:
        _ilxl_axes(ax, grid)
    ax.set_title(title or f"Acquisition geometry  -  {len(geom.ffids):,} shots", loc="left", fontsize=12, color=INK, pad=20)
    if handles:
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles),
                  frameon=False, fontsize=10, labelcolor=INK_2, borderaxespad=0.3)
    return fig


def plot_spread(spread: Spread, unit: str = "", *, figsize: tuple[float, float] = (14.0, 4.6)) -> Figure:
    """One shot: the receiver positions relative to its source."""
    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.065, 0.14, 0.905, 0.68])
    ax.plot(spread.rec_x - spread.src_x, spread.rec_y - spread.src_y, linestyle="none", marker="o", markersize=3.5,
            color=FIXED_COLOR, label=f"receivers ({len(spread.rec_x)})")
    ax.plot([0], [0], linestyle="none", marker="*", markersize=15, markerfacecolor=FIXED_COLOR, markeredgecolor=INK,
            markeredgewidth=1.0, label="source")
    _style_axes(ax)
    _coord_axes(ax, unit, "X relative to source", "Y relative to source")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=10,
              labelcolor=INK_2, borderaxespad=0.3)
    fig.suptitle(f"Receiver spread of FFID {spread.ffid}  -  {len(spread.rec_x)} traces", x=0.065, y=0.97,
                 ha="left", fontsize=12, color=INK)
    return fig


def plot_fold(fold: FoldMap, geom: Geometry | None = None, *, grid: Grid | None = None, show_grid: bool = True,
              ilxl_ticks: bool = False, show_sources: bool = False, show_receivers: bool = False,
              cmap: str = "viridis", vmax: float = 0, fig_height: float = 8.0, title: str | None = None,
              figsize: tuple[float, float] | None = None) -> Figure:
    """Fold (traces per bin) as a plan-view map on the same canvas as plot_geometry: the same figure size, the same
    axes box, so the two line up.  The colour bar sits INSIDE the map (an inset in the empty right-hand margin),
    so it takes nothing from the figure's width or height."""
    fig = Figure(figsize=figsize or (14.0, float(fig_height)), facecolor="white")
    ax = fig.add_axes([0.065, 0.10, 0.905, 0.80])
    live = fold.filled
    top = float(vmax) if vmax and vmax > 0 else (float(np.ceil(np.percentile(live, 99.5))) if len(live) else 1.0)
    mesh = ax.pcolormesh(fold.node_x, fold.node_y, np.ma.masked_equal(fold.counts, 0), cmap=cmap,
                         vmin=1, vmax=max(top, 2.0), shading="flat", rasterized=True, zorder=1)
    handles = []
    if geom is not None and show_receivers and len(geom.rec_x):
        ax.plot(geom.rec_x, geom.rec_y, linestyle="none", marker="o", markersize=2, color=INK_2, alpha=0.55, zorder=3)
        handles.append(Line2D([0], [0], color=INK_2, marker="o", markersize=5, linestyle="none", label="receivers"))
    if geom is not None and show_sources:
        ax.plot(geom.src_x, geom.src_y, linestyle="none", marker="^", markersize=3.5, color=BAD_COLOR, alpha=0.8, zorder=4)
        handles.append(Line2D([0], [0], color=BAD_COLOR, marker="^", markersize=6, linestyle="none", label="sources"))
    if grid is not None and show_grid:
        handles.append(_draw_grid(ax, grid))
    _pad_limits(ax)
    _style_axes(ax)
    ax.grid(False)
    _coord_axes(ax, fold.unit, "X (easting)", "Y (northing)")
    if grid is not None and ilxl_ticks:
        _ilxl_axes(ax, grid)

    # colour bar inside the map: a white card and a narrow inset axes in the right-hand margin
    ax.add_patch(Rectangle((0.915, 0.04), 0.078, 0.64, transform=ax.transAxes, facecolor="white", edgecolor=GRID,
                           linewidth=0.8, alpha=0.92, zorder=8))
    cax = ax.inset_axes([0.926, 0.085, 0.016, 0.50], zorder=9)
    cb = fig.colorbar(mesh, cax=cax)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=3)
    cb.outline.set_edgecolor(MUTED)
    ax.text(0.954, 0.635, "Fold", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=INK, zorder=9)

    n = len(live)
    ax.set_title(title or (f"Fold map  -  {fold.traces:,} traces in {n:,} bins, maximum fold {int(live.max()):,}, "
                           f"mean {live.mean():.1f}" if n else "Fold map  -  no data"),
                 loc="left", fontsize=12, color=INK, pad=20)
    if handles:
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles), frameon=False,
                  fontsize=10, labelcolor=INK_2, borderaxespad=0.3)
    return fig



def plot_nmo(gather_traces: np.ndarray, nmo_traces: np.ndarray, full_t: np.ndarray, vel_of_t0: np.ndarray | None,
             match_info: dict | None, velocity_unit: str, *, clip_pct: float = 98.0, title: str = "",
             fig_height: float = 6.0, stack_trace: np.ndarray | None = None, nmo_title: str = "NMO stretch section",
             no_velocity_text: str = "No velocity model") -> Figure:
    """The brute-stack notebook's NMO preview (cdp_nmo_stack_marimo.py, section 6): the CDP gather (offset-sorted), the
    NMO stretch section, the stack (when stack_trace is given) and the velocity used - output RMS vs the model's own
    input trace (on its own axis: time, or depth for a depth model)."""
    y_max_ms = full_t[-1] * 1000.0
    t_ms = full_t * 1000.0
    fig = Figure(figsize=(13.5, float(fig_height)), facecolor="white")
    if stack_trace is not None:
        axes = fig.subplots(1, 4, gridspec_kw={"width_ratios": [1, 1, 0.4, 0.6]})
        vclip_stack = np.percentile(np.abs(stack_trace), clip_pct) or 1e-9
        axes[2].plot(stack_trace, t_ms, color="black", linewidth=0.7)
        axes[2].set_xlim(-vclip_stack, vclip_stack)
        axes[2].set_ylim(y_max_ms, 0)
        axes[2].set_title("Stack")
        axes[2].set_xlabel("Amp")
        axes = [axes[0], axes[1], axes[3]]
    else:
        axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.6]})
    vclip_raw = np.percentile(np.abs(gather_traces), clip_pct) or 1.0
    vclip_nmo = np.percentile(np.abs(nmo_traces), clip_pct) or 1.0
    axes[0].imshow(gather_traces.T, aspect="auto", cmap="gray", vmin=-vclip_raw, vmax=vclip_raw,
                   extent=[0, gather_traces.shape[0], y_max_ms, 0])
    axes[0].set_ylim(y_max_ms, 0)
    axes[0].set_title("CDP gather (offset-sorted)")
    axes[0].set_xlabel("Trace #")
    axes[0].set_ylabel("Time (ms)")
    axes[1].imshow(nmo_traces.T, aspect="auto", cmap="gray", vmin=-vclip_nmo, vmax=vclip_nmo,
                   extent=[0, nmo_traces.shape[0], y_max_ms, 0])
    axes[1].set_ylim(y_max_ms, 0)
    axes[1].set_title(nmo_title)
    axes[1].set_xlabel("Trace #")
    ax = axes[2]
    if vel_of_t0 is not None:
        line_out, = ax.plot(vel_of_t0, t_ms, color="tab:blue", linewidth=1.2, label="Output (RMS)")
        ax.set_title("Velocity used")
        ax.set_xlabel(velocity_unit)
        handles = [line_out]
        if match_info is not None:
            kind = match_info["native_axis_kind"]
            axis = match_info["native_axis"] * (1000.0 if kind == "Time" else 1.0)
            length_unit = "ft" if velocity_unit == "ft/s" else "m"
            twin = ax.twinx()
            line_in, = twin.plot(match_info["native_velocity"], axis, color="tab:red", linewidth=1.2,
                                 label=f"Input ({kind.lower()})")
            twin.set_ylabel(f"{kind} ({'ms' if kind == 'Time' else length_unit})", color="tab:red", fontsize=7)
            twin.tick_params(axis="y", labelcolor="tab:red", labelsize=6)
            # same real TWT range on both axes (converted into depth for a depth model)
            if kind == "Time":
                twin.set_ylim(y_max_ms, 0)
                model_max_ms = match_info["model_t"][-1] * 1000.0
            else:
                model_t_ms = match_info["model_t"] * 1000.0
                d0 = np.interp(0.0, model_t_ms, axis)
                d1 = np.interp(y_max_ms, model_t_ms, axis, right=axis[-1])
                twin.set_ylim(d1, d0)
                model_max_ms = model_t_ms[-1]
            if model_max_ms < y_max_ms * 0.999:          # the model ends before the data: held flat below this line
                ax.axhline(model_max_ms, color="gray", linestyle="--", linewidth=0.7)
            handles.append(line_in)
        ax.legend(handles=handles, loc="lower right", fontsize=6)
    else:
        ax.text(0.5, 0.5, no_velocity_text, ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Velocity used")
    ax.set_ylim(y_max_ms, 0)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_model_overlay(model_xl: np.ndarray, model_il: np.ndarray, data_xl: np.ndarray, data_il: np.ndarray, *,
                       cdp_xl: float | None = None, cdp_il: float | None = None, match_xl: float | None = None,
                       match_il: float | None = None, cdp: int | None = None, fig_height: float = 6.0) -> Figure:
    """The brute-stack notebook's "Data vs. model grid coverage" overlay, in the data's IL / XL: the velocity model's grid
    (converted to data IL / XL) underneath, the data's CDP bins on top, the CDP being corrected and the model trace its
    velocity came from."""
    fig = Figure(figsize=(9.0, float(fig_height)), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(model_xl, model_il, s=6, marker="s", color="lightgray", linewidths=0, rasterized=True,
               label="Model grid (converted to data IL/XL)")
    ax.scatter(data_xl, data_il, s=2, color="steelblue", linewidths=0, rasterized=True, label="Data (unique IL/XL bins)")
    if cdp_xl is not None:
        ax.scatter([cdp_xl], [cdp_il], s=160, marker="*", color="red", edgecolors="black", linewidths=0.5, zorder=5,
                   label=f"CDP {cdp}" if cdp is not None else "CDP")
    if match_xl is not None:
        ax.scatter([match_xl], [match_il], s=70, marker="x", color="darkorange", linewidths=2, zorder=6,
                   label="Matched model trace")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Data XL")
    ax.set_ylabel("Data IL")
    ax.set_title("Data vs. model grid coverage")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, frameon=False, markerscale=1.5)
    fig.tight_layout()
    return fig



def _two_sections(fig_height):
    fig = Figure(figsize=(12.0, float(fig_height)), facecolor="white")
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [60, 40]}, sharey=True)
    return fig, axes


def _extent(vary_vals, y_max_ms):
    a, b = float(vary_vals[0]), float(vary_vals[-1])
    if a == b:                                          # one CDP on the line: give it a visible width
        a, b = a - 0.5, b + 0.5
    return [a, b, y_max_ms, 0]


def _empty(ax, fixed_label, fixed_val):
    ax.text(0.5, 0.5, f"No CDPs at {fixed_label}={int(fixed_val)}", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{fixed_label}={int(fixed_val)} (empty)")


def plot_stack_sections(il_result, xl_result, il_value: int, xl_value: int, full_t: np.ndarray, *,
                        clip_pct: float = 98.0, fig_height: float = 6.5) -> Figure:
    """The notebook's stacked sections: IL section (fixed IL, varying XL) left, XL section right, same time axis.
    *_result = (vary values, section [n, ns]) or None."""
    y_max_ms = full_t[-1] * 1000.0
    fig, axes = _two_sections(fig_height)
    for ax, result, fixed_label, fixed_val, vary_label in ((axes[0], il_result, "IL", il_value, "XL"),
                                                            (axes[1], xl_result, "XL", xl_value, "IL")):
        if result is None:
            _empty(ax, fixed_label, fixed_val)
            continue
        vary_vals, section = result
        vclip = np.percentile(np.abs(section), clip_pct) or 1.0
        ax.imshow(section.T, aspect="auto", cmap="gray", vmin=-vclip, vmax=vclip,
                  extent=_extent(vary_vals, y_max_ms))
        ax.set_title(f"{fixed_label}={int(fixed_val)}, {len(vary_vals)} CDPs")
        ax.set_xlabel(vary_label)
    axes[0].set_ylim(y_max_ms, 0)
    axes[0].set_ylabel("Time (ms)")
    fig.tight_layout()
    return fig


def _inset_colorbar(fig, ax, im):
    cax = ax.inset_axes([0.55, 0.06, 0.4, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)


def plot_velocity_sections(il_result, xl_result, il_value: int, xl_value: int, full_t: np.ndarray, kind: str = "RMS", *,
                           fig_height: float = 6.5) -> Figure:
    """The notebook's velocity model sections along the same two lines (gray = beyond the model's own range).
    *_result = (vary values, velocity section [n, ns] with NaN beyond the model) or None."""
    import matplotlib
    y_max_ms = full_t[-1] * 1000.0
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgray")
    fig, axes = _two_sections(fig_height)
    im = None
    for ax, result, fixed_label, fixed_val, vary_label in ((axes[0], il_result, "IL", il_value, "XL"),
                                                            (axes[1], xl_result, "XL", xl_value, "IL")):
        if result is None:
            _empty(ax, fixed_label, fixed_val)
            continue
        vary_vals, section = result
        im = ax.imshow(section.T, aspect="auto", cmap=cmap, extent=_extent(vary_vals, y_max_ms))
        ax.set_title(f"{fixed_label}={int(fixed_val)} {kind.lower()} velocity")
        ax.set_xlabel(vary_label)
    axes[0].set_ylim(y_max_ms, 0)
    axes[0].set_ylabel("Time (ms)")
    if im is not None:
        _inset_colorbar(fig, axes[0], im)
    fig.tight_layout()
    return fig


def plot_overlay_sections(il_result, xl_result, il_value: int, xl_value: int, full_t: np.ndarray, kind: str = "RMS", *,
                          clip_pct: float = 98.0, fig_height: float = 6.5) -> Figure:
    """The notebook's overlay: the velocity in colour (half transparent) over the grayscale stack; where the model has no
    data the stack shows through. *_result = (vary values, stack section, velocity section) or None."""
    import matplotlib
    y_max_ms = full_t[-1] * 1000.0
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(alpha=0.0)
    fig, axes = _two_sections(fig_height)
    im = None
    for ax, result, fixed_label, fixed_val, vary_label in ((axes[0], il_result, "IL", il_value, "XL"),
                                                            (axes[1], xl_result, "XL", xl_value, "IL")):
        if result is None:
            _empty(ax, fixed_label, fixed_val)
            continue
        vary_vals, stack_section, vel_section = result
        extent = _extent(vary_vals, y_max_ms)
        vclip = np.percentile(np.abs(stack_section), clip_pct) or 1.0
        ax.imshow(stack_section.T, aspect="auto", cmap="gray", vmin=-vclip, vmax=vclip, extent=extent)
        im = ax.imshow(vel_section.T, aspect="auto", cmap=cmap, alpha=0.5, extent=extent)
        ax.set_title(f"{fixed_label}={int(fixed_val)} + {kind.lower()} vel, {len(vary_vals)} CDPs")
        ax.set_xlabel(vary_label)
    axes[0].set_ylim(y_max_ms, 0)
    axes[0].set_ylabel("Time (ms)")
    if im is not None:
        _inset_colorbar(fig, axes[0], im)
    fig.tight_layout()
    return fig
