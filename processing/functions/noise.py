"""Coherent-noise removal on a shot gather (no UI): F-K velocity-fan filter and least-squares Radon (linear tau-p for
ground roll / linear noise, parabolic for multiples). Gathers are [ntr, ns] arrays, as everywhere in this package.

Both work by *subtraction* where they can (Radon models the noise and takes it away), so whatever is not noise is
left exactly as it was.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .segy_io import ShotGather


# ---------------------------------------------------------------------------------------------------------------------
# geometry of a shot gather: receiver lines and the position of every trace along its line
# ---------------------------------------------------------------------------------------------------------------------
def line_groups(g: ShotGather) -> list[np.ndarray]:
    """Trace indices of every receiver line of the shot (one group when the header has no line number)."""
    lines = np.asarray(g.headers.get("rec_line", np.zeros(g.ntr)))
    return [np.flatnonzero(lines == v) for v in np.unique(lines)]


def along_line(g: ShotGather, idx: np.ndarray) -> np.ndarray:
    """Signed position of the traces `idx` along their receiver line, in the offset unit of the header, increasing
    along the line. From the receiver X / Y (projected on the line's direction and scaled so that the source distance
    matches the header offset - this also takes care of any coordinate scalar); without usable coordinates, the
    channel order with the median offset step."""
    h = g.headers
    off = np.abs(np.asarray(h["offset"], np.float64)[idx])
    rx, ry = (np.asarray(h[k], np.float64)[idx] for k in ("rec_x", "rec_y"))
    sx, sy = (np.asarray(h[k], np.float64)[idx] for k in ("src_x", "src_y"))
    xy = np.column_stack([rx, ry])
    if len(idx) >= 2 and np.ptp(xy, axis=0).max() > 0:
        c = xy - xy.mean(axis=0)
        direction = np.linalg.svd(c, full_matrices=False)[2][0]
        s = c @ direction
        dist = np.hypot(rx - sx, ry - sy)
        ok = (dist > 0) & (off > 0)
        scale = float(np.median(off[ok] / dist[ok])) if ok.any() else 1.0
        return s * scale
    order = np.argsort(np.asarray(h["chan"])[idx], kind="stable")
    step = float(np.median(np.abs(np.diff(np.sort(off))))) if len(off) > 1 else 1.0
    pos = np.empty(len(idx))
    pos[order] = np.arange(len(idx)) * max(step, 1.0)
    return pos


def _regular(x: np.ndarray) -> tuple[np.ndarray, float]:
    """(grid column of every trace, spacing): the traces on a regular grid along the line (gaps stay empty columns).
    Falls back to one column per trace, in order, when traces share a column or the gaps are huge."""
    order = np.argsort(x, kind="stable")
    d = np.diff(x[order])
    dx = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    col = np.rint((x - x.min()) / dx).astype(np.int64)
    if len(np.unique(col)) < len(col) or col.max() + 1 > 4 * len(col):
        col = np.empty(len(x), np.int64)
        col[order] = np.arange(len(x))
    return col, dx


# ---------------------------------------------------------------------------------------------------------------------
# F-K filter
# ---------------------------------------------------------------------------------------------------------------------
def fk_weight(f: np.ndarray, k: np.ndarray, v_reject: float, v_pass: float, f_max: float = 0.0) -> np.ndarray:
    """Pass weight [nf, nk]: 0 for apparent velocities |f / k| below v_reject, 1 above v_pass, a cosine taper in
    between; frequencies above f_max (> 0) are passed untouched (5 Hz taper)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.abs(f[:, None]) / np.abs(k[None, :])
    v = np.where(np.abs(k[None, :]) == 0, np.inf, v)
    lo, hi = float(v_reject), float(max(v_pass, v_reject + 1e-6))
    r = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * r)
    if f_max > 0:
        keep = np.clip((f - f_max) / 5.0, 0.0, 1.0)[:, None]
        w = w + (1.0 - w) * keep
    return w


K_COL, F_COL, Z_COL = "k (cycles per 1000)", "f (Hz)", "zone"   # columns of the drawn / typed F-K reject zone


def polygon_rows(rows) -> list[np.ndarray]:
    """The reject zone's polygons [[k per 1000 units, f Hz], ...] from the table rows - one per 'zone' number (rows
    without one belong to zone 1); empty rows skipped."""
    parts: dict = {}
    for r in rows or []:
        low = {str(a).strip().lower(): b for a, b in (r.items() if isinstance(r, dict) else [])}
        k, f, z = low.get(K_COL.lower()), low.get(F_COL.lower()), low.get(Z_COL)
        if k in (None, "") or f in (None, ""):
            continue
        parts.setdefault(int(float(z)) if z not in (None, "") else 1, []).append((float(k), float(f)))
    return [np.array(v, float) for _, v in sorted(parts.items())]


def polygon_weight(polys: list[np.ndarray], mirror: bool = True, smooth_bins: int = 3):
    """A weight function (f, k) -> pass weight for a reject zone of one or more polygons (any shape - box, lasso)
    drawn in (k per 1000 units, f Hz): 0 inside any of them (and inside their mirror images at -k), 1 outside,
    edges smoothed over a few bins (no ringing)."""
    from matplotlib.path import Path as _Path
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        raise ValueError("the reject zone needs a polygon of at least 3 corners - drag a box or Shift + drag a lasso "
                         "on the F-K plot, or type the corners in the table")
    paths = [_Path(p) for p in polys]

    def weight(f: np.ndarray, k: np.ndarray) -> np.ndarray:
        kk, ff = np.meshgrid(k * 1000.0, f)
        pts = np.column_stack([kk.ravel(), ff.ravel()])
        inside = np.zeros(len(pts), bool)
        for path in paths:
            inside |= path.contains_points(pts)
            if mirror:
                inside |= path.contains_points(np.column_stack([-kk.ravel(), ff.ravel()]))
        w = 1.0 - inside.reshape(kk.shape).astype(float)
        if smooth_bins > 1:                              # soft edges: a short running mean in f and in k
            ker = np.ones(smooth_bins) / smooth_bins
            w = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, w)
            w = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 1, w)
        return np.clip(w, 0.0, 1.0)
    return weight


def fan_weight(v_reject: float, v_pass: float, f_max: float = 0.0):
    return lambda f, k: fk_weight(f, k, v_reject, v_pass, f_max)


def _fk_line(block: np.ndarray, dt: float, dx: float, weight, spectra=False):
    """F-K filter of one regular block [ns, nx] (time down, distance across) with the pass weight weight(f, k);
    returns the filtered block and, with spectra=True, (f, k, |before|, |after|) for the plot."""
    ns, nx = block.shape
    nt = int(2 ** np.ceil(np.log2(ns * 1.25)))
    nk = int(2 ** np.ceil(np.log2(max(2 * nx, 8))))
    spec = np.fft.fft(np.fft.rfft(block, n=nt, axis=0), n=nk, axis=1)
    f = np.fft.rfftfreq(nt, dt)
    k = np.fft.fftfreq(nk, dx)
    w = weight(f, k)
    out = np.fft.irfft(np.fft.ifft(spec * w, axis=1)[:, :nx], n=nt, axis=0)[:ns]
    if not spectra:
        return out, None
    return out, (f, np.fft.fftshift(k), np.fft.fftshift(np.abs(spec), axes=1), np.fft.fftshift(np.abs(spec * w), axes=1))


def fk_filter(g: ShotGather, weight, spectra: bool = False):
    """F-K filter of a shot gather with the pass weight weight(f, k) (fan_weight / polygon_weight), one receiver line
    at a time, traces at their position along the line.
    Returns (filtered data [ntr, ns], the spectra of the line with the most traces or None, note)."""
    dt = g.dt_ms / 1000.0
    out = np.array(g.data, dtype=np.float64, copy=True)
    shown, done = None, 0
    groups = sorted(line_groups(g), key=len, reverse=True)
    for n, idx in enumerate(groups):
        if len(idx) < 8:                                   # too few traces for a wavenumber spectrum: left as is
            continue
        col, dx = _regular(along_line(g, idx))
        block = np.zeros((g.ns, col.max() + 1))
        block[:, col] = g.data[idx].T
        res, sp = _fk_line(block, dt, dx, weight, spectra=spectra and n == 0)
        out[idx] = res[:, col].T
        if sp is not None:
            shown = (*sp, dx, len(idx))
        done += len(idx)
    note = f"{len(groups)} receiver line(s), {done} traces"
    return out.astype(g.data.dtype, copy=False), shown, note


# ---------------------------------------------------------------------------------------------------------------------
# least-squares Radon (Hampson: one damped least-squares problem per frequency)
# ---------------------------------------------------------------------------------------------------------------------
def _radon_nt(ns: int, dt: float, x: np.ndarray, p: np.ndarray, kind: str, x_ref: float) -> int:
    """FFT length that holds the record plus the largest time shift p * phi(x) - shorter would wrap events around."""
    phi = x if kind == "linear" else (x / x_ref) ** 2
    shift = float(np.abs(phi).max() * np.abs(p).max()) / dt if len(x) and len(p) else 0.0
    return int(2 ** np.ceil(np.log2(ns * 1.1 + shift + 1)))


def radon_forward(data: np.ndarray, dt: float, x: np.ndarray, p: np.ndarray, kind: str, f_max: float,
                  damping_pct: float, x_ref: float = 1.0) -> np.ndarray:
    """Radon model m [np, ns] of data [ntr, ns] (traces at distances x): data(t, x) = sum_p m(t - p * phi(x), p),
    phi(x) = x (linear: p = slowness, s per unit) or (x / x_ref)^2 (parabolic: p = moveout in s at x_ref).
    Least squares with damping (pre-whitening, % of the mean diagonal) - the frequencies above f_max are left 0."""
    ntr, ns = data.shape
    nt = _radon_nt(ns, dt, x, p, kind, x_ref)
    D = np.fft.rfft(data, n=nt, axis=1)                    # [ntr, nf]
    f = np.fft.rfftfreq(nt, dt)
    phi = x if kind == "linear" else (x / x_ref) ** 2
    M = np.zeros((len(p), len(f)), complex)
    use = np.flatnonzero((f > 0) & (f <= (f_max if f_max > 0 else f[-1])))
    for a in range(0, len(use), 32):                        # frequencies in chunks: memory stays small
        fi = use[a:a + 32]
        L = np.exp(-2j * np.pi * f[fi, None, None] * phi[None, :, None] * p[None, None, :])     # [nf, ntr, np]
        LH = np.conj(np.transpose(L, (0, 2, 1)))
        A = LH @ L
        mu = damping_pct / 100.0 * np.real(np.trace(A, axis1=1, axis2=2)) / len(p)
        A = A + mu[:, None, None] * np.eye(len(p))[None]
        rhs = LH @ D[:, fi].T[:, :, None]
        M[:, fi] = np.linalg.solve(A, rhs)[:, :, 0].T
    return np.fft.irfft(M, n=nt, axis=1)[:, :ns]


def radon_inverse(model: np.ndarray, dt: float, x: np.ndarray, p: np.ndarray, kind: str, x_ref: float = 1.0) -> np.ndarray:
    """Traces [ntr, ns] at distances x made from the Radon model [np, ns] (or a part of it, the rest zero)."""
    npp, ns = model.shape
    nt = _radon_nt(ns, dt, x, p, kind, x_ref)
    M = np.fft.rfft(model, n=nt, axis=1)
    f = np.fft.rfftfreq(nt, dt)
    phi = x if kind == "linear" else (x / x_ref) ** 2
    D = np.empty((len(x), len(f)), complex)
    for a in range(0, len(f), 64):
        fs = slice(a, a + 64)
        L = np.exp(-2j * np.pi * f[fs, None, None] * phi[None, :, None] * p[None, None, :])
        D[:, fs] = (L @ M[:, fs].T[:, :, None])[:, :, 0].T
    return np.fft.irfft(D, n=nt, axis=1)[:, :ns]


def nmo_times(t0: np.ndarray, x: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Recording time [ntr, ns] of every zero-offset time t0 at |offset| x for the RMS velocity vel(t0)."""
    return np.sqrt(t0[None, :] ** 2 + (np.abs(x)[:, None] / vel[None, :]) ** 2)


def nmo_apply(data: np.ndarray, t: np.ndarray, x: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """NMO-correct (no stretch mute: the result is only used to model, then taken back)."""
    tx = nmo_times(t, x, vel)
    return np.stack([np.interp(tx[i], t, data[i], left=0.0, right=0.0) for i in range(len(x))])


def nmo_remove(data: np.ndarray, t: np.ndarray, x: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Undo nmo_apply: back to recording time."""
    tx = nmo_times(t, x, vel)
    return np.stack([np.interp(t, tx[i], data[i], left=0.0, right=0.0) for i in range(len(x))])


def radon_linear(g: ShotGather, v_cut: float, v_min: float, n_p: int, f_max: float, damping_pct: float):
    """Ground roll / linear noise by linear (tau-p) Radon in |offset|: everything slower than v_cut (slowness above
    1 / v_cut, down to v_min) is modelled and subtracted. Returns (filtered, noise, (model, p, cut), note)."""
    x = np.abs(np.asarray(g.headers["offset"], np.float64))
    dt = g.dt_ms / 1000.0
    p = np.linspace(0.0, 1.0 / max(v_min, 1e-6), int(n_p))
    data = np.asarray(g.data, np.float64)
    model = radon_forward(data, dt, x, p, "linear", f_max, damping_pct)
    noise_part = np.where((p > 1.0 / v_cut)[:, None], model, 0.0)
    noise = radon_inverse(noise_part, dt, x, p, "linear")
    note = f"linear Radon: removed events slower than {v_cut:g} (modelled down to {v_min:g}), {len(p)} slownesses"
    return (data - noise).astype(g.data.dtype), noise.astype(g.data.dtype), (model, p * 1000.0, 1000.0 / v_cut), note


def radon_parabolic(g: ShotGather, vel: np.ndarray, q_min_ms: float, q_max_ms: float, q_cut_ms: float, n_q: int,
                    f_max: float, damping_pct: float):
    """Multiples by parabolic Radon: NMO with the (primary) velocity function flattens the primaries; the multiples,
    slower, keep a residual moveout. Everything with a moveout at the far offset above q_cut is modelled, taken back
    to recording time and subtracted. Returns (filtered, multiples, (model, q ms, cut ms), note)."""
    x = np.abs(np.asarray(g.headers["offset"], np.float64))
    dt = g.dt_ms / 1000.0
    t = np.arange(g.ns) * dt
    x_ref = float(x.max()) if x.max() > 0 else 1.0
    q = np.linspace(q_min_ms, q_max_ms, int(n_q)) / 1000.0
    data = np.asarray(g.data, np.float64)
    flat = nmo_apply(data, t, x, vel)
    model = radon_forward(flat, dt, x, q, "parabolic", f_max, damping_pct, x_ref)
    mult_part = np.where((q > q_cut_ms / 1000.0)[:, None], model, 0.0)
    mult = nmo_remove(radon_inverse(mult_part, dt, x, q, "parabolic", x_ref), t, x, vel)
    note = (f"parabolic Radon: removed moveout above {q_cut_ms:g} ms at {x_ref:g} offset after NMO "
            f"(q {q_min_ms:g} to {q_max_ms:g} ms, {len(q)} values)")
    return (data - mult).astype(g.data.dtype), mult.astype(g.data.dtype), (model, q * 1000.0, q_cut_ms), note


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
INK, INK_2, MUTED = "#1f2430", "#4a5263", "#9aa3b2"


def _db(a: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(a / max(float(a.max()), 1e-30), 1e-4))


def plot_fk(shown, f_lim: float, which: str = "before", fan=None, poly=None, mirror=True,
            figsize=(14.0, 6.0)) -> Figure:
    """F-K amplitude (dB) of the longest receiver line - before / after the filter or what it removed - all on the
    input's scale (so they flip-flop), with the reject zone: fan = (v_reject, v_pass) lines, or the polygon.
    Axes: k in cycles per 1000 length units, f in Hz (the units a drawn polygon is read in)."""
    f, k, before, after, dx, n = shown
    amp = {"before": before, "after": after, "removed": np.abs(before - after)}[which]
    ref = float(before.max()) or 1.0
    fig = Figure(figsize=figsize, facecolor="white")
    sel = f <= f_lim
    ax = fig.add_axes([0.06, 0.10, 0.86, 0.78])
    im = ax.imshow(20 * np.log10(np.maximum(amp[sel] / ref, 1e-4)), aspect="auto", origin="lower", cmap="viridis",
                   vmin=-60, vmax=0, extent=[k[0] * 1000, k[-1] * 1000, f[sel][0], f[sel][-1]])
    if fan is not None:
        for v, style in zip(fan, ("-", "--")):
            kk = np.array([0.0, f_lim / v]) * 1000
            for sgn in (1, -1):
                ax.plot(sgn * kk, [0, f_lim], color="white", linestyle=style, linewidth=1.2)
        zone = f"solid: reject velocity {fan[0]:g}, dashed: pass velocity {fan[1]:g}"
    else:
        zone = "red: the reject polygon" + (" (and its mirror at -k)" if mirror else "")
    for part in (poly or []):
        if len(part) < 3:
            continue
        for sgn in ((1, -1) if mirror else (1,)):
            closed = np.vstack([part, part[:1]])
            ax.plot(sgn * closed[:, 0], closed[:, 1], color="#ff4d4d", linewidth=1.6)
    ax.set_xlim(k[0] * 1000, k[-1] * 1000)
    ax.set_ylim(0, f_lim)
    ax.set_xlabel("Wavenumber k (cycles per 1000 length units)", color=INK_2)
    ax.set_ylabel("Frequency f (Hz)", color=INK_2)
    ax.tick_params(labelsize=9, colors=INK_2)
    cax = fig.add_axes([0.935, 0.10, 0.012, 0.78])
    fig.colorbar(im, cax=cax).set_label("dB re input peak", color=INK_2)
    what = {"before": "BEFORE the filter", "after": "AFTER the filter", "removed": "REMOVED noise"}[which]
    title = (f"F-K spectrum {what} - longest receiver line ({n} traces, spacing {dx:g}) - {zone}" if figsize[0] >= 10
             else f"F-K {what.lower()} ({n} traces, spacing {dx:g})\n{zone}")
    fig.suptitle(title, x=0.06, y=0.985, ha="left", va="top", fontsize=11 if figsize[0] < 10 else 12, color=INK)
    return fig


def plot_radon(model: np.ndarray, p: np.ndarray, cut: float, dt_ms: float, kind: str, figsize=(14.0, 5.5)) -> Figure:
    """The Radon panel (tau-p or tau-q) with the cut: right of the line = removed."""
    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.76])
    clip = float(np.percentile(np.abs(model), 99)) or 1.0
    t_end = model.shape[1] * dt_ms
    ax.imshow(model.T, aspect="auto", cmap="gray_r", vmin=-clip, vmax=clip, extent=[p[0], p[-1], t_end, 0])
    ax.axvline(cut, color="#e34948", linewidth=1.6)
    ax.axvspan(cut, p[-1], color="#e34948", alpha=0.08)
    ax.set_xlabel("Slowness p (ms per length unit)" if kind == "linear" else "Moveout q at the far offset (ms)",
                  color=INK_2)
    ax.set_ylabel("Intercept time tau (ms)", color=INK_2)
    ax.tick_params(labelsize=9, colors=INK_2)
    what = "slow linear events (ground roll)" if kind == "linear" else "residual moveout (multiples)"
    fig.suptitle(f"{'Linear (tau-p)' if kind == 'linear' else 'Parabolic (tau-q)'} Radon panel - right of the red line "
                 f"= {what}, removed", x=0.06, y=0.97, ha="left", fontsize=12, color=INK)
    return fig
