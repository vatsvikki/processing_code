"""NMO correction of a CDP gather with velocities from a velocity-model SEG-Y - the code of the brute-stack notebook
(cdp_nmo_stack_marimo.py: fit_corners, model_velocity_for, nmo_stack), ported unchanged in what it computes.

* The velocity model's trace positions come from its INLINE_3D / CROSSLINE_3D words (bytes 189 / 193, SEG-Y rev1) and a
  corner-point registration (IL, XL <-> X, Y: X linear in XL, Y linear in IL - `fit_corners`).
* `model_velocity_for(x, y)` takes the model trace nearest to a CDP's real position, converts its units, turns it into
  an RMS velocity against two-way time (Dix for interval velocities; depth models integrated to time first) and
  interpolates it onto the data's time axis.
* `nmo_stack` moves every sample to t0: t(x) = sqrt(t0^2 + (x / V(t0))^2), with an NMO-stretch mute.
No UI here - the "NMO Correction" step of the Flow (steps.py) uses it.
"""
from __future__ import annotations

from functools import lru_cache
import os
import re

import numpy as np

from . import segy_io
from .segy_io import header_word

FT_PER_M = 3.280839895
MODEL_TYPES = ["RMS velocity", "Interval velocity"]
UNITS = ["ft/s", "m/s"]
DOMAINS = ["TWTT (time)", "Depth"]
_INLINE_3D, _CROSSLINE_3D = (189, "i4"), (193, "i4")    # the model's grid words (SEG-Y rev1), as in the notebook


def fit_corners(rows):
    """(IL, XL) -> (X, Y) forward transform from a corner table: X = mx * XL + cx, Y = my * IL + cy (least squares).
    rows: (IL, XL, X, Y) tuples. None when there are not 2 rows with distinct IL and distinct XL."""
    rows = [r for r in rows if len(r) == 4]
    _il = np.array([r[0] for r in rows], dtype=np.float64)
    _xl = np.array([r[1] for r in rows], dtype=np.float64)
    _x = np.array([r[2] for r in rows], dtype=np.float64)
    _y = np.array([r[3] for r in rows], dtype=np.float64)
    _valid = len(rows) >= 2 and _xl.max() != _xl.min() and _il.max() != _il.min()
    if not _valid:
        return None
    _mx, _cx = np.polyfit(_xl, _x, 1)
    _my, _cy = np.polyfit(_il, _y, 1)
    return {"mx": _mx, "cx": _cx, "my": _my, "cy": _cy}


@lru_cache(maxsize=2)
def _model_scan(path: str, mtime: float) -> dict:
    """INLINE_3D / CROSSLINE_3D of every trace of the velocity model (headers only, once per file version)."""
    sgy = segy_io.open_segy(path)
    raw = sgy.read_headers(0, sgy.ntraces)
    return {"inline3d": header_word(raw, *_INLINE_3D, sgy.order), "crossline3d": header_word(raw, *_CROSSLINE_3D, sgy.order),
            "trace_idx": np.arange(sgy.ntraces, dtype=np.int64), "ns": sgy.ns, "dt_us": sgy.dt_us}


def model_scan(path: str) -> dict:
    sgy = segy_io.open_segy(path)
    return _model_scan(sgy.path, os.path.getmtime(sgy.path))


def model_velocity_for(model_path: str, scan: dict, model_transform: dict, x: float, y: float, full_t: np.ndarray, *,
                       velocity_model_type: str, model_velocity_units: str, model_domain: str,
                       model_sample_interval: float, output_velocity_units: str):
    """Real-position-matched (not CDP-matched) velocity lookup: find the model trace nearest this (x, y), decode it,
    convert to a time-domain RMS velocity, and interpolate onto the data's own time axis.
    Returns (vel_of_t0, match_info)."""
    _model_x = model_transform["mx"] * scan["crossline3d"].astype(np.float64) + model_transform["cx"]
    _model_y = model_transform["my"] * scan["inline3d"].astype(np.float64) + model_transform["cy"]
    _dist2 = (_model_x - x) ** 2 + (_model_y - y) ** 2
    _best = int(np.argmin(_dist2))
    _trace_idx = int(scan["trace_idx"][_best])
    _match_info = {
        "inline3d": int(scan["inline3d"][_best]),
        "crossline3d": int(scan["crossline3d"][_best]),
        "distance_ft": float(np.sqrt(_dist2[_best])),
    }

    _ns = scan["ns"]
    _native = segy_io.open_segy(model_path).read_traces_at([_trace_idx])[1][0].astype("float64")

    # Convert the input velocity (whatever units it's declared in) into the chosen OUTPUT unit BEFORE any Dix
    # conversion, so interval->RMS happens in one consistent system rather than mixing units mid-calculation. The
    # data's offsets get the same treatment (cdp_offsets), so NMO always sees offset and velocity in matching units.
    if model_velocity_units == "m/s" and output_velocity_units == "ft/s":
        _native = _native * FT_PER_M
    elif model_velocity_units == "ft/s" and output_velocity_units == "m/s":
        _native = _native / FT_PER_M

    if model_domain == "Depth":
        # Depth-domain models are always treated as interval velocity vs depth -- integrate to get two-way time per
        # depth step, then Dix-convert on that (irregular) time axis.
        _dz = model_sample_interval
        _vint = _native
        _dt_i = np.where(_vint > 0, 2.0 * _dz / np.where(_vint > 0, _vint, 1.0), 0.0)
        _model_t = np.cumsum(_dt_i)
        _cum_v2dt = np.cumsum(_vint ** 2 * _dt_i)
        _vrms_native = np.sqrt(
            np.divide(_cum_v2dt, _model_t, out=np.full_like(_cum_v2dt, _vint[0] ** 2), where=_model_t > 0)
        )
        _vint_on_t = _vint
        _native_axis = np.arange(_ns) * _dz
        _native_axis_kind = "Depth"
    else:
        _model_dt_s = model_sample_interval / 1000.0
        _model_t = np.arange(_ns) * _model_dt_s
        if velocity_model_type == "Interval velocity":
            _dt_arr = np.diff(_model_t, prepend=0.0)
            _cum = np.cumsum(_native ** 2 * _dt_arr)
            _vrms_native = np.sqrt(
                np.divide(_cum, _model_t, out=np.full_like(_cum, _native[0] ** 2), where=_model_t > 0)
            )
            _vint_on_t = _native
        else:
            _vrms_native = _native
            # Native data is RMS-only -- invert Dix to get the equivalent interval velocity.
            _dt_arr = np.diff(_model_t, prepend=0.0)
            _cum = _model_t * _vrms_native ** 2
            _prev_cum = np.concatenate(([0.0], _cum[:-1]))
            _vint2 = np.divide(
                _cum - _prev_cum, _dt_arr,
                out=np.full_like(_cum, _vrms_native[0] ** 2), where=_dt_arr > 0,
            )
            _vint_on_t = np.sqrt(np.maximum(_vint2, 0.0))
        _native_axis = _model_t
        _native_axis_kind = "Time"

    # Exposed for the input-vs-output velocity comparison plot -- not needed for the interpolated result used in NMO.
    _match_info["native_velocity"] = _native
    _match_info["native_axis"] = _native_axis
    _match_info["native_axis_kind"] = _native_axis_kind
    _match_info["vrms_native"] = _vrms_native
    _match_info["model_t"] = _model_t

    _vel_of_t0 = np.interp(full_t, _model_t, _vrms_native, left=_vrms_native[0], right=_vrms_native[-1])
    _match_info["interval_of_t0"] = np.interp(
        full_t, _model_t, _vint_on_t, left=_vint_on_t[0], right=_vint_on_t[-1]
    )
    return _vel_of_t0, _match_info


def nmo_stack(traces, offsets, full_t, vel_of_t0, mute_frac, post_nmo_filter=None):
    """NMO of every trace (t(x) = sqrt(t0^2 + (x / V(t0))^2)) with the stretch mute; returns (NMO-corrected traces with
    muted samples 0, stack = mean of the live samples at each time)."""
    _n_traces, _ns = traces.shape
    _nmo = np.full((_n_traces, _ns), np.nan)
    for _i in range(_n_traces):
        _x = offsets[_i]
        _t_moveout = np.sqrt(full_t ** 2 + (_x / vel_of_t0) ** 2)
        _amp = np.interp(_t_moveout, full_t, traces[_i], left=0.0, right=0.0)
        if mute_frac > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                _stretch = np.where(full_t > 0, (_t_moveout - full_t) / full_t, 0.0)
            _amp = np.where(np.abs(_stretch) > mute_frac, np.nan, _amp)
        _nmo[_i] = _amp
    _live_count = np.sum(~np.isnan(_nmo), axis=0)
    _nmo_filled = np.nan_to_num(_nmo, nan=0.0)
    if post_nmo_filter is not None:
        # Filtering after the mute zeroes out muted samples smears a little energy across that boundary -- an expected
        # real effect of filtering a muted gather, not a bug.
        _nmo_filled = post_nmo_filter(_nmo_filled)
    _stack = np.divide(
        np.sum(_nmo_filled, axis=0), _live_count,
        out=np.zeros(_ns), where=_live_count > 0,
    )
    return _nmo_filled, _stack


def cdp_offsets(offsets: np.ndarray, output_velocity_units: str) -> np.ndarray:
    """Offsets in the output velocity's length unit (the survey's headers are in feet, as the notebook assumes)."""
    off = np.abs(offsets.astype(np.float64))
    return off / FT_PER_M if output_velocity_units == "m/s" else off


def sanity_warnings(vel_of_t0: np.ndarray, match_info: dict, unit: str) -> list[str]:
    """The notebook's two checks: a model trace far from the CDP, and velocities outside the usual seismic range."""
    w = []
    if match_info["distance_ft"] > 5000:
        w.append(f"Matched model trace is {match_info['distance_ft']:,.0f} ft away -- that's far for a velocity lookup. "
                 "Check the model's corner points (same-as-data is only correct if the model truly uses the data's own "
                 "grid numbering) and the INLINE_3D/CROSSLINE_3D byte-offset assumption (bytes 189 / 193).")
    lo, hi = (5000, 25000) if unit == "ft/s" else (1500, 7600)
    if vel_of_t0.min() < lo or vel_of_t0.max() > hi:
        w.append(f"Velocity range {vel_of_t0.min():,.0f}-{vel_of_t0.max():,.0f} {unit} is outside the typical seismic "
                 f"range ({lo:,}-{hi:,} {unit}) -- check the input velocity units and RMS/Interval selection.")
    return w


def corners_from_text_header(lines: list[str]) -> list[tuple[float, float, float, float]]:
    """Grid corners written in a SEG-Y text header (as in this survey's velocity model: a "GRID CORNERS" line, then one
    "IL  XL  X  Y" row per corner). [] when there is no such block."""
    out, on = [], False
    num = r"-?\d+(?:\.\d+)?"
    for line in lines:
        body = re.sub(r"^C\s*\d+\s?", "", line)          # drop the "C12" card number
        if "CORNER" in body.upper():
            on = True
        if not on:
            continue
        m = re.findall(num, body)
        if len(m) >= 4 and re.fullmatch(r"[\s\d.\-]*", body.split(":")[-1] if ":" in body else body):
            out.append(tuple(float(v) for v in m[-4:]))
        elif out:                                          # the block has ended
            break
    return out[:4] if len(out) >= 2 else []


def model_corners_from_header(path: str) -> list[tuple[float, float, float, float]]:
    return corners_from_text_header(segy_io.open_segy(path).ebcdic)


def overlay_points(scan: dict, model_transform: dict, data_transform: dict, cdp_x: np.ndarray, cdp_y: np.ndarray,
                   max_points: int = 30000):
    """The notebook's geometry overlay in data IL / XL terms: (model grid XL, IL), (data CDP XL, IL) - both de-duplicated
    and evenly decimated to at most max_points. The model's unique (INLINE_3D, CROSSLINE_3D) go to X / Y through the
    model's corner points and back through the data's; the CDPs go from their X / Y through the data's."""
    m = np.unique(np.column_stack([scan["crossline3d"], scan["inline3d"]]), axis=0).astype(np.float64)
    mx = model_transform["mx"] * m[:, 0] + model_transform["cx"]
    my = model_transform["my"] * m[:, 1] + model_transform["cy"]
    model_xl = (mx - data_transform["cx"]) / data_transform["mx"]
    model_il = (my - data_transform["cy"]) / data_transform["my"]
    d = np.unique(np.column_stack([np.round((cdp_x - data_transform["cx"]) / data_transform["mx"]),
                                   np.round((cdp_y - data_transform["cy"]) / data_transform["my"])]), axis=0)

    def _dec(x, y):
        if len(x) <= max_points:
            return x, y
        step = int(np.ceil(len(x) / max_points))
        return x[::step], y[::step]

    return _dec(model_xl, model_il), _dec(d[:, 0], d[:, 1])


def to_data_ilxl(data_transform: dict, x: float, y: float) -> tuple[float, float]:
    """(XL, IL) of a real X / Y position in the data's grid."""
    return (x - data_transform["cx"]) / data_transform["mx"], (y - data_transform["cy"]) / data_transform["my"]


# ---------------------------------------------------------------------------------------------------------------------
# a velocity FUNCTION instead of a model: one time / RMS-velocity table for every CDP
# ---------------------------------------------------------------------------------------------------------------------
VELFN_TIME, VELFN_VEL = "Time (ms)", "Vrms"
DEFAULT_VELFN = [{VELFN_TIME: 0.0, VELFN_VEL: 5000.0}, {VELFN_TIME: 1000.0, VELFN_VEL: 6500.0},
                 {VELFN_TIME: 2000.0, VELFN_VEL: 8000.0}, {VELFN_TIME: 4000.0, VELFN_VEL: 10000.0}]


def velocity_function(rows, full_t: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """RMS velocity at every time of full_t (s) from a table of (time ms, Vrms) rows: linear between rows, held flat
    before the first and after the last. Returns (velocity, the rows used); empty rows are skipped."""
    pts = []
    for r in rows or []:
        low = {str(k).strip().lower(): v for k, v in r.items()} if isinstance(r, dict) else {}
        t = low.get(VELFN_TIME.lower(), low.get("time"))
        v = low.get(VELFN_VEL.lower(), low.get("velocity"))
        if t in (None, "") or v in (None, ""):
            continue
        t, v = float(t), float(v)
        if v <= 0:
            raise ValueError(f"velocity at {t:g} ms is {v:g} - every velocity must be > 0")
        if t < 0:
            raise ValueError(f"time {t:g} ms is negative")
        pts.append((t, v))
    if not pts:
        raise ValueError("the velocity function is empty - enter at least one time / velocity row")
    pts.sort()
    times = np.array([p[0] for p in pts]) / 1000.0
    if np.any(np.diff(times) == 0):
        raise ValueError("two rows of the velocity function have the same time")
    return np.interp(full_t, times, np.array([p[1] for p in pts])), pts


def interval_from_rms(full_t: np.ndarray, vrms: np.ndarray) -> np.ndarray:
    """Equivalent interval velocity of an RMS function (inverse Dix, as the notebook does for an RMS model)."""
    dt_arr = np.diff(full_t, prepend=0.0)
    cum = full_t * vrms ** 2
    prev = np.concatenate(([0.0], cum[:-1]))
    vint2 = np.divide(cum - prev, dt_arr, out=np.full_like(cum, vrms[0] ** 2), where=dt_arr > 0)
    return np.sqrt(np.maximum(vint2, 0.0))


# ---------------------------------------------------------------------------------------------------------------------
# the velocity of ONE model trace (model_velocity_for = nearest trace + this), and the nearest trace of MANY positions
# ---------------------------------------------------------------------------------------------------------------------
def model_trace_velocity(model_path: str, scan: dict, trace_idx: int, full_t: np.ndarray, *, velocity_model_type: str,
                         model_velocity_units: str, model_domain: str, model_sample_interval: float,
                         output_velocity_units: str):
    """(RMS velocity on full_t, interval velocity on full_t) of one model trace - the conversion part of
    model_velocity_for (same steps, same numbers)."""
    vel, info = model_velocity_for(model_path, {**scan, "inline3d": scan["inline3d"][trace_idx:trace_idx + 1],
                                                "crossline3d": scan["crossline3d"][trace_idx:trace_idx + 1],
                                                "trace_idx": scan["trace_idx"][trace_idx:trace_idx + 1]},
                                   {"mx": 0.0, "cx": 0.0, "my": 0.0, "cy": 0.0}, 0.0, 0.0, full_t,
                                   velocity_model_type=velocity_model_type, model_velocity_units=model_velocity_units,
                                   model_domain=model_domain, model_sample_interval=model_sample_interval,
                                   output_velocity_units=output_velocity_units)
    return vel, info["interval_of_t0"], info


def nearest_model_traces(scan: dict, model_transform: dict, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Index (into the scan) of the model trace nearest each (x, y) - the same trace model_velocity_for picks by brute
    force, found fast through the model's regular grid (round to the nearest INLINE_3D / CROSSLINE_3D node); positions
    whose node is missing from the model fall back to the brute-force search."""
    il = scan["inline3d"].astype(np.int64)
    xl = scan["crossline3d"].astype(np.int64)
    key = il * 10_000_000 + xl
    order = np.argsort(key, kind="stable")
    skey = key[order]
    il_f = (np.asarray(ys, np.float64) - model_transform["cy"]) / model_transform["my"]
    xl_f = (np.asarray(xs, np.float64) - model_transform["cx"]) / model_transform["mx"]
    want = np.rint(il_f).astype(np.int64) * 10_000_000 + np.rint(xl_f).astype(np.int64)
    pos = np.searchsorted(skey, want)
    found = (pos < len(skey)) & (skey[np.minimum(pos, len(skey) - 1)] == want)
    out = np.where(found, order[np.minimum(pos, len(skey) - 1)], -1)
    if (~found).any():
        mx = model_transform["mx"] * scan["crossline3d"].astype(np.float64) + model_transform["cx"]
        my = model_transform["my"] * scan["inline3d"].astype(np.float64) + model_transform["cy"]
        for i in np.flatnonzero(~found):
            out[i] = int(np.argmin((mx - xs[i]) ** 2 + (my - ys[i]) ** 2))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the notebook's filter, and NMO of many traces at once (the same numbers as nmo_stack, without its per-trace loop)
# ---------------------------------------------------------------------------------------------------------------------
FILTER_TYPES = ["Low-pass", "High-pass", "Bandpass"]


def bandpass_filter(arr, dt_s, low_hz, high_hz, order=4):
    """Zero-phase Butterworth bandpass along the last (time) axis. low_hz<=0 becomes a low-pass; high_hz at/above
    Nyquist becomes a high-pass; a band covering the whole spectrum is a no-op (as in the notebook)."""
    from scipy.signal import butter, filtfilt
    _nyq = 0.5 / dt_s
    _lo = max(low_hz, 0.0) / _nyq
    _hi = min(high_hz, _nyq * 0.99) / _nyq
    if _lo <= 0.0 and _hi >= 0.99:
        return arr
    if _lo <= 0.0:
        _b, _a = butter(order, _hi, btype="low")
    elif _hi >= 0.99:
        _b, _a = butter(order, _lo, btype="high")
    else:
        _b, _a = butter(order, [_lo, _hi], btype="band")
    return filtfilt(_b, _a, arr, axis=-1)


def filter_band(filter_type: str, cutoff_hz: float, low_hz: float, high_hz: float) -> tuple[float, float]:
    """The (low, high) pair bandpass_filter understands, from the notebook's filter panel."""
    if filter_type == "Low-pass":
        return 0.0, cutoff_hz
    if filter_type == "High-pass":
        return cutoff_hz, 1.0e9
    return low_hz, high_hz


def nmo_many(traces: np.ndarray, offsets: np.ndarray, full_t: np.ndarray, vel: np.ndarray, mute_frac: float):
    """NMO of every trace (rows of `traces`) with the stretch mute: (corrected traces with muted samples 0, live mask).
    vel: [ns] for all traces, or [ntr, ns] one row per trace. Same as nmo_stack's loop (linear interpolation, 0 past
    the end of the record, stretch taken as 0 at t0 = 0), vectorized."""
    ntr, ns = traces.shape
    dt = float(full_t[1] - full_t[0])
    x = np.abs(np.asarray(offsets, np.float64))[:, None]
    v = np.asarray(vel, np.float64)
    v = v[None, :] if v.ndim == 1 else v
    t0 = full_t[None, :]
    tx = np.sqrt(t0 ** 2 + (x / v) ** 2)
    pos = (tx - full_t[0]) / dt
    inside = pos <= ns - 1
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, ns - 2)
    w = pos - i0
    rows = np.arange(ntr)[:, None]
    amp = traces[rows, i0] * (1.0 - w) + traces[rows, i0 + 1] * w
    amp = np.where(inside, amp, 0.0)
    live = np.ones((ntr, ns), bool)
    if mute_frac > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = np.where(t0 > 0, (tx - t0) / t0, 0.0)
        live = np.abs(stretch) <= mute_frac
        amp = np.where(live, amp, 0.0)
    return amp, live
