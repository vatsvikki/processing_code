"""Acquisition geometry from the trace headers: the source of every shot and the receiver positions (no UI, no plotting).

Coordinates are the SEG-Y header words src_x / src_y (bytes 73, 77) and rec_x / rec_y (81, 85), multiplied by the
coordinate scalar (byte 71: > 0 multiply, < 0 divide, 0 = 1).  Only the headers are read, never the samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np

from . import segy_io
from .segy_io import header_word

_SCALAR, _UNITS = (71, "i2"), (89, "i2")
_SRC_X, _SRC_Y, _REC_X, _REC_Y = (73, "i4"), (77, "i4"), (81, "i4"), (85, "i4")
_INLINE, _XLINE = (189, "i4"), (193, "i4")
_CDP = (21, "i4")                                     # ensemble (CDP) number


@dataclass
class Geometry:
    ffids: np.ndarray                # every shot
    src_x: np.ndarray                # source position of each shot (first trace's header)
    src_y: np.ndarray
    rec_x: np.ndarray                # distinct receiver positions found in the shots that were read
    rec_y: np.ndarray
    rec_line: np.ndarray
    rec_stn: np.ndarray
    shots_read: int                  # number of shots the receiver positions come from
    traces_read: int
    unit: str                        # "ft" | "m" | "deg" | "DMS" | ""
    hdr_il: np.ndarray               # inline / crossline words and the source-receiver midpoint of up to 20 000
    hdr_xl: np.ndarray               # of the traces read (to test whether the headers describe a regular grid)
    hdr_x: np.ndarray
    hdr_y: np.ndarray


@dataclass
class Spread:
    """One shot: its source and the receivers that recorded it."""
    ffid: int
    src_x: float
    src_y: float
    rec_x: np.ndarray
    rec_y: np.ndarray
    offset: np.ndarray               # header offset word, one per trace
    chan: np.ndarray
    rec_line: np.ndarray
    rec_stn: np.ndarray


def _scale(raw: np.ndarray, order: str) -> np.ndarray:
    """Per-trace factor of the coordinate scalar and units (arc seconds -> degrees)."""
    sc = header_word(raw, *_SCALAR, order).astype(float)
    f = np.ones(len(sc))
    f[sc > 0] = sc[sc > 0]
    f[sc < 0] = 1.0 / -sc[sc < 0]
    if len(raw) and int(header_word(raw[:1], *_UNITS, order)[0]) == 2:
        f = f / 3600.0
    return f


def _unit(sgy: segy_io.SegyFile, raw: np.ndarray) -> str:
    cu = int(header_word(raw[:1], *_UNITS, sgy.order)[0]) if len(raw) else 1
    if cu in (2, 3):
        return "deg"
    if cu == 4:
        return "DMS"
    return {1: "m", 2: "ft"}.get(int(sgy.binary_header.get("measurement_system", 0)), "")


def _xy(raw: np.ndarray, order: str, kx, ky) -> tuple[np.ndarray, np.ndarray]:
    f = _scale(raw, order)
    return header_word(raw, *kx, order) * f, header_word(raw, *ky, order) * f


@lru_cache(maxsize=4)
def _geometry_cached(path: str, mtime: float, rec_shots: int) -> Geometry:
    sgy = segy_io.open_segy(path)
    idx = segy_io.shot_index(path)
    src = sgy.read_headers_at(idx.first)
    sx, sy = _xy(src, sgy.order, _SRC_X, _SRC_Y)

    n = len(idx)
    pick = np.arange(n) if rec_shots <= 0 or rec_shots >= n else np.unique(np.linspace(0, n - 1, rec_shots).round().astype(int))
    blocks = [sgy.read_headers(int(idx.first[k]), int(idx.first[k] + idx.count[k])) for k in pick]
    raw = np.concatenate(blocks) if blocks else np.zeros((0, segy_io.TRACE_HEADER_BYTES), np.uint8)
    key = np.stack([header_word(raw, *_REC_X, sgy.order), header_word(raw, *_REC_Y, sgy.order),
                    header_word(raw, *segy_io.TRACE_HEADER_FIELDS["rec_line"], sgy.order),
                    header_word(raw, *segy_io.TRACE_HEADER_FIELDS["rec_stn"], sgy.order)], axis=1)
    _, first = np.unique(key[:, :2], axis=0, return_index=True)      # one row per distinct receiver position
    raw_u = raw[np.sort(first)]
    smp = raw[np.linspace(0, max(len(raw) - 1, 0), min(len(raw), 20000)).astype(int)] if len(raw) else raw
    f = _scale(smp, sgy.order) if len(smp) else np.zeros(0)
    mid_x = (header_word(smp, *_SRC_X, sgy.order) + header_word(smp, *_REC_X, sgy.order)) / 2 * f
    mid_y = (header_word(smp, *_SRC_Y, sgy.order) + header_word(smp, *_REC_Y, sgy.order)) / 2 * f
    rx, ry = _xy(raw_u, sgy.order, _REC_X, _REC_Y)
    return Geometry(
        ffids=idx.ffids.copy(), src_x=sx, src_y=sy, rec_x=rx, rec_y=ry,
        rec_line=header_word(raw_u, *segy_io.TRACE_HEADER_FIELDS["rec_line"], sgy.order),
        rec_stn=header_word(raw_u, *segy_io.TRACE_HEADER_FIELDS["rec_stn"], sgy.order),
        shots_read=len(pick), traces_read=len(raw), unit=_unit(sgy, src),
        hdr_il=header_word(smp, *_INLINE, sgy.order).astype(float), hdr_xl=header_word(smp, *_XLINE, sgy.order).astype(float),
        hdr_x=mid_x, hdr_y=mid_y,
    )


@lru_cache(maxsize=2)
def _midpoints_cached(path: str, mtime: float) -> tuple:
    sgy = segy_io.open_segy(path)
    n = sgy.ntraces
    mx, my, cdp = np.empty(n), np.empty(n), np.empty(n, np.int64)
    off_hdr, off_xy = np.empty(n, np.int64), np.empty(n)              # offset header word / offset from the coordinates
    key, zero_xy = np.empty(n, np.int64), np.zeros(n, bool)           # FFID * 100000 + channel; coordinates all zero
    step = 100_000
    for i0 in range(0, n, step):                     # headers only, a block at a time (the samples are never read)
        raw = sgy.read_headers(i0, i0 + step)
        f = _scale(raw, sgy.order)
        i1 = i0 + len(raw)
        sx, sy = header_word(raw, *_SRC_X, sgy.order), header_word(raw, *_SRC_Y, sgy.order)
        rx, ry = header_word(raw, *_REC_X, sgy.order), header_word(raw, *_REC_Y, sgy.order)
        mx[i0:i1], my[i0:i1] = (sx + rx) / 2 * f, (sy + ry) / 2 * f
        cdp[i0:i1] = header_word(raw, *_CDP, sgy.order)
        off_hdr[i0:i1] = header_word(raw, *segy_io.TRACE_HEADER_FIELDS["offset"], sgy.order)
        off_xy[i0:i1] = np.hypot(sx - rx, sy - ry) * f
        key[i0:i1] = header_word(raw, *segy_io.TRACE_HEADER_FIELDS["ffid"], sgy.order) * 100000 + \
            header_word(raw, *segy_io.TRACE_HEADER_FIELDS["chan"], sgy.order)
        zero_xy[i0:i1] = ((sx == 0) & (sy == 0)) | ((rx == 0) & (ry == 0))
    return mx, my, cdp, off_hdr, off_xy, key, zero_xy


def read_midpoints(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Source-receiver midpoint (x, y), scaled, of EVERY trace of the file.  The first call reads all trace headers
    (about 20 s for 950 000 traces); the result is kept, so binning it again (fold maps) is instant."""
    path = os.path.expanduser(str(path).strip())
    segy_io.open_segy(path)
    return _midpoints_cached(path, os.path.getmtime(path))[:2]


def read_trace_table(path: str) -> dict:
    """Header words of EVERY trace from the same single pass as read_midpoints: midpoint x / y, CDP number, the offset
    header word, the offset computed from the source / receiver coordinates, FFID * 100000 + channel (a unique trace key)
    and whether the source or receiver coordinates are all zero.  Used by the CMP sort QC."""
    path = os.path.expanduser(str(path).strip())
    segy_io.open_segy(path)
    mx, my, cdp, off_hdr, off_xy, key, zero_xy = _midpoints_cached(path, os.path.getmtime(path))
    return {"mx": mx, "my": my, "cdp": cdp, "offset": off_hdr, "offset_xy": off_xy, "key": key, "zero_xy": zero_xy}


def read_cdp(path: str) -> np.ndarray:
    """The CDP / ensemble number (header bytes 21-24) of every trace - from the same pass as read_midpoints."""
    path = os.path.expanduser(str(path).strip())
    segy_io.open_segy(path)
    return _midpoints_cached(path, os.path.getmtime(path))[2]


def read_geometry(path: str, rec_shots: int = 60) -> Geometry:
    """Source of every shot + the distinct receiver positions of `rec_shots` evenly spaced shots (0 = every shot)."""
    path = os.path.expanduser(str(path).strip())
    segy_io.open_segy(path)
    return _geometry_cached(path, os.path.getmtime(path), int(rec_shots))


def read_spread(path: str, ffid: int = 0) -> Spread:
    """Source and receivers of one shot (the FFID nearest to `ffid`; 0 = the first shot)."""
    path = os.path.expanduser(str(path).strip())
    sgy = segy_io.open_segy(path)
    idx = segy_io.shot_index(path)
    k = idx.position(idx.nearest(int(ffid)))
    raw = sgy.read_headers(int(idx.first[k]), int(idx.first[k] + idx.count[k]))
    sx, sy = _xy(raw[:1], sgy.order, _SRC_X, _SRC_Y)
    rx, ry = _xy(raw, sgy.order, _REC_X, _REC_Y)
    w = lambda name: header_word(raw, *segy_io.TRACE_HEADER_FIELDS[name], sgy.order)
    return Spread(int(idx.ffids[k]), float(sx[0]), float(sy[0]), rx, ry, w("offset"), w("chan"), w("rec_line"), w("rec_stn"))


def summary_rows(g: Geometry) -> list[dict]:
    """Headline numbers of the survey as [{'Item', 'Value'}]."""
    u = f" {g.unit}" if g.unit else ""
    rows = [
        {"Item": "Shots", "Value": f"{len(g.ffids):,}  (FFID {int(g.ffids[0])} - {int(g.ffids[-1])})"},
        {"Item": "Source X range", "Value": f"{g.src_x.min():,.1f} - {g.src_x.max():,.1f}{u}"},
        {"Item": "Source Y range", "Value": f"{g.src_y.min():,.1f} - {g.src_y.max():,.1f}{u}"},
    ]
    if len(g.ffids) > 1:
        d = np.hypot(np.diff(g.src_x), np.diff(g.src_y))
        d = d[d > 0]
        if len(d):
            rows.append({"Item": "Distance between consecutive shots (median)", "Value": f"{np.median(d):,.1f}{u}"})
    part = "" if g.shots_read >= len(g.ffids) else f"  (from {g.shots_read} of {len(g.ffids)} shots)"
    rows += [
        {"Item": "Receiver positions", "Value": f"{len(g.rec_x):,}{part}"},
        {"Item": "Receiver X range", "Value": f"{g.rec_x.min():,.1f} - {g.rec_x.max():,.1f}{u}"},
        {"Item": "Receiver Y range", "Value": f"{g.rec_y.min():,.1f} - {g.rec_y.max():,.1f}{u}"},
        {"Item": "Receiver lines", "Value": f"{len(np.unique(g.rec_line)):,}  (line {int(g.rec_line.min())} - {int(g.rec_line.max())})"},
        {"Item": "Receiver stations", "Value": f"{int(g.rec_stn.min())} - {int(g.rec_stn.max())}"},
    ]
    steps = []
    for line in np.unique(g.rec_line):
        m = g.rec_line == line
        o = np.argsort(g.rec_stn[m])
        stn, x, y = g.rec_stn[m][o], g.rec_x[m][o], g.rec_y[m][o]
        ds = np.abs(np.diff(stn))
        ok = ds > 0
        steps += list(np.hypot(np.diff(x), np.diff(y))[ok] / ds[ok])
    if steps:
        rows.append({"Item": "Receiver spacing per station number (median)", "Value": f"{np.median(steps):,.1f}{u}"})
    return rows
