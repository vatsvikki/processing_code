"""CDP stacking - the brute-stack notebook (cdp_nmo_stack_marimo.py, sections 6-8) for the whole survey: every CDP's
traces NMO-corrected (with that CDP's own velocity) and stacked; IL / XL sections of the stack and of the velocity; the
stack written to SEG-Y. No UI here - the "CDP Stack" step of the Flow (steps.py) uses it.

The stack of a CDP is the notebook's: the mean, at every time, of the live (not stretch-muted) NMO-corrected samples
(sum / live count) - or, without NMO, the plain mean of the traces. Instead of reading the file CDP by CDP (a jump
through the whole file for every CDP) it is read ONCE from start to end, each trace NMO-corrected and added into its
CDP's row of two arrays kept on disk (sums and live counts, so a big survey never has to fit in memory); the stack is
sums / counts at the end. The result is kept in the output folder per file and settings, so a section, a different
IL / XL or a replot never stacks again.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import nmo, segy_io
from .cdp_sort import CdpIndex
from .progress import report
from .saving import DEFAULT_DIR
from .velocity_setup import VelocitySetup

_CHUNK = 2000                                   # traces per read / NMO block
_THREADS = max(1, min(4, os.cpu_count() or 1))   # NMO of a block in this many threads (numpy runs them in parallel)


def _memory_available() -> int:
    """Free memory in bytes (Linux: MemAvailable), 0 when unknown (then the sums go to a file on disk)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0
FILTER_STAGES = ("raw", "post_nmo", "stack")


@dataclass
class FullStack:
    folder: Path
    cdps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ns: int
    dt_ms: float
    meta: dict

    @property
    def data(self) -> np.ndarray:
        """The stack, one row per CDP (memory-mapped, read-only)."""
        return np.memmap(self.folder / "stack.f32", dtype=np.float32, mode="r", shape=(len(self.cdps), self.ns))


def _signature(path: str, vs: VelocitySetup, mute_pct: float, band, stages: dict) -> str:
    sgy = segy_io.open_segy(path)
    key = repr((sgy.path, os.path.getmtime(sgy.path), sgy.ntraces, vs.signature, vs.unit,
                round(float(mute_pct), 6) if vs.nmo else None, band, tuple(sorted(stages.items()))))
    return hashlib.sha1(key.encode()).hexdigest()[:16], key


def _folder(path: str, sig: str) -> Path:
    return Path(DEFAULT_DIR) / "stacks" / f"{Path(path).stem}_{sig}"


def cached(path: str, idx: CdpIndex, vs: VelocitySetup, mute_pct: float, band, stages: dict) -> FullStack | None:
    """The full stack already computed with these settings, if any."""
    sig, _ = _signature(path, vs, mute_pct, band, stages)
    folder = _folder(path, sig)
    try:
        meta = json.loads((folder / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    if not meta.get("done"):
        return None
    return FullStack(folder, idx.cdps, idx.mid_x, idx.mid_y, int(meta["ns"]), float(meta["dt_ms"]), meta)


def describe(vs: VelocitySetup, mute_pct: float, band, stages: dict) -> str:
    """The settings of a stack in words (kept with it)."""
    vel = {"model": "velocity model", "function": "velocity function", "none": "no NMO (plain mean of the gathers)"}.get(
        vs.signature[0] if vs.signature else "none", "")
    if vs.signature and vs.signature[0] == "model":
        vel += f" {os.path.basename(vs.model_path)}"
    filt = (f"filter {band[0]:g}-{band[1]:g} Hz at " + ", ".join(k.replace("_", "-") for k, on in stages.items() if on)
            if band is not None else "no filter")
    return f"{vel}; " + (f"stretch mute {mute_pct:g} %; " if vs.nmo else "") + filt


def full_stack(path: str, idx: CdpIndex, vs: VelocitySetup, mute_pct: float, band, stages: dict) -> FullStack:
    """Stack every CDP of the file (or return the stack already computed with the same settings).
    band = (low_hz, high_hz) of the filter, used at the stages ticked in `stages` (raw / post_nmo / stack)."""
    hit = cached(path, idx, vs, mute_pct, band, stages)
    if hit is not None:
        return hit
    sig, key = _signature(path, vs, mute_pct, band, stages)
    folder = _folder(path, sig)
    if folder.exists():
        shutil.rmtree(folder)                         # an interrupted earlier run
    folder.mkdir(parents=True)
    sgy = segy_io.open_segy(path)
    n, ns, dt_s = sgy.ntraces, sgy.ns, sgy.dt_ms / 1000.0
    n_cdp = len(idx.cdps)
    full_t = vs.full_t
    t_start = time.time()
    filt = (lambda a: nmo.bandpass_filter(a, dt_s, band[0], band[1])) if band is not None else None
    # Without NMO there is no separate correction step, so "after NMO" and "raw gather" both mean "before averaging"
    # - either one filters there once (as in the notebook).
    raw_filter = filt is not None and (stages.get("raw") or (not vs.nmo and stages.get("post_nmo")))
    post_filter = filt is not None and vs.nmo and stages.get("post_nmo")
    stack_filter = filt is not None and stages.get("stack")
    near = vs.nearest_traces(idx.mid_x, idx.mid_y) if vs.nmo and vs.scan is not None else None

    mute = mute_pct / 100.0 if vs.nmo else 0.0
    # the sums of every CDP: in memory when there is room (much faster than a file), else in a file on disk
    in_ram = _memory_available() > n_cdp * ns * 4 * 1.6 + 1.0e9
    sums = (np.zeros((n_cdp, ns), np.float32) if in_ram
            else np.memmap(folder / "sums.f32", dtype=np.float32, mode="w+", shape=(n_cdp, ns)))

    def _read(i0):                                    # the next block is read while this one is worked on
        _, hdr, data = sgy.read_block(i0, min(n, i0 + _CHUNK))
        return hdr["cdp"], hdr["offset"], data

    def _vel_of(c):                                   # Vrms of CDP number index c (model: its nearest model trace)
        return vs.model_trace_velocity(int(near[c]))[0] if near is not None else vs.function

    with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(_THREADS) as pool:
        nxt = reader.submit(_read, 0)
        for i0 in range(0, n, _CHUNK):
            cdp_w, off_w, data = nxt.result()
            if i0 + _CHUNK < n:
                nxt = reader.submit(_read, i0 + _CHUNK)
            d = data.astype(np.float64)
            pos = np.searchsorted(idx.cdps, cdp_w)
            if raw_filter:
                d = filt(d)
            if vs.nmo:
                uniq, inv = np.unique(pos, return_inverse=True)
                vel = np.stack([_vel_of(c) for c in uniq])[inv] if near is not None else vs.function
                off = nmo.cdp_offsets(off_w, vs.unit)
                parts = [r for r in np.array_split(np.arange(len(pos)), _THREADS) if len(r)]
                amp = np.vstack(list(pool.map(
                    lambda r: nmo.nmo_many(d[r], off[r], full_t, vel[r] if vel.ndim == 2 else vel, mute)[0], parts)))
                if post_filter:
                    # Filtering after the mute zeroes out muted samples smears a little energy across that boundary -
                    # an expected real effect of filtering a muted gather (as in the notebook)
                    amp = filt(amp)
            else:
                amp = d
            order = np.argsort(pos, kind="stable")
            u, starts = np.unique(pos[order], return_index=True)
            sums[u] += np.add.reduceat(amp[order], starts, axis=0).astype(np.float32)
            report(min(n, i0 + _CHUNK), n, "Stacking the full data" + ("" if vs.nmo else " (no NMO)"))

    # sums / live counts -> the stack. The live count of a CDP at t0 is the number of its traces not stretch-muted
    # there: with one velocity per CDP, |x| <= V(t0) * t0 * sqrt((1 + mute)^2 - 1) (all of them at t0 = 0, or with no
    # mute) - counted from the CDP's sorted offsets instead of being added up trace by trace.
    out = np.memmap(folder / "stack.f32", dtype=np.float32, mode="w+", shape=(n_cdp, ns)) if in_ram else sums
    k_mute = np.sqrt((1.0 + mute) ** 2 - 1.0) if mute > 0 else None
    for a in range(0, n_cdp, 2000):
        b = min(n_cdp, a + 2000)
        sv = np.asarray(sums[a:b], np.float64)
        if k_mute is None:
            cnt = np.broadcast_to(idx.fold[a:b, None].astype(np.float64), sv.shape)
        else:
            cnt = np.empty(sv.shape)
            for j, c in enumerate(range(a, b)):
                x = nmo.cdp_offsets(idx.offsets[idx.first[c]:idx.first[c] + idx.fold[c]], vs.unit)
                cnt[j] = np.searchsorted(x, _vel_of(c) * full_t * k_mute, side="right")
                cnt[j][full_t <= 0] = idx.fold[c]
        st = np.divide(sv, cnt, out=np.zeros_like(sv), where=cnt > 0)
        if stack_filter:
            st = filt(st)
        out[a:b] = st.astype(np.float32)
        report(b, n_cdp, "Finishing the stack")
    out.flush()
    del sums, out
    if not in_ram:
        os.replace(folder / "sums.f32", folder / "stack.f32")
    meta = {"done": True, "file": str(sgy.path), "n_cdp": n_cdp, "ns": ns, "dt_ms": sgy.dt_ms, "key": key,
            "describe": describe(vs, mute_pct, band, stages), "seconds": round(time.time() - t_start, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return FullStack(folder, idx.cdps, idx.mid_x, idx.mid_y, ns, sgy.dt_ms, meta)


# ---------------------------------------------------------------------------------------------------------------------
# sections (notebook: "Stacked sections", "Velocity model sections", "Overlay")
# ---------------------------------------------------------------------------------------------------------------------
def cdp_ilxl(idx: CdpIndex, data_transform: dict) -> tuple[np.ndarray, np.ndarray]:
    """Data IL / XL of every CDP from its real position and the data's corner points (as the notebook's cdp_il / cdp_xl)."""
    il = np.round((idx.mid_y - data_transform["cy"]) / data_transform["my"]).astype(np.int64)
    xl = np.round((idx.mid_x - data_transform["cx"]) / data_transform["mx"]).astype(np.int64)
    return il, xl


def most_common(a: np.ndarray) -> int:
    u, c = np.unique(a, return_counts=True)
    return int(u[np.argmax(c)])


def section_rows(fixed_arr: np.ndarray, vary_arr: np.ndarray, fixed_value: int):
    """(the CDP rows on the line fixed_arr == value, ordered by vary_arr, their vary values) or None (no CDP there)."""
    rows = np.flatnonzero(fixed_arr == int(fixed_value))
    if not len(rows):
        return None
    order = np.argsort(vary_arr[rows], kind="stable")
    return rows[order], vary_arr[rows][order]


def velocity_rows(vs: VelocitySetup, idx: CdpIndex, rows: np.ndarray, kind: str = "RMS") -> np.ndarray:
    """Velocity of the given CDPs on the data's time axis (RMS or interval); NaN beyond the model's own range."""
    out = np.empty((len(rows), len(vs.full_t)), np.float64)
    if vs.function is not None:
        v = vs.function if kind == "RMS" else nmo.interval_from_rms(vs.full_t, vs.function)
        out[:] = v
        return out
    near = vs.nearest_traces(idx.mid_x[rows], idx.mid_y[rows])
    for i, tr in enumerate(near):
        vel, vint, tmax = vs.model_trace_velocity(int(tr))
        out[i] = vel if kind == "RMS" else vint
        out[i][vs.full_t > tmax] = np.nan
    return out


def export_segy(fs: FullStack, il: np.ndarray, xl: np.ndarray, out_path: str, order: str = ">") -> tuple[str, float]:
    """The full stack as SEG-Y, as the notebook writes it: IEEE float32 samples, one zero-offset trace per CDP with its
    CDP number (bytes 21-24), INLINE_3D / CROSSLINE_3D (189-196) and X / Y x 100 (205-212) in the header."""
    out = Path(os.path.expanduser(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    n_cdp, ns = len(fs.cdps), fs.ns
    dt_us = int(round(fs.dt_ms * 1000))
    data = fs.data
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(b"\x00" * 3200)
        bin_hdr = bytearray(400)
        struct.pack_into(order + "H", bin_hdr, 16, dt_us)
        struct.pack_into(order + "H", bin_hdr, 20, ns)
        struct.pack_into(order + "h", bin_hdr, 24, 5)                    # format 5 = IEEE float32
        f.write(bytes(bin_hdr))
        for a in range(0, n_cdp, 5000):
            b = min(n_cdp, a + 5000)
            hdr = np.zeros((b - a, 240), np.uint8)
            for byte, vals, code in ((1, np.arange(a, b) + 1, "i4"), (21, fs.cdps[a:b], "i4"), (115, np.full(b - a, ns), "u2"),
                                     (117, np.full(b - a, dt_us), "u2"), (189, il[a:b], "i4"), (193, xl[a:b], "i4"),
                                     (205, np.round(fs.x[a:b] * 100), "i4"), (209, np.round(fs.y[a:b] * 100), "i4")):
                segy_io.set_header_word(hdr, byte, vals, code, order)
            rec = np.concatenate([hdr, np.ascontiguousarray(data[a:b]).astype(order + "f4").view(np.uint8)], axis=1)
            rec.tofile(f)
            report(b, n_cdp, "Writing the full stack to SEG-Y")
    os.replace(tmp, out)
    return str(out), out.stat().st_size / 1e6
