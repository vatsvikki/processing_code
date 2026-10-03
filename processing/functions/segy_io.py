"""SEG-Y reading utilities (no UI, no plotting).

Only the pieces needed for shot-gather QC are implemented:
  * EBCDIC textual header (+ any extended textual headers)
  * 400-byte binary header
  * random access to traces (a whole shot is read with ONE contiguous read)
  * binary search for a shot by FFID -- the file is "sorted in field record
    order" (per its EBCDIC header), so a 15 GB file needs ~60 seeks instead of
    a scan of ~950,000 trace headers.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

TEXT_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
TRACE_HEADER_BYTES = 240

# Trace header words used by QC: name -> (1-based start byte, numpy int code).
# Bytes follow this survey's EBCDIC header (FFID 9, OFFSET 37, RECLN 173 ...)
# which agree with the SEG-Y rev1 standard for the words used here.
TRACE_HEADER_FIELDS: dict[str, tuple[int, str]] = {
    "trace_seq_file": (5, "i4"),
    "ffid": (9, "i4"),
    "chan": (13, "i4"),        # trace number within the field record
    "cdp": (21, "i4"),         # ensemble (CDP) number
    "offset": (37, "i4"),
    "rec_elev": (41, "i4"),
    "src_x": (73, "i4"),
    "src_y": (77, "i4"),
    "rec_x": (81, "i4"),
    "rec_y": (85, "i4"),
    "ns": (115, "i2"),
    "dt": (117, "i2"),
    "rec_line": (173, "i4"),   # RECLN  (custom, per EBCDIC)
    "rec_stn": (181, "i4"),    # RECSTN (custom, per EBCDIC)
}

# Binary header words: (1-based byte in the 400-byte block relative to file
# byte 3201, name, numpy int code, description).
BINARY_HEADER_FIELDS = [
    (3201, "job_id", "i4", "Job identification number"),
    (3205, "line_no", "i4", "Line number"),
    (3209, "reel_no", "i4", "Reel number"),
    (3213, "traces_per_ensemble", "i2", "Number of data traces per ensemble"),
    (3215, "aux_per_ensemble", "i2", "Number of auxiliary traces per ensemble"),
    (3217, "dt_us", "i2", "Sample interval (microseconds)"),
    (3219, "dt_us_orig", "i2", "Sample interval of original field recording (us)"),
    (3221, "ns", "i2", "Number of samples per data trace"),
    (3223, "ns_orig", "i2", "Number of samples per trace, original field recording"),
    (3225, "format", "i2", "Data sample format code"),
    (3227, "ensemble_fold", "i2", "Ensemble fold"),
    (3229, "trace_sorting", "i2", "Trace sorting code"),
    (3231, "vertical_sum", "i2", "Vertical sum code"),
    (3233, "sweep_start_hz", "i2", "Sweep frequency at start (Hz)"),
    (3235, "sweep_end_hz", "i2", "Sweep frequency at end (Hz)"),
    (3237, "sweep_len_ms", "i2", "Sweep length (ms)"),
    (3239, "sweep_type", "i2", "Sweep type code"),
    (3241, "sweep_chan", "i2", "Trace number of sweep channel"),
    (3243, "sweep_taper_start_ms", "i2", "Sweep trace taper length at start (ms)"),
    (3245, "sweep_taper_end_ms", "i2", "Sweep trace taper length at end (ms)"),
    (3247, "taper_type", "i2", "Taper type"),
    (3249, "correlated", "i2", "Correlated data traces"),
    (3251, "binary_gain", "i2", "Binary gain recovered"),
    (3253, "amp_recovery", "i2", "Amplitude recovery method"),
    (3255, "measurement_system", "i2", "Measurement system (1=m, 2=ft)"),
    (3257, "impulse_polarity", "i2", "Impulse signal polarity"),
    (3259, "vibratory_polarity", "i2", "Vibratory polarity code"),
    (3501, "segy_revision", "u2", "SEG-Y format revision number"),
    (3503, "fixed_length_flag", "i2", "Fixed length trace flag"),
    (3505, "n_ext_text_headers", "i2", "Number of extended textual headers"),
]

FORMAT_NAMES = {
    1: "4-byte IBM floating point",
    2: "4-byte two's complement integer",
    3: "2-byte two's complement integer",
    5: "4-byte IEEE floating point",
    8: "1-byte two's complement integer",
}
_FORMAT_DTYPE = {1: "u4", 2: "i4", 3: "i2", 5: "f4", 8: "i1"}


# --------------------------------------------------------------------------
# sample conversion
# --------------------------------------------------------------------------
def ibm2ieee(raw_u4: np.ndarray) -> np.ndarray:
    """Vectorised IBM System/360 float -> IEEE float32 (input: native uint32)."""
    a = raw_u4.astype(np.uint32, copy=False)
    sign = (a >> 31).astype(bool)
    exp = ((a >> 24) & 0x7F).astype(np.int32)
    frac = (a & 0x00FFFFFF).astype(np.float32) / np.float32(16777216.0)  # exact
    out = np.ldexp(frac, 4 * (exp - 64))
    out[sign] *= -1.0
    return out.astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# file object
# --------------------------------------------------------------------------
@dataclass
class SegyFile:
    path: str
    file_size: int
    order: str                       # '>' big-endian (standard) or '<'
    ns: int
    dt_us: int
    fmt: int
    data_start: int                  # byte where trace 0 starts
    trace_bytes: int
    ntraces: int
    ebcdic: list[str]                # 40 lines of 80 chars
    ext_headers: list[list[str]] = field(default_factory=list)
    binary_header: dict = field(default_factory=dict)

    @property
    def dt_ms(self) -> float:
        return self.dt_us / 1000.0

    # -- dtype of one trace record: named header words + samples ----------
    def _record_dtype(self) -> np.dtype:
        names = list(TRACE_HEADER_FIELDS)
        hdr = np.dtype({
            "names": names,
            "formats": [self.order + TRACE_HEADER_FIELDS[n][1] for n in names],
            "offsets": [TRACE_HEADER_FIELDS[n][0] - 1 for n in names],
            "itemsize": TRACE_HEADER_BYTES,
        })
        return np.dtype([("h", hdr), ("d", self.order + _FORMAT_DTYPE[self.fmt], self.ns)])

    def read_block(self, i0: int, i1: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """Traces i0..i1-1 (0-based) in ONE read -> (raw 240-byte trace headers [ntr, 240] uint8,
        parsed header words {name: int64 [ntr]}, float32 samples [ntr, ns])."""
        i0, i1 = max(0, i0), min(self.ntraces, i1)
        n = i1 - i0
        if n <= 0:
            raise ValueError("empty trace range")
        with open(self.path, "rb") as f:
            f.seek(self.data_start + i0 * self.trace_bytes)
            buf = np.fromfile(f, dtype=np.uint8, count=n * self.trace_bytes)
        if len(buf) != n * self.trace_bytes:
            raise IOError(f"short read: wanted {n} traces, got {len(buf) // self.trace_bytes}")
        arr = buf.view(self._record_dtype())
        hdr = {k: arr["h"][k].astype(np.int64) for k in TRACE_HEADER_FIELDS}
        raw = arr["d"]
        if self.fmt == 1:
            data = ibm2ieee(raw.astype(np.uint32))
        else:
            data = raw.astype(np.float32)
        return buf.reshape(n, self.trace_bytes)[:, :TRACE_HEADER_BYTES].copy(), hdr, data

    def read_traces_at(self, indices, chunk: int = 1500) -> tuple[np.ndarray, np.ndarray]:
        """Traces at the given indices (0-based, any order, e.g. the traces of one CMP bin, which come from many shots)
        -> (raw 240-byte headers uint8 [n, 240], float32 samples [n, ns]), rows in the order given.
        The file is read in ascending trace order, `chunk` traces at a time."""
        indices = np.asarray(indices, np.int64)
        n = len(indices)
        raw = np.empty((n, TRACE_HEADER_BYTES), np.uint8)
        data = np.empty((n, self.ns), np.float32)
        if n == 0:
            return raw, data
        order = np.argsort(indices, kind="stable")
        mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_start, shape=(self.ntraces, self.trace_bytes))
        for k in range(0, n, chunk):
            rows = order[k:k + chunk]
            blk = np.ascontiguousarray(mm[indices[rows]])                  # [m, trace_bytes]
            rec = blk.reshape(-1).view(self._record_dtype())
            raw[rows] = blk[:, :TRACE_HEADER_BYTES]
            samples = rec["d"]
            data[rows] = ibm2ieee(samples.astype(np.uint32)) if self.fmt == 1 else samples.astype(np.float32)
        del mm
        return raw, data

    def read_traces(self, i0: int, i1: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Traces i0..i1-1 (0-based) -> (header dict, float32 array [ntr, ns])."""
        _, hdr, data = self.read_block(i0, i1)
        return hdr, data

    def read_headers(self, i0: int, i1: int) -> np.ndarray:
        """Raw 240-byte trace headers of traces i0..i1-1 (0-based) -> uint8 [n, 240].
        Memory-mapped, so only the header pages are read, not the samples."""
        i0, i1 = max(0, i0), min(self.ntraces, i1)
        if i1 <= i0:
            return np.zeros((0, TRACE_HEADER_BYTES), np.uint8)
        mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_start, shape=(self.ntraces, self.trace_bytes))
        out = np.array(mm[i0:i1, :TRACE_HEADER_BYTES])
        del mm
        return out

    def read_headers_at(self, indices) -> np.ndarray:
        """Raw trace headers of the given trace indices (0-based, any order) -> uint8 [len(indices), 240]."""
        indices = np.asarray(indices, np.int64)
        if len(indices) == 0:
            return np.zeros((0, TRACE_HEADER_BYTES), np.uint8)
        mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_start, shape=(self.ntraces, self.trace_bytes))
        out = np.array(mm[indices, :TRACE_HEADER_BYTES])
        del mm
        return out

    def read_word(self, idx: int, byte: int, code: str = "i4") -> int:
        """One header word of trace idx (cheap: 4-byte read)."""
        size = np.dtype(code).itemsize
        with open(self.path, "rb") as f:
            f.seek(self.data_start + idx * self.trace_bytes + byte - 1)
            return int(np.frombuffer(f.read(size), dtype=self.order + code)[0])

    def ffid_at(self, idx: int) -> int:
        return self.read_word(idx, TRACE_HEADER_FIELDS["ffid"][0])

    @property
    def first_ffid(self) -> int:
        return self.ffid_at(0)

    @property
    def last_ffid(self) -> int:
        return self.ffid_at(self.ntraces - 1)

    def nearest_ffids(self, ffid: int) -> tuple[int | None, int | None]:
        """FFIDs just below / above `ffid` (for 'not found' messages)."""
        lo, hi = 0, self.ntraces
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ffid_at(mid) < ffid:
                lo = mid + 1
            else:
                hi = mid
        below = self.ffid_at(lo - 1) if lo > 0 else None
        above = self.ffid_at(lo) if lo < self.ntraces else None
        return below, above

    def find_shot(self, ffid: int) -> tuple[int, int] | None:
        """[first, last+1) trace indices of the shot with this FFID, or None.

        Binary search; relies on traces being in field-record order.
        """
        lo, hi = 0, self.ntraces - 1
        hit = None
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self.ffid_at(mid)
            if v == ffid:
                hit = mid
                break
            if v < ffid:
                lo = mid + 1
            else:
                hi = mid - 1
        if hit is None:
            return None

        def edge(lo, hi, go_left):
            """lowest (go_left) / highest index in [lo, hi] whose ffid == target."""
            best = hit
            while lo <= hi:
                mid = (lo + hi) // 2
                if self.ffid_at(mid) == ffid:
                    best = mid
                    if go_left:
                        hi = mid - 1
                    else:
                        lo = mid + 1
                elif go_left:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best

        first = edge(0, hit, True)
        last = edge(hit, self.ntraces - 1, False)
        return first, last + 1


def header_word(raw: np.ndarray, byte1: int, code: str = "i4", order: str = ">") -> np.ndarray:
    """One header word (1-based start byte, numpy int code) of every row of raw [n, 240] headers -> int64 [n]."""
    size = np.dtype(code).itemsize
    return np.ascontiguousarray(raw[:, byte1 - 1: byte1 - 1 + size]).view(order + code).ravel().astype(np.int64)


def set_header_word(raw: np.ndarray, byte1: int, values: np.ndarray, code: str = "i4", order: str = ">") -> None:
    """Write `values` (int-like [n]) into one header word (1-based start byte) of every row of raw [n, 240], in
    place - the inverse of header_word(); used to carry a header a step changed in memory into the written bytes."""
    size = np.dtype(code).itemsize
    packed = np.ascontiguousarray(np.asarray(values)).astype(order + code)
    raw[:, byte1 - 1: byte1 - 1 + size] = packed.view(np.uint8).reshape(len(packed), size)


# --------------------------------------------------------------------------
# open / parse
# --------------------------------------------------------------------------
def decode_ebcdic(raw: bytes) -> list[str]:
    """3200 EBCDIC bytes -> 40 lines of 80 characters (ASCII files also handled)."""
    text = raw.decode("cp037", errors="replace")
    if sum(ch.isalpha() for ch in text) < sum(ch.isalpha() for ch in raw.decode("latin-1")) * 0.5:
        text = raw.decode("latin-1")     # header was already ASCII
    return [text[i:i + 80] for i in range(0, len(raw), 80)]


def _unpack(buf: bytes, byte1: int, code: str, order: str):
    """byte1 = 1-based file byte offset of the field; buf starts at file byte 3201."""
    off = byte1 - 3201
    size = np.dtype(code).itemsize
    return int(np.frombuffer(buf[off:off + size], dtype=order + code)[0])


@lru_cache(maxsize=4)
def _open_cached(path: str, mtime: float) -> SegyFile:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        text = f.read(TEXT_HEADER_BYTES)
        binh = f.read(BINARY_HEADER_BYTES)
        if len(binh) < BINARY_HEADER_BYTES:
            raise ValueError("file is too small to be SEG-Y")

        # byte order: the format code must be 1..8 and ns sane
        order = None
        for cand in (">", "<"):
            fmt = _unpack(binh, 3225, "u2", cand)
            ns = _unpack(binh, 3221, "u2", cand)
            if fmt in _FORMAT_DTYPE and 0 < ns < 65535:
                order = cand
                break
        if order is None:
            raise ValueError("not a readable SEG-Y file: the binary header has no valid data-format code / sample count")

        n_ext = _unpack(binh, 3505, "i2", order)
        ext: list[list[str]] = []
        if n_ext > 0:
            for _ in range(n_ext):
                ext.append(decode_ebcdic(f.read(TEXT_HEADER_BYTES)))
        elif n_ext == -1:                       # variable: read until EndText
            for _ in range(100):
                blk = f.read(TEXT_HEADER_BYTES)
                if len(blk) < TEXT_HEADER_BYTES:
                    break
                lines = decode_ebcdic(blk)
                ext.append(lines)
                if any("EndText" in ln for ln in lines):
                    break
        data_start = TEXT_HEADER_BYTES + BINARY_HEADER_BYTES + TEXT_HEADER_BYTES * len(ext)

        ns = _unpack(binh, 3221, "u2", order)
        dt = _unpack(binh, 3217, "u2", order)
        fmt = _unpack(binh, 3225, "u2", order)
        if ns == 0:                              # fall back to first trace header
            f.seek(data_start + 114)
            ns = int(np.frombuffer(f.read(2), dtype=order + "u2")[0])
        if dt == 0:
            f.seek(data_start + 116)
            dt = int(np.frombuffer(f.read(2), dtype=order + "u2")[0])

    bps = np.dtype(_FORMAT_DTYPE[fmt]).itemsize
    trace_bytes = TRACE_HEADER_BYTES + ns * bps
    ntr, rem = divmod(size - data_start, trace_bytes)

    bh = {}
    for byte1, name, code, desc in BINARY_HEADER_FIELDS:
        bh[name] = _unpack(binh, byte1, code, order)

    return SegyFile(
        path=path, file_size=size, order=order, ns=ns, dt_us=dt, fmt=fmt,
        data_start=data_start, trace_bytes=trace_bytes, ntraces=ntr,
        ebcdic=decode_ebcdic(text), ext_headers=ext, binary_header=bh,
    )


def open_segy(path: str) -> SegyFile:
    path = os.path.expanduser(str(path).strip())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")
    return _open_cached(path, os.path.getmtime(path))


# --------------------------------------------------------------------------
# shot index: every FFID in the file with its first trace and trace count
# --------------------------------------------------------------------------
@dataclass
class ShotIndex:
    ffids: np.ndarray                # int64, ascending, one entry per shot
    first: np.ndarray                # index of each shot's first trace in the file
    count: np.ndarray                # traces in each shot

    def __len__(self) -> int:
        return len(self.ffids)

    def position(self, ffid: int) -> int | None:
        k = int(np.searchsorted(self.ffids, ffid))
        return k if k < len(self.ffids) and self.ffids[k] == ffid else None

    def nearest(self, ffid: int) -> int:
        """The existing FFID closest to `ffid` (ffid <= 0 -> the first shot)."""
        k = int(np.searchsorted(self.ffids, ffid))
        cand = [c for c in (k - 1, k) if 0 <= c < len(self.ffids)]
        return int(min((self.ffids[c] for c in cand), key=lambda v: abs(int(v) - ffid)))

    def ntr(self, ffid: int) -> int:
        k = self.position(self.nearest(ffid))
        return int(self.count[k])


def _build_shot_index(path: str, mtime: float, progress=None) -> ShotIndex:
    """Walk the file shot by shot: read the FFID of a trace, gallop + bisect to the
    last trace of that shot, jump past it.  ~20 four-byte reads per shot, so the
    index of a 15 GB / 950 000-trace file takes about a second.

    progress(fraction 0..1, text) is called about once per 16 shots (used by the Load Data tool)."""
    sgy = _open_cached(path, mtime)
    n, ffid_byte = sgy.ntraces, TRACE_HEADER_FIELDS["ffid"][0]
    dt = np.dtype(sgy.order + "i4")
    ffids, first, count = [], [], []
    with open(path, "rb") as f:
        def ffid_at(i: int) -> int:
            f.seek(sgy.data_start + i * sgy.trace_bytes + ffid_byte - 1)
            return int(np.frombuffer(f.read(4), dtype=dt)[0])

        i = 0
        while i < n:
            v = ffid_at(i)
            step = 1
            while i + step < n and ffid_at(i + step) == v:      # gallop
                step *= 2
            lo, hi = i + step // 2, min(i + step, n)             # ffid(lo)==v, ffid(hi)!=v (or end)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ffid_at(mid) == v:
                    lo = mid
                else:
                    hi = mid
            ffids.append(v)
            first.append(i)
            count.append(lo + 1 - i)
            i = lo + 1
            if progress is not None and len(ffids) % 16 == 0:
                progress(i / n, f"Indexing shots: {len(ffids):,} found, trace {i:,} of {n:,}")
    idx = ShotIndex(np.array(ffids, np.int64), np.array(first, np.int64), np.array(count, np.int64))
    if len(idx) > 1 and not (np.diff(idx.ffids) > 0).all():
        raise ValueError("FFIDs are not in ascending field-record order, so a shot index cannot be built")
    return idx


_INDEX_CACHE: dict[tuple[str, float], ShotIndex] = {}      # (path, mtime) -> index; the 4 newest are kept


def shot_index(path: str, progress=None, force: bool = False) -> ShotIndex:
    """The shot index of the file (cached).  force=True rebuilds it from disk; `progress` is only called
    when it is really built (see _build_shot_index)."""
    path = os.path.expanduser(str(path).strip())
    open_segy(path)                                   # validates the path
    key = (path, os.path.getmtime(path))
    if force or key not in _INDEX_CACHE:
        idx = _build_shot_index(path, key[1], progress)
        _INDEX_CACHE.pop(key, None)
        _INDEX_CACHE[key] = idx
        while len(_INDEX_CACHE) > 4:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    return _INDEX_CACHE[key]


def norm_path(path: str) -> str:
    """Absolute path with ~ expanded and surrounding blanks removed (how a typed path is compared)."""
    return os.path.abspath(os.path.expanduser(str(path).strip()))


def clear_caches() -> None:
    """Forget everything read from disk (headers, shot indexes, shots) so the next call reads the file again."""
    _open_cached.cache_clear()
    _read_shot_cached.cache_clear()
    _INDEX_CACHE.clear()


# --------------------------------------------------------------------------
# shot gather
# --------------------------------------------------------------------------
@dataclass
class ShotGather:
    ffid: int
    i0: int                          # first trace index in file
    data: np.ndarray                 # float32 [ntr, ns], file order
    headers: dict[str, np.ndarray]
    dt_ms: float

    @property
    def ntr(self) -> int:
        return self.data.shape[0]

    @property
    def ns(self) -> int:
        return self.data.shape[1]

    @property
    def time_ms(self) -> np.ndarray:
        return np.arange(self.ns) * self.dt_ms


@lru_cache(maxsize=3)
def _read_shot_cached(path: str, mtime: float, ffid: int) -> ShotGather:
    sgy = _open_cached(path, mtime)
    if ffid <= 0:
        ffid = sgy.first_ffid
    rng = sgy.find_shot(ffid)
    if rng is None:
        below, above = sgy.nearest_ffids(ffid)
        near = ", ".join(str(x) for x in (below, above) if x is not None)
        raise LookupError(
            f"FFID {ffid} not found in this file (FFIDs run {sgy.first_ffid} to {sgy.last_ffid}; "
            f"nearest existing shots: {near})."
        )
    hdr, data = sgy.read_traces(*rng)
    return ShotGather(ffid=ffid, i0=rng[0], data=data, headers=hdr, dt_ms=sgy.dt_ms)


def read_shot(path: str, ffid: int = 0) -> ShotGather:
    """Read one shot gather. ffid <= 0 means 'first shot in the file'.

    The result is cached (last 3 shots) so re-running QC with new thresholds is
    instant -- treat the returned arrays as read-only.
    """
    path = os.path.expanduser(str(path).strip())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")
    return _read_shot_cached(path, os.path.getmtime(path), int(ffid))


def trace_order(gather: ShotGather, sort_by: str = "offset") -> np.ndarray:
    """Permutation that sorts the gather for display / neighbour statistics."""
    h = gather.headers
    if sort_by == "offset":
        return np.argsort(np.abs(h["offset"]), kind="stable")
    if sort_by == "channel":
        return np.argsort(h["chan"], kind="stable")
    if sort_by == "receiver":
        return np.lexsort((h["rec_stn"], h["rec_line"]))
    if sort_by == "cdp":
        return np.argsort(h["cdp"], kind="stable")
    if sort_by == "cdp_offset":       # CDP first, within a CDP by absolute offset
        return np.lexsort((np.abs(h["offset"]), h["cdp"]))
    return np.arange(gather.ntr)      # "file"
