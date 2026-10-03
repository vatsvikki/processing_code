"""Write a processed shot gather back to a SEG-Y file (no UI).

The output keeps everything of the source file except the samples: the EBCDIC text header (a
processing note goes into its blank last lines), the binary header, and the full 240-byte trace
header of every trace that is still in the gather (removed traces are simply not written).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import segy_io
from .segy_io import TRACE_HEADER_FIELDS, ShotGather, TEXT_HEADER_BYTES, TRACE_HEADER_BYTES

FORMATS = ("ibm", "ieee")            # sample format of the output: 4-byte IBM float (as most field files) / IEEE float
_FORMAT_CODE = {"ibm": 1, "ieee": 5}


@dataclass
class ProcessedShot:
    """A shot gather ready to be saved, with what is needed to find its trace headers in the source file."""
    path: str                        # source SEG-Y file
    gather: ShotGather               # processed data (float32 [ntr, ns])
    src: np.ndarray                  # index within the source shot of every trace of `gather`
    label: str = "processed"         # e.g. "3. Spiking Decon"
    notes: list[str] = field(default_factory=list)   # one line per processing step, for the text header
    flow: list[dict] = field(default_factory=list)   # the steps that produced it, to repeat them on the whole file


def ieee2ibm(x: np.ndarray) -> np.ndarray:
    """IEEE float array -> IBM System/360 float words (uint32, native order)."""
    a = np.asarray(x, np.float64)
    out = np.zeros(a.shape, np.uint32)
    nz = np.isfinite(a) & (a != 0)
    m, e = np.frexp(np.abs(a[nz]))                    # |a| = m * 2**e, 0.5 <= m < 1
    ex = -(-e // 4)                                   # ceil(e / 4): IBM exponent, base 16
    frac = np.rint(m * np.exp2(e - 4 * ex) * 16777216.0).astype(np.int64)
    over = frac >= 16777216                           # rounding carried into a 25th bit
    frac[over] >>= 4
    ex[over] += 1
    ex = np.clip(ex + 64, 0, 127)
    out[nz] = ((a[nz] < 0).astype(np.uint32) << 31) | (ex.astype(np.uint32) << 24) | frac.astype(np.uint32)
    return out


def _empty(card: bytes) -> bool:
    """True for a blank text-header card, or one that holds only its "C37" label."""
    return re.fullmatch(r"\s*(C\s*\d{1,2})?\s*", card.decode("cp037", errors="replace").replace("\x00", " ")) is not None


def build_head(sgy: segy_io.SegyFile, lines: list[str], fmt: str) -> bytearray:
    """The source file's text + binary (+ extended text) headers, verbatim, except that

    * `lines` are written into the blank cards at the end of the EBCDIC header (never over existing text)
    * the sample-format code of the binary header is set for `fmt`
    """
    with open(sgy.path, "rb") as f:
        head = bytearray(f.read(sgy.data_start))
    cards = [bytes(head[i:i + 80]) for i in range(0, TEXT_HEADER_BYTES, 80)]
    blank = 0
    while blank < 4 and _empty(cards[39 - blank]):
        blank += 1
    n = min(blank, len(lines))
    for j, text in enumerate(lines[:n]):
        start = (40 - n + j) * 80
        card = f"C{41 - n + j:02d} {text}"[:80].ljust(80).encode("cp037", errors="replace")
        head[start:start + 80] = card
    head[3224:3226] = int(_FORMAT_CODE[fmt]).to_bytes(2, "big" if sgy.order == ">" else "little")
    return head


def encode_samples(data: np.ndarray, fmt: str, order: str) -> np.ndarray:
    """float32 samples -> IBM words / IEEE floats in the file's byte order."""
    if fmt == "ibm":
        return ieee2ibm(data).astype(order + "u4")
    return np.asarray(data, np.float32).astype(order + "f4")


def pack_traces(raw_headers: np.ndarray, headers: dict, data: np.ndarray, fmt: str, order: str) -> np.ndarray:
    """Original 240-byte trace headers, patched with any of the named header words a step changed in memory
    (unchanged fields round-trip byte for byte), + encoded samples ->
    uint8 [ntr, 240 + 4 * ns], ready to write.  `headers` must be in the same (output) trace order as `raw_headers`
    and `data` - e.g. the gather's own .headers, carried through the flow's steps."""
    n, ns = data.shape
    out = np.empty((n, TRACE_HEADER_BYTES + 4 * ns), np.uint8)
    hdr = np.array(raw_headers[:, :TRACE_HEADER_BYTES], copy=True)
    for _key, (_byte1, _code) in TRACE_HEADER_FIELDS.items():
        segy_io.set_header_word(hdr, _byte1, headers[_key], _code, order)
    out[:, :TRACE_HEADER_BYTES] = hdr
    out[:, TRACE_HEADER_BYTES:] = np.ascontiguousarray(encode_samples(data, fmt, order)).view(np.uint8).reshape(n, 4 * ns)
    return out


def write_shot_segy(shot: ProcessedShot, out_path: str, fmt: str = "ibm") -> str:
    """Write `shot` as a SEG-Y file (one shot gather); returns the path."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown sample format {fmt!r}; use one of {FORMATS}")
    g = shot.gather
    sgy = segy_io.open_segy(shot.path)
    idx = segy_io.shot_index(shot.path)
    k = idx.position(g.ffid)
    if k is None:
        raise LookupError(f"FFID {g.ffid} is not in {shot.path}")
    src = np.asarray(shot.src, np.int64)
    n_src = int(idx.count[k])
    if len(src) != g.ntr or (len(src) and (src.min() < 0 or src.max() >= n_src)):
        raise ValueError("trace index list does not match the gather")
    raw_headers, _, _ = sgy.read_block(int(idx.first[k]), int(idx.first[k]) + n_src)

    head = build_head(sgy, [f"PROCESSED SHOT FFID {g.ffid}: {shot.label.upper()}"] + [n.upper() for n in shot.notes], fmt)
    out = pack_traces(raw_headers[src], g.headers, g.data, fmt, sgy.order)
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(head)
        out.tofile(f)
    return out_path
