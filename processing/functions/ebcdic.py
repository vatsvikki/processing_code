"""EBCDIC / binary header -> plain table rows (no UI)."""
from __future__ import annotations

import numpy as np

from .segy_io import BINARY_HEADER_FIELDS, FORMAT_NAMES, TRACE_HEADER_BYTES, SegyFile

# The whole 240-byte trace header: (1-based byte, name, code, description).  code = numpy int code, or "xN" = N raw
# bytes shown as hex.  Names of the words the QC uses match segy_io.TRACE_HEADER_FIELDS.  Bytes 173 and 181 follow
# this survey's EBCDIC header (RECLN / RECSTN, 4 bytes each) instead of SEG-Y rev1's two 2-byte words at 173-176.
TRACE_HEADER_FULL = [
    (1, "trace_seq_line", "i4", "Trace sequence number within line"),
    (5, "trace_seq_file", "i4", "Trace sequence number within SEG-Y file"),
    (9, "ffid", "i4", "Original field record number (FFID)"),
    (13, "chan", "i4", "Trace number within the original field record"),
    (17, "energy_src_pt", "i4", "Energy source point number"),
    (21, "cdp", "i4", "Ensemble number (CDP, CMP ...)"),
    (25, "cdp_trace", "i4", "Trace number within the ensemble"),
    (29, "trace_id", "i2", "Trace identification code (1 = seismic data)"),
    (31, "n_vert_summed", "i2", "Number of vertically summed traces"),
    (33, "n_horz_stacked", "i2", "Number of horizontally stacked traces"),
    (35, "data_use", "i2", "Data use (1 = production, 2 = test)"),
    (37, "offset", "i4", "Distance from source point to receiver group"),
    (41, "rec_elev", "i4", "Receiver group elevation"),
    (45, "src_elev", "i4", "Surface elevation at source"),
    (49, "src_depth", "i4", "Source depth below surface"),
    (53, "datum_elev_rec", "i4", "Datum elevation at receiver group"),
    (57, "datum_elev_src", "i4", "Datum elevation at source"),
    (61, "water_depth_src", "i4", "Water depth at source"),
    (65, "water_depth_rec", "i4", "Water depth at group"),
    (69, "elev_scalar", "i2", "Scalar for all elevations and depths (bytes 41-68)"),
    (71, "coord_scalar", "i2", "Scalar for all coordinates (bytes 73-88 and 181-188)"),
    (73, "src_x", "i4", "Source coordinate X"),
    (77, "src_y", "i4", "Source coordinate Y"),
    (81, "rec_x", "i4", "Group coordinate X"),
    (85, "rec_y", "i4", "Group coordinate Y"),
    (89, "coord_units", "i2", "Coordinate units (1 length, 2 arc seconds, 3 decimal degrees, 4 DMS)"),
    (91, "weathering_vel", "i2", "Weathering velocity"),
    (93, "subweathering_vel", "i2", "Subweathering velocity"),
    (95, "uphole_src", "i2", "Uphole time at source (ms)"),
    (97, "uphole_rec", "i2", "Uphole time at group (ms)"),
    (99, "src_static", "i2", "Source static correction (ms)"),
    (101, "rec_static", "i2", "Group static correction (ms)"),
    (103, "total_static", "i2", "Total static applied (ms)"),
    (105, "lag_a", "i2", "Lag time A: end of header to time break (ms)"),
    (107, "lag_b", "i2", "Lag time B: time break to source firing (ms)"),
    (109, "delay_rec", "i2", "Delay recording time: source firing to start of recording (ms)"),
    (111, "mute_start", "i2", "Mute time - start (ms)"),
    (113, "mute_end", "i2", "Mute time - end (ms)"),
    (115, "ns", "i2", "Number of samples in this trace"),
    (117, "dt", "i2", "Sample interval of this trace (microseconds)"),
    (119, "gain_type", "i2", "Gain type of field instruments (1 fixed, 2 binary, 3 floating point)"),
    (121, "gain_const", "i2", "Instrument gain constant (dB)"),
    (123, "initial_gain", "i2", "Instrument early or initial gain (dB)"),
    (125, "correlated", "i2", "Correlated (1 = no, 2 = yes)"),
    (127, "sweep_start", "i2", "Sweep frequency at start (Hz)"),
    (129, "sweep_end", "i2", "Sweep frequency at end (Hz)"),
    (131, "sweep_len", "i2", "Sweep length (ms)"),
    (133, "sweep_type", "i2", "Sweep type (1 linear, 2 parabolic, 3 exponential, 4 other)"),
    (135, "sweep_taper_start", "i2", "Sweep trace taper length at start (ms)"),
    (137, "sweep_taper_end", "i2", "Sweep trace taper length at end (ms)"),
    (139, "taper_type", "i2", "Taper type (1 linear, 2 cos^2, 3 other)"),
    (141, "alias_freq", "i2", "Alias filter frequency (Hz)"),
    (143, "alias_slope", "i2", "Alias filter slope (dB/octave)"),
    (145, "notch_freq", "i2", "Notch filter frequency (Hz)"),
    (147, "notch_slope", "i2", "Notch filter slope (dB/octave)"),
    (149, "low_cut_freq", "i2", "Low-cut frequency (Hz)"),
    (151, "high_cut_freq", "i2", "High-cut frequency (Hz)"),
    (153, "low_cut_slope", "i2", "Low-cut slope (dB/octave)"),
    (155, "high_cut_slope", "i2", "High-cut slope (dB/octave)"),
    (157, "year", "i2", "Year data recorded"),
    (159, "day", "i2", "Day of year"),
    (161, "hour", "i2", "Hour of day (24 hour clock)"),
    (163, "minute", "i2", "Minute of hour"),
    (165, "second", "i2", "Second of minute"),
    (167, "time_basis", "i2", "Time basis code (1 local, 2 GMT, 3 other, 4 UTC)"),
    (169, "trace_weight", "i2", "Trace weighting factor (2^-N volts for the least significant bit)"),
    (171, "roll_switch_pos", "i2", "Geophone group number of roll switch position one"),
    (173, "rec_line", "i4", "Receiver line number (RECLN; rev1: group first / last trace of the field record)"),
    (177, "gap_size", "i2", "Gap size (total number of groups dropped)"),
    (179, "taper_overtravel", "i2", "Over-travel associated with taper at beginning or end of line"),
    (181, "rec_stn", "i4", "Receiver station number (RECSTN; rev1: X of ensemble position)"),
    (185, "ens_y", "i4", "Y coordinate of ensemble position"),
    (189, "inline_no", "i4", "In-line number (3-D)"),
    (193, "xline_no", "i4", "Cross-line number (3-D)"),
    (197, "shotpoint", "i4", "Shotpoint number"),
    (201, "shotpoint_scalar", "i2", "Scalar for shotpoint number (bytes 197-200)"),
    (203, "value_unit", "i2", "Trace value measurement unit"),
    (205, "transduction_mant", "i4", "Transduction constant - mantissa"),
    (209, "transduction_exp", "i2", "Transduction constant - power of ten exponent"),
    (211, "transduction_unit", "i2", "Transduction units"),
    (213, "device_id", "i2", "Device / trace identifier"),
    (215, "time_scalar", "i2", "Scalar for times in bytes 95-114"),
    (217, "src_orientation", "i2", "Source type / orientation"),
    (219, "src_direction", "x6", "Source energy direction (6 bytes)"),
    (225, "src_measure_mant", "i4", "Source measurement - mantissa"),
    (229, "src_measure_exp", "i2", "Source measurement - power of ten exponent"),
    (231, "src_measure_unit", "i2", "Source measurement unit"),
    (233, "unassigned", "x8", "Unassigned (8 bytes)"),
]


def _field_size(code: str) -> int:
    return int(code[1:]) if code.startswith("x") else np.dtype(code).itemsize


def decode_trace_header(raw, order: str = ">") -> list[dict]:
    """All fields of ONE 240-byte trace header -> [{'Bytes': '9-12', 'Field': 'ffid', 'Value': '398', 'Description': ...}]."""
    buf = bytes(np.asarray(raw, np.uint8).tobytes())
    rows = []
    for byte1, name, code, desc in TRACE_HEADER_FULL:
        size = _field_size(code)
        chunk = buf[byte1 - 1: byte1 - 1 + size]
        val = chunk.hex(" ") if code.startswith("x") else str(int(np.frombuffer(chunk, dtype=order + code)[0]))
        rows.append({"Bytes": f"{byte1}-{byte1 + size - 1}", "Field": name, "Value": val, "Description": desc})
    return rows


def trace_header_stats(raw: np.ndarray, order: str = ">") -> list[dict]:
    """[ntr, 240] raw headers -> per numeric field: min, max and number of distinct values over the traces."""
    raw = np.asarray(raw, np.uint8)
    rows = []
    for byte1, name, code, desc in TRACE_HEADER_FULL:
        if code.startswith("x"):
            continue
        size = _field_size(code)
        v = np.ascontiguousarray(raw[:, byte1 - 1: byte1 - 1 + size]).view(order + code).ravel()
        n_distinct = int(len(np.unique(v)))
        rows.append({
            "Bytes": f"{byte1}-{byte1 + size - 1}", "Field": name, "Min": str(int(v.min())), "Max": str(int(v.max())),
            "Distinct": str(n_distinct), "Varies": "yes" if n_distinct > 1 else "constant", "Description": desc,
        })
    return rows


def unassigned_binary_row(sgy: SegyFile) -> dict:
    """Bytes 3261-3500 of the binary header (unassigned in SEG-Y rev1): 'all zero' or their hex."""
    with open(sgy.path, "rb") as f:
        f.seek(3260)
        raw = f.read(240)
    shown = "all zero" if not any(raw) else raw.hex(" ")
    return {"Byte": "3261", "Field": "(unassigned)", "Value": shown, "Description": "Unassigned bytes 3261-3500"}


def ebcdic_rows(lines: list[str]) -> list[dict]:
    """40 card images -> [{'Line': '01', 'Text': '...'}, ...] (trailing blanks stripped)."""
    return [{"Line": f"{i + 1:02d}", "Text": ln.rstrip()} for i, ln in enumerate(lines)]


def binary_header_rows(sgy: SegyFile) -> list[dict]:
    rows = []
    for byte1, name, _code, desc in BINARY_HEADER_FIELDS:
        v = sgy.binary_header[name]
        shown = f"{v}  ({FORMAT_NAMES.get(v, 'unknown')})" if name == "format" else str(v)
        rows.append({"Byte": str(byte1), "Field": name, "Value": shown, "Description": desc})
    return rows


def file_summary(sgy: SegyFile) -> dict:
    return {
        "file": sgy.path,
        "size_gb": sgy.file_size / 1e9,
        "traces": sgy.ntraces,
        "samples": sgy.ns,
        "dt_ms": sgy.dt_ms,
        "record_ms": sgy.ns * sgy.dt_ms,
        "format": f"{sgy.fmt} - {FORMAT_NAMES.get(sgy.fmt, 'unknown')}",
        "byte_order": "big-endian" if sgy.order == ">" else "little-endian",
        "first_ffid": sgy.first_ffid,
        "last_ffid": sgy.last_ffid,
        "ext_headers": len(sgy.ext_headers),
    }
