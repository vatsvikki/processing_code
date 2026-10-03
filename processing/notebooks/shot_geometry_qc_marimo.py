import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import struct
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from scipy.signal import hilbert

    _SC_SCALING_DIR = Path(__file__).resolve().parent.parent / "SC_scaling"
    if str(_SC_SCALING_DIR) not in sys.path:
        sys.path.insert(0, str(_SC_SCALING_DIR))
    from sc_scaling_pipeline import build_dtypes

    # Where per-shot QC images get saved (mo.image references a file on
    # disk instead of embedding the figure's data inline in the cell
    # output -- a Plotly go.Heatmap for ~700-900 traces x thousands of
    # samples was found to produce a 28MB+ inline JSON payload, well past
    # marimo's output-size limit; a saved PNG referenced by path has no
    # such limit regardless of how large the underlying array is). Plotly
    # is still fine for small scatter plots (a few thousand points, not a
    # full trace-sample heatmap) -- used below for the one map that needs
    # hover interactivity.
    qc_plot_dir = Path(__file__).resolve().parent / "qc_plots"
    qc_plot_dir.mkdir(parents=True, exist_ok=True)
    return Path, build_dtypes, go, hilbert, json, mo, np, plt, qc_plot_dir, struct


@app.cell
def _(mo):
    mo.md("""
    # Shot Geometry QC — Direct-Arrival Linear-Velocity Overlay

    Overlay a straight reference line `t = offset / V` (a direct/refracted arrival is linear with
    offset, unlike a reflection hyperbola) on a real shot record.
    If the line tracks the leading edge of the real data, the
    geometry is consistent. If it clearly diverges, that's
    evidence of a coordinate/offset error for this shot.

    Toggles to test directional fixes (negate X/Y,
    recomputed geometric offset vs. the stored header offset)
    """)
    return


@app.cell
def _(mo):
    # the file shown in the path box below. Opened from the Processing Tool's 🛠 Tools menu, the main window's file
    # comes in as ?segy=<path> (or, when the app embeds this notebook, through app.embed(defs={"default_segy_path": ...}),
    # which replaces this cell)
    default_segy_path = str(mo.query_params().get("segy") or "../../EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.sgy")
    return (default_segy_path,)


@app.cell
def _(default_segy_path, mo):
    su_path = mo.ui.text(
        label="Input .sgy file path",
        value=default_segy_path,
        full_width=True,
    )
    su_path
    return (su_path,)


@app.cell
def _(struct):
    # This tool works exclusively with SEG-Y (.sgy): every trace is
    # preceded by a fixed 3200-byte text + 400-byte binary reel header,
    # so every byte offset below is relative to the start of the DATA
    # (after that 3600-byte header), not the start of the file.
    REEL_BYTES = 3600
    HEADER_BYTES = 240
    FLDR_OFF, OFFSET_OFF = 8, 36
    SX_OFF, SY_OFF, GX_OFF, GY_OFF = 72, 76, 80, 84
    NS_OFF, DT_OFF = 114, 116
    # Custom (non-standard) fields this survey's own EBCDIC text header
    # documents: byte 173 = RECEIVER LINE NUMBER, byte 181 = RECEIVER
    # STATION NUMBER. Verified directly against the real data: RECLN
    # gives exactly 7 clean, evenly-spaced line groups for a sample shot,
    # matching this survey's documented 2100 ft receiver line interval --
    # authoritative, and simpler than inferring lines from GY gaps.
    RECLN_OFF, RECSTN_OFF = 172, 180

    def detect_endianness(path):
        with open(path, "rb") as f:
            f.seek(REEL_BYTES)
            buf = f.read(HEADER_BYTES)
        for order in ("<", ">"):
            ns = struct.unpack_from(order + "H", buf, NS_OFF)[0]
            if 1 <= ns <= 20000:
                return order
        raise ValueError("Could not determine byte order.")

    def get_fldr_at_index(f, idx, record_size, order):
        """Jump directly to trace idx and read just its FLDR field -- no header parse, no sample read."""
        f.seek(REEL_BYTES + idx * record_size + FLDR_OFF)
        return struct.unpack(order + "i", f.read(4))[0]

    def find_shot_range(path, order, record_size, n_traces, target_fldr):
        """
        Binary search for the first and last trace index matching
        target_fldr, relying on this survey's own EBCDIC header
        claim that traces are sorted in field-record order. This
        turns a lookup that could touch ~1,000,000 traces (a slow
        linear scan) into ~20 seeks total, regardless of file size
        or where in the file the target shot happens to sit.
        """
        import numpy as np
        with open(path, "rb") as f:
            # Find any index with fldr == target (standard binary search)
            lo, hi = 0, n_traces - 1
            found = None
            while lo <= hi:
                mid = (lo + hi) // 2
                v = get_fldr_at_index(f, mid, record_size, order)
                if v == target_fldr:
                    found = mid
                    break
                elif v < target_fldr:
                    lo = mid + 1
                else:
                    hi = mid - 1

            if found is None:
                return None

            # Expand outward to the full contiguous block (still O(log n)
            # via two more binary searches for the exact left/right edges)
            lo, hi = 0, found
            left = found
            while lo <= hi:
                mid = (lo + hi) // 2
                v = get_fldr_at_index(f, mid, record_size, order)
                if v == target_fldr:
                    left = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            lo, hi = found, n_traces - 1
            right = found
            while lo <= hi:
                mid = (lo + hi) // 2
                v = get_fldr_at_index(f, mid, record_size, order)
                if v == target_fldr:
                    right = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

        return left, right

    def read_shot(path, order, target_fldr, format_code=5):
        """
        Binary-search for the target shot's trace range, then read
        just that contiguous block directly -- no scanning past
        unrelated traces at all.
        """
        import os
        import numpy as np

        with open(path, "rb") as f:
            f.seek(REEL_BYTES)
            first_header = f.read(HEADER_BYTES)
            ns = struct.unpack_from(order + "H", first_header, NS_OFF)[0]

        record_size = HEADER_BYTES + ns * 4
        data_bytes = os.path.getsize(path) - REEL_BYTES
        if data_bytes % record_size != 0:
            raise ValueError(
                "File size (after the 3600-byte SEG-Y reel header) doesn't "
                "divide evenly by record size -- ns may not be constant "
                "across the file, so binary search (which assumes "
                "fixed-length records) isn't safe here."
            )
        n_traces = data_bytes // record_size

        shot_range = find_shot_range(path, order, record_size, n_traces, target_fldr)
        if shot_range is None:
            return None
        left, right = shot_range

        offsets, sx_list, sy_list, gx_list, gy_list, traces = [], [], [], [], [], []
        recln_list, recstn_list = [], []
        dt_val = None

        with open(path, "rb") as f:
            f.seek(REEL_BYTES + left * record_size)
            for _ in range(left, right + 1):
                header = f.read(HEADER_BYTES)
                this_ns = struct.unpack_from(order + "H", header, NS_OFF)[0]
                offset = struct.unpack_from(order + "i", header, OFFSET_OFF)[0]
                sx = struct.unpack_from(order + "i", header, SX_OFF)[0]
                sy = struct.unpack_from(order + "i", header, SY_OFF)[0]
                gx = struct.unpack_from(order + "i", header, GX_OFF)[0]
                gy = struct.unpack_from(order + "i", header, GY_OFF)[0]
                dt_us = struct.unpack_from(order + "H", header, DT_OFF)[0]
                recln = struct.unpack_from(order + "i", header, RECLN_OFF)[0]
                recstn = struct.unpack_from(order + "i", header, RECSTN_OFF)[0]

                raw = f.read(this_ns * 4)
                samples = decode_samples(raw, order, format_code)

                offsets.append(offset)
                sx_list.append(sx); sy_list.append(sy)
                gx_list.append(gx); gy_list.append(gy)
                recln_list.append(recln); recstn_list.append(recstn)
                traces.append(samples)
                dt_val = dt_us

        return {
            "offset": np.array(offsets, dtype=np.float64),
            "sx": np.array(sx_list, dtype=np.float64),
            "sy": np.array(sy_list, dtype=np.float64),
            "gx": np.array(gx_list, dtype=np.float64),
            "gy": np.array(gy_list, dtype=np.float64),
            "recln": np.array(recln_list, dtype=np.int64),
            "recstn": np.array(recstn_list, dtype=np.int64),
            "traces": np.column_stack(traces),
            "ns": ns,
            "dt_us": dt_val,
        }

    def probe_sgy(path):
        """
        Same contract as sc_scaling_pipeline.probe_file, but accounting
        for the 3600-byte SEG-Y reel header this tool now always expects
        in front of the trace data. Also reads the binary header's own
        declared sample format code (bytes 3225-3226, i.e. offset 24
        within the 400-byte binary header) -- this survey's real .sgy
        file declares format 1 (IBM floating point), NOT 5 (IEEE
        float), even though every dtype in this codebase historically
        assumed IEEE float for the .su version of this data. Verified
        directly: decoding this file's samples as IBM float reproduces
        the known-good .su file's IEEE-float sample values bit-for-bit;
        decoding as IEEE (the old assumption) produced a completely
        different, garbage waveform.
        """
        import os
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(REEL_BYTES)
            first = f.read(HEADER_BYTES)
            f.seek(REEL_BYTES - 400 + 24)
            _fmt_bytes = f.read(2)
        if len(first) < HEADER_BYTES:
            raise RuntimeError(f"File shorter than the 3600-byte reel header + one trace header: {path}")
        order = None
        for cand in ("<", ">"):
            _ns = struct.unpack_from(cand + "H", first, NS_OFF)[0]
            if 1 <= _ns <= 20000:
                order = cand
                break
        if order is None:
            raise RuntimeError(
                "Could not determine byte order (ns field unreasonable in "
                "both endiannesses) -- is this really a SEG-Y file?"
            )
        ns = struct.unpack_from(order + "H", first, NS_OFF)[0]
        dt = struct.unpack_from(order + "H", first, DT_OFF)[0]
        format_code = struct.unpack_from(order + "h", _fmt_bytes)[0]
        trace_bytes = HEADER_BYTES + ns * 4
        data_bytes = size - REEL_BYTES
        if trace_bytes <= 0 or data_bytes % trace_bytes != 0:
            raise RuntimeError(
                f"File size after the 3600-byte SEG-Y reel header "
                f"({data_bytes:,} bytes) is not an exact multiple of trace "
                f"size {trace_bytes:,} bytes (ns={ns})."
            )
        n_traces = data_bytes // trace_bytes
        return {
            "order": order, "ns": ns, "dt_us": dt, "trace_bytes": trace_bytes,
            "n_traces": n_traces, "file_size": size, "format_code": format_code,
        }

    def ibm_bits_to_float(bits):
        """
        Vectorized IBM System/360 floating point -> IEEE float32.
        `bits` must be a numpy uint32 array holding the raw 4-byte
        words in their natural (already byte-order-corrected) integer
        value -- i.e. read with the file's own endianness, not a plain
        host-order reinterpretation.
        """
        bits = bits.astype(np.int64)
        sign = np.where((bits >> 31) & 1, -1.0, 1.0)
        exponent = ((bits >> 24) & 0x7F).astype(np.float64)
        mantissa = (bits & 0x00FFFFFF).astype(np.float64)
        out = sign * (mantissa / 16777216.0) * np.power(16.0, exponent - 64.0)
        out = np.where(mantissa == 0, 0.0, out)
        return out.astype(np.float32)

    def decode_samples(raw_bytes, order, format_code):
        """
        Bytes -> float32 samples, honoring the SEG-Y binary header's
        own declared format code instead of assuming IEEE float.
        Format 1 = IBM float (this survey's actual .sgy), format 5 =
        IEEE float (the .su convention, and SEG-Y rev1's recommended
        format) -- anything else falls back to IEEE with the
        assumption unverified, since that's what every prior tool in
        this codebase already assumed.
        """
        if format_code == 1:
            _bits = np.frombuffer(raw_bytes, dtype=order + "u4")
            return ibm_bits_to_float(_bits)
        return np.frombuffer(raw_bytes, dtype=order + "f4").astype(np.float32)

    return (
        REEL_BYTES,
        decode_samples,
        detect_endianness,
        ibm_bits_to_float,
        probe_sgy,
        read_shot,
    )


@app.cell
def _(np):
    def build_header_scan_dtype(order, ns):
        """
        Extended 240-byte header dtype exposing this survey's custom
        fields (RECLN etc. at byte 168+, per its own EBCDIC text
        header) alongside the standard ones, with the sample block
        left as an opaque void field so numpy skips converting it --
        needed for a fast whole-survey header-only scan (~350K
        traces/s measured against the real 15GB file) rather than
        the slower full-sample read the FLDR-only scan above does.
        Field order/sizes verified against real trace 0 (fldr=398,
        recln=128, gy=431850, ns=4000, dt=2000 -- all matched known
        values from earlier single-shot lookups).
        """
        header_dtype = np.dtype([
            ('tracl', order + 'i4'), ('tracr', order + 'i4'), ('fldr', order + 'i4'),
            ('tracf', order + 'i4'), ('ep', order + 'i4'), ('cdp', order + 'i4'), ('cdpt', order + 'i4'),
            ('trid', order + 'i2'), ('nvs', order + 'i2'), ('nhs', order + 'i2'), ('duse', order + 'i2'),
            ('offset', order + 'i4'), ('gelev', order + 'i4'), ('selev', order + 'i4'),
            ('sdepth', order + 'i4'), ('gdel', order + 'i4'), ('sdel', order + 'i4'),
            ('swdep', order + 'i4'), ('gwdep', order + 'i4'),
            ('scalel', order + 'i2'), ('scalco', order + 'i2'),
            ('sx', order + 'i4'), ('sy', order + 'i4'), ('gx', order + 'i4'), ('gy', order + 'i4'),
            ('counit', order + 'i2'),
            ('pad1', 'V24'),
            ('ns', order + 'u2'), ('dt', order + 'u2'),
            ('pad_a', 'V50'),
            ('shtln', order + 'i4'), ('recln', order + 'i4'), ('shtstn', order + 'i4'), ('recstn', order + 'i4'),
            ('shtstat', order + 'i4'), ('recstat', order + 'i4'), ('shot', order + 'i4'),
            ('cdplbls', order + 'i4'), ('cdplblx', order + 'i4'), ('cdpx', order + 'i4'), ('cdpy', order + 'i4'),
            ('offset2', order + 'i4'),
            ('pad_b', 'V24'),
        ])
        assert header_dtype.itemsize == 240, header_dtype.itemsize
        full_dtype = np.dtype([('header', header_dtype), ('samples', f'V{ns * 4}')])
        return header_dtype, full_dtype

    return (build_header_scan_dtype,)


@app.cell
def _(mo):
    velocity = mo.ui.slider(2000, 12000, value=6000, step=100, label="Reference velocity (ft/s)")
    velocity
    return (velocity,)


@app.cell
def _(mo):
    # STA/LTA-on-Hilbert-envelope ratio a trace must sustain to count as
    # a pick. 8.0 is the value this tool was originally tuned with and
    # tested well against -- an EARLIER bug (this .sgy's samples are
    # IBM float, not IEEE float; see decode_samples/ibm_bits_to_float
    # above) made every trace's actual amplitude garbage, which is why
    # 8.0 looked broken (picked ~3% of traces). With that fixed, 8.0
    # is back to a sensible default (44-86% of traces picked across a
    # few real shots, checked directly). Kept adjustable regardless,
    # since the right value still depends on whatever data is loaded.
    sta_lta_threshold = mo.ui.slider(2.0, 10.0, value=8.0, step=0.5, label="STA/LTA pick threshold")
    sta_lta_threshold
    return (sta_lta_threshold,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Browse every shot in the survey

    Scan once (a full-file header pass, a few seconds for a 15GB survey
    -- this reads only the tiny FLDR field per trace, not the samples),
    then drag the slider to page through **every shot in the file**,
    plotted live.
    """)
    return


@app.cell
def _(mo):
    scan_button = mo.ui.run_button(label="🔍 Scan whole survey for shot list")
    scan_button
    return (scan_button,)


@app.cell
def _(mo):
    get_shot_list, set_shot_list = mo.state(None)
    return get_shot_list, set_shot_list


@app.cell
def _(
    Path,
    REEL_BYTES,
    build_dtypes,
    mo,
    np,
    probe_sgy,
    scan_button,
    set_shot_list,
    su_path,
):
    if scan_button.value:
        # progress bar (traces read, rate, time left) instead of a spinner: this reads the whole file
        _info = probe_sgy(Path(su_path.value))
        with mo.status.progress_bar(total=int(_info["n_traces"]), title="Scanning whole survey for shot (FLDR) list", subtitle=f"reading {_info['n_traces']:,} traces", completion_title="Done", show_rate=True, show_eta=True, remove_on_exit=True) as _bar:
            _, _full_dtype = build_dtypes(_info["order"], _info["ns"])
            _fldrs = np.empty(_info["n_traces"], dtype=np.int64)
            _n_done = 0
            with open(su_path.value, "rb") as _f:
                _f.seek(REEL_BYTES)
                _remaining = _info["n_traces"]
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_f, dtype=_full_dtype, count=_take)
                    _fldrs[_n_done : _n_done + _take] = _chunk["header"]["fldr"]
                    _n_done += _take
                    _remaining -= _take
                    _bar.update(increment=_take)
            set_shot_list(np.unique(_fldrs))
    return


@app.cell
def _(get_shot_list, mo):
    shot_list = get_shot_list()
    if shot_list is None:
        mo.stop(True, mo.md(
            "*Click **Scan whole survey for shot list** above, then use the "
            "slider below to browse every shot.*"
        ))
    mo.md(f"**{len(shot_list):,} unique shots found**, "
          f"FFID {int(shot_list[0])}-{int(shot_list[-1])}.")
    return (shot_list,)


@app.cell
def _(mo, shot_list):
    shot_slider = mo.ui.slider(
        start=0, stop=len(shot_list) - 1, value=0, step=1,
        label="Shot index (drag to browse the whole survey)",
        show_value=True, full_width=True,
    )
    shot_slider
    return (shot_slider,)


@app.cell
def _(shot_list, shot_slider):
    current_fldr = int(shot_list[shot_slider.value])
    return (current_fldr,)


@app.cell
def _(Path, current_fldr, detect_endianness, mo, probe_sgy, read_shot, su_path):
    order = detect_endianness(su_path.value)
    _format_code = probe_sgy(Path(su_path.value))["format_code"]
    shot = read_shot(su_path.value, order, current_fldr, _format_code)
    if shot is None:
        mo.stop(True, mo.md(f"**No traces found for FFID {current_fldr}.**"))
    mo.md(
        f"**FFID {current_fldr}** (shot {current_fldr} of the scanned list) -- "
        f"**{shot['traces'].shape[1]} traces loaded**. ns={shot['ns']}, "
        f"dt={shot['dt_us']} us. Byte order: `{order!r}`, sample format "
        f"code: `{_format_code}` ({'IBM float' if _format_code == 1 else 'IEEE float' if _format_code == 5 else 'unknown, assumed IEEE'})"
    )
    return (shot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### All records for this FFID

    This is 3D land data -- one field record (FFID) holds traces from
    every receiver line that was live for that shot simultaneously.
    Every trace for this FFID is plotted together below, sorted by
    offset.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Directional fix toggles
    """)
    return


@app.cell
def _(mo):
    negate_sx = mo.ui.checkbox(label="Negate Source X")
    negate_sy = mo.ui.checkbox(label="Negate Source Y")
    negate_gx = mo.ui.checkbox(label="Negate Receiver X")
    negate_gy = mo.ui.checkbox(label="Negate Receiver Y")
    swap_source_xy = mo.ui.checkbox(label="Swap Source X/Y only")
    swap_receiver_xy = mo.ui.checkbox(label="Swap Receiver X/Y only")
    mo.hstack([negate_sx, negate_sy, negate_gx, negate_gy, swap_source_xy, swap_receiver_xy])
    return (
        negate_gx,
        negate_gy,
        negate_sx,
        negate_sy,
        swap_receiver_xy,
        swap_source_xy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Manual grid-shift toggle

    A different fix than the sign/swap toggles above: nudge SX, SY, GX,
    and GY **independently and all at once** by a constant amount, and
    see the effect live in the **recomputed** panel below -- for
    testing an off-by-one-station (300 ft) or off-by-one-line (2,100
    ft) hypothesis, on one field or several together. Leave any field
    at 0 to leave it unshifted.
    """)
    return


@app.cell
def _(mo):
    shift_sx = mo.ui.number(value=0.0, step=100.0, label="Shift SX by (ft)")
    shift_sy = mo.ui.number(value=0.0, step=100.0, label="Shift SY by (ft)")
    shift_gx = mo.ui.number(value=0.0, step=100.0, label="Shift GX by (ft)")
    shift_gy = mo.ui.number(value=0.0, step=100.0, label="Shift GY by (ft)")
    mo.hstack([shift_sx, shift_sy, shift_gx, shift_gy])
    return shift_gx, shift_gy, shift_sx, shift_sy


@app.cell
def _(
    mo,
    negate_gx,
    negate_gy,
    negate_sx,
    negate_sy,
    np,
    shift_gx,
    shift_gy,
    shift_sx,
    shift_sy,
    shot,
    swap_receiver_xy,
    swap_source_xy,
):
    # Defensive guard: don't rely solely on the upstream cell's mo.stop()
    # to keep `shot` out of this cell -- toggling a checkbox here can
    # trigger this cell to re-run independently, and if "Load shot"
    # hasn't been clicked yet `shot` is still None at that point.
    if shot is None:
        mo.stop(True, mo.md("*Click **Load shot** above to begin.*"))

    sx = shot["sx"].copy()
    sy = shot["sy"].copy()
    gx = shot["gx"].copy()
    gy = shot["gy"].copy()

    if swap_source_xy.value:
        sx, sy = sy, sx
    if swap_receiver_xy.value:
        gx, gy = gy, gx
    if negate_sx.value:
        sx = -sx
    if negate_sy.value:
        sy = -sy
    if negate_gx.value:
        gx = -gx
    if negate_gy.value:
        gy = -gy

    # Grid shift is independent of (and stacks on top of) the sign/swap
    # toggles above -- purely for live visual experimentation; all four
    # fields can be shifted at once, each by its own amount (0 = no
    # shift on that field).
    sx = sx + shift_sx.value
    sy = sy + shift_sy.value
    gx = gx + shift_gx.value
    gy = gy + shift_gy.value

    offset_recomputed = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
    return (offset_recomputed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Queue this correction, then export a corrected copy of the file

    Once a toggle combination above visibly fixes the current shot,
    save it here. Corrections queue up per-FFID (different shots can
    have different fixes, or none). Shots queued to be **dropped**
    (further down, from the audit's flagged-shots table) are removed
    entirely instead of patched — if a shot is in both lists, dropping
    wins. Nothing is written to disk until you explicitly click
    **Write corrected copy** below, and it is always written to a
    **new file** — the original `.su` is never modified.
    """)
    return


@app.cell
def _(current_fldr, mo):
    save_correction_button = mo.ui.run_button(
        label=f"💾 Save current toggle combination for FFID {current_fldr}"
    )
    remove_correction_button = mo.ui.run_button(
        label=f"🗑 Remove any saved correction for FFID {current_fldr}"
    )
    mo.hstack([save_correction_button, remove_correction_button])
    return remove_correction_button, save_correction_button


@app.cell
def _(mo):
    get_corrections, set_corrections = mo.state({})
    return get_corrections, set_corrections


@app.cell
def _(
    current_fldr,
    get_corrections,
    mo,
    negate_gx,
    negate_gy,
    negate_sx,
    negate_sy,
    remove_correction_button,
    save_correction_button,
    set_corrections,
    swap_receiver_xy,
    swap_source_xy,
):
    _msg = None
    if save_correction_button.value:
        _toggle_vals = {
            "negate_sx": negate_sx.value, "negate_sy": negate_sy.value,
            "negate_gx": negate_gx.value, "negate_gy": negate_gy.value,
            "swap_source_xy": swap_source_xy.value, "swap_receiver_xy": swap_receiver_xy.value,
        }
        if any(_toggle_vals.values()):
            set_corrections({**get_corrections(), current_fldr: {"type": "toggle", **_toggle_vals}})
        else:
            _msg = mo.md(
                f"*No toggles are enabled -- nothing to save for FFID "
                f"{current_fldr}. Enable at least one toggle first.*"
            )
    if remove_correction_button.value:
        set_corrections({k: v for k, v in get_corrections().items() if k != current_fldr})
    _msg
    return


@app.cell
def _(current_fldr, mo):
    save_shift_button = mo.ui.run_button(
        label=f"💾 Save current grid shift for FFID {current_fldr}"
    )
    save_shift_button
    return (save_shift_button,)


@app.cell
def _(
    current_fldr,
    get_corrections,
    mo,
    save_shift_button,
    set_corrections,
    shift_gx,
    shift_gy,
    shift_sx,
    shift_sy,
):
    # A shot's correction is one type or the other, never both -- saving
    # a grid shift here replaces any toggle correction already queued
    # for this FFID (and vice versa, in the toggle-save handler above).
    # All four fields are stored together (not just the one last set),
    # so shifting SX and then SY and saving both keeps both instead of
    # the second overwriting the first.
    _msg = None
    if save_shift_button.value:
        _deltas = {
            "sx": float(shift_sx.value), "sy": float(shift_sy.value),
            "gx": float(shift_gx.value), "gy": float(shift_gy.value),
        }
        if not any(_deltas.values()):
            _msg = mo.md(
                f"*No shift is set -- nothing to save for FFID "
                f"{current_fldr}. Set at least one field to a nonzero "
                f"amount first.*"
            )
        else:
            set_corrections({
                **get_corrections(),
                current_fldr: {"type": "shift", "deltas": _deltas},
            })
    _msg
    return


@app.cell
def _(get_corrections, mo):
    corrections = get_corrections()
    if not corrections:
        _out = mo.md(
            "*No corrections queued yet. Load a flagged shot above (see "
            "the audit section below, or the toggles above), enable "
            "whichever toggle fixes it, then click **Save current toggle "
            "combination**.*"
        )
    else:
        _rows = []
        for _fldr, _combo in sorted(corrections.items()):
            if _combo.get("type") == "shift":
                _rows.append({
                    "FFID": int(_fldr), "Type": "grid shift",
                    **{f"{k.upper()} shift (ft)": v for k, v in _combo["deltas"].items() if v},
                })
            else:
                _rows.append({
                    "FFID": int(_fldr), "Type": "toggle",
                    **{k: v for k, v in _combo.items() if k != "type" and v},
                })
        _out = mo.vstack([
            mo.md(f"**{len(corrections)} correction(s) queued:**"),
            mo.ui.table(_rows),
        ])
    _out
    return (corrections,)


@app.cell
def _(Path, mo, su_path):
    output_path = mo.ui.text(
        label="Output corrected .sgy path (must differ from the input)",
        value=str(Path(su_path.value).with_name(Path(su_path.value).stem + "_geom_corrected.sgy")),
        full_width=True,
    )
    output_path
    return (output_path,)


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="🚀 Write corrected copy of the .sgy file")
    export_button
    return (export_button,)


@app.cell
def _(
    Path,
    REEL_BYTES,
    build_dtypes,
    corrections,
    dropped_fldrs,
    export_button,
    mo,
    np,
    output_path,
    probe_sgy,
    su_path,
):
    _out = None
    if export_button.value:
        if not corrections and not dropped_fldrs:
            mo.stop(True, mo.md(
                "**Nothing queued -- nothing to export.** Save at least one "
                "toggle combination, or queue at least one shot to drop, "
                "first."
            ))
        if Path(output_path.value).resolve() == Path(su_path.value).resolve():
            mo.stop(True, mo.md(
                "**Refusing to write.** Output path is the same as the "
                "input file -- this tool never overwrites the original. "
                "Pick a different output path."
            ))

        _info = probe_sgy(Path(su_path.value))
        with mo.status.progress_bar(total=int(_info["n_traces"]), title=f"Writing corrected copy to {output_path.value}", subtitle=f"full {Path(su_path.value).stat().st_size / 1e9:.1f} GB file copy", completion_title="Done", show_rate=True, show_eta=True, remove_on_exit=True) as _bar:
            _, _full_dtype = build_dtypes(_info["order"], _info["ns"])
            _n_traces = _info["n_traces"]
            _corr_fldrs = np.array(sorted(corrections.keys()), dtype=np.int64)
            _drop_fldrs = np.array(sorted(dropped_fldrs), dtype=np.int64)

            # Reel header is copied through byte-for-byte from the input
            # file -- same textual/binary header, not regenerated, so
            # anything already in there (line/reel numbers, a genuinely
            # custom text header, etc.) survives the correction pass
            # unchanged.
            with open(su_path.value, "rb") as _f:
                _reel_header = _f.read(REEL_BYTES)

            _n_kept = 0
            _n_dropped = 0
            _n_patched = 0
            with open(su_path.value, "rb") as _fin, open(output_path.value, "wb") as _fout:
                _fin.seek(REEL_BYTES)
                _fout.write(_reel_header)
                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_fin, dtype=_full_dtype, count=_take)
                    _h = _chunk["header"]

                    # Dropping wins if a shot is queued both ways -- no
                    # point patching the geometry of a shot you're about
                    # to remove entirely.
                    _drop_mask = np.isin(_h["fldr"], _drop_fldrs) if _drop_fldrs.size else np.zeros(_take, dtype=bool)
                    _keep_mask = ~_drop_mask
                    _corr_mask = np.isin(_h["fldr"], _corr_fldrs) & _keep_mask if _corr_fldrs.size else np.zeros(_take, dtype=bool)

                    if _corr_mask.any():
                        _idx = np.where(_corr_mask)[0]
                        for _i in _idx:
                            _combo = corrections[int(_h["fldr"][_i])]
                            _sx, _sy = float(_h["sx"][_i]), float(_h["sy"][_i])
                            _gx, _gy = float(_h["gx"][_i]), float(_h["gy"][_i])
                            if _combo.get("type") == "shift":
                                _deltas = _combo["deltas"]
                                _sx += _deltas.get("sx", 0.0)
                                _sy += _deltas.get("sy", 0.0)
                                _gx += _deltas.get("gx", 0.0)
                                _gy += _deltas.get("gy", 0.0)
                            else:
                                if _combo.get("swap_source_xy"):
                                    _sx, _sy = _sy, _sx
                                if _combo.get("swap_receiver_xy"):
                                    _gx, _gy = _gy, _gx
                                if _combo.get("negate_sx"):
                                    _sx = -_sx
                                if _combo.get("negate_sy"):
                                    _sy = -_sy
                                if _combo.get("negate_gx"):
                                    _gx = -_gx
                                if _combo.get("negate_gy"):
                                    _gy = -_gy
                            _new_offset = int(round(((_gx - _sx) ** 2 + (_gy - _sy) ** 2) ** 0.5))
                            _h["sx"][_i] = int(_sx)
                            _h["sy"][_i] = int(_sy)
                            _h["gx"][_i] = int(_gx)
                            _h["gy"][_i] = int(_gy)
                            _h["offset"][_i] = _new_offset
                        _n_patched += len(_idx)

                    _chunk[_keep_mask].tofile(_fout)
                    _n_kept += int(_keep_mask.sum())
                    _n_dropped += int(_drop_mask.sum())
                    _remaining -= _take
                    _bar.update(increment=_take)

        _in_size = Path(su_path.value).stat().st_size
        _out_size = Path(output_path.value).stat().st_size
        _record_size = 240 + _info["ns"] * 4
        _expected_size = _in_size - _n_dropped * _record_size
        _out = mo.vstack([
            mo.md(
                f"**Done.** Wrote **{_n_kept:,} traces** to "
                f"`{output_path.value}`, patching **{_n_patched:,} traces** "
                f"across **{len(corrections)} shot(s)** and dropping "
                f"**{_n_dropped:,} traces** across **{len(dropped_fldrs)} "
                f"shot(s)**. Every other trace was copied byte-for-byte "
                f"unchanged."
            ),
            mo.md(
                f"QC: output size {_out_size / 1e9:.3f} GB vs. expected "
                f"{_expected_size / 1e9:.3f} GB (input {_in_size / 1e9:.3f} "
                f"GB minus dropped traces) -- "
                + ("**match, as expected**." if _out_size == _expected_size
                   else "**MISMATCH -- something went wrong, do not use this output.**")
            ),
        ])
    _out
    return


@app.cell
def _(np):
    def decimate_for_display(data, time_ms, target_rows=600):
        """
        Plotly's go.Heatmap is not WebGL-accelerated -- it renders every
        cell individually, so a full ns=4000-row grid (x2 panels) is the
        actual source of the slowness here, not the file I/O (binary
        search + block read for one shot is a few milliseconds; this was
        checked directly against the real 15GB file before touching this
        code). Collapsing the time axis into `target_rows` bins by taking
        the max-abs sample per bin keeps the leading-edge/arrival shape
        this QC actually depends on (a bin's amplitude only drops out if
        every sample in it is small) while cutting heatmap cell count by
        ~6-7x. All calculations (offsets, the reference line, percentile
        clipping) still use the full-resolution data -- only the plotted
        grid is decimated.
        """
        n_samples = data.shape[0]
        if n_samples <= target_rows:
            return data, time_ms

        factor = int(np.ceil(n_samples / target_rows))
        n_bins = -(-n_samples // factor)  # ceil division
        pad = n_bins * factor - n_samples
        if pad:
            data = np.pad(data, ((0, pad), (0, 0)), mode="edge")

        reshaped = data.reshape(n_bins, factor, data.shape[1])
        idx = np.argmax(np.abs(reshaped), axis=1)
        binned = np.take_along_axis(reshaped, idx[:, None, :], axis=1)[:, 0, :]
        binned_time_ms = time_ms[::factor][:n_bins]
        return binned, binned_time_ms

    return (decimate_for_display,)


@app.cell
def _(mo):
    full_resolution = mo.ui.checkbox(
        label="Show full time resolution (every record/sample, no decimation -- slower to render)"
    )
    overlay_hilbert_picks = mo.ui.checkbox(
        label="Overlay Hilbert-based first-break picks (green dots)"
    )
    mo.hstack([full_resolution, overlay_hilbert_picks])
    return full_resolution, overlay_hilbert_picks


@app.cell
def _(hilbert, np, shot, sta_lta_threshold):
    # Same STA/LTA-on-Hilbert-envelope-energy method as the whole-survey
    # audit above, run here for just the one loaded shot (cheap -- a few
    # hundred traces). Always computed (not gated behind the overlay
    # checkbox) so the automatic geometry-search section below can reuse
    # it too; the checkbox only controls whether it's drawn on the plot.
    hilbert_pick_ms = None
    hilbert_has_pick = None
    if shot is not None:
        _dt_s = shot["dt_us"] * 1e-6
        _short_win = max(2, int(round(0.010 / _dt_s)))
        _long_win = max(_short_win + 1, int(round(0.100 / _dt_s)))
        _threshold = sta_lta_threshold.value
        _min_consecutive = 5

        _samples = shot["traces"].T.astype(np.float64)  # (n_traces, ns)
        _energy = np.abs(hilbert(_samples, axis=1)) ** 2

        _csum = np.concatenate([np.zeros((_energy.shape[0], 1)), np.cumsum(_energy, axis=1)], axis=1)
        _idx = np.arange(_long_win - 1, _energy.shape[1])
        _sta = (_csum[:, _idx + 1] - _csum[:, _idx + 1 - _short_win]) / _short_win
        _lta = (_csum[:, _idx + 1] - _csum[:, _idx + 1 - _long_win]) / _long_win
        _exceed = (_sta / (_lta + 1e-30)) > _threshold

        if _min_consecutive > 1:
            _sustained = _exceed.copy()
            _n = _exceed.shape[1]
            for _j in range(1, _min_consecutive):
                _shifted = np.zeros_like(_exceed)
                if _n - _j > 0:
                    _shifted[:, : _n - _j] = _exceed[:, _j:]
                _sustained &= _shifted
            _exceed = _sustained

        hilbert_has_pick = _exceed.any(axis=1)
        _first_pos = np.argmax(_exceed, axis=1)
        _pick_sample = np.where(hilbert_has_pick, _idx[_first_pos], -1)
        hilbert_pick_ms = np.where(hilbert_has_pick, _pick_sample * _dt_s * 1000.0, np.nan)
    return hilbert_has_pick, hilbert_pick_ms


@app.cell
def _(
    current_fldr,
    decimate_for_display,
    full_resolution,
    hilbert_has_pick,
    hilbert_pick_ms,
    mo,
    np,
    offset_recomputed,
    overlay_hilbert_picks,
    plt,
    qc_plot_dir,
    shot,
    velocity,
):
    if shot is None:
        mo.stop(True, mo.md("*Click **Load shot** above to begin.*"))

    def make_qc_figure(offsets, title):
        """
        Traces left in as-stored (acquisition) order, x-axis = trace
        number -- matches suximage's default axis. Each receiver line
        active for this shot occupies its own contiguous block of trace
        numbers, so the reference line (drawn here using each trace's
        own offset, unsorted) breaks into one zigzag per line instead
        of a single smooth curve -- letting a problem on just one
        specific receiver line stand out.
        """
        data = shot["traces"].astype(np.float64)
        x_vals = np.arange(1, len(offsets) + 1)

        dt_ms = shot["dt_us"] / 1000.0
        time_ms = np.arange(shot["ns"]) * dt_ms
        clip = np.percentile(np.abs(data), 98) or 1.0

        if full_resolution.value:
            data_disp, time_ms_disp = data, time_ms
        else:
            data_disp, time_ms_disp = decimate_for_display(data, time_ms)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(
            data_disp, aspect="auto", cmap="gray", vmin=-clip, vmax=clip,
            extent=[x_vals.min(), x_vals.max(), time_ms_disp[-1], time_ms_disp[0]],
        )

        # Reference line: t = |offset| / V, in ms
        v = velocity.value
        ref_time_ms = (np.abs(offsets) / v) * 1000.0
        ax.plot(x_vals, ref_time_ms, color="red", lw=1.5, label=f"{v} ft/s linear velocity")

        if overlay_hilbert_picks.value and hilbert_pick_ms is not None:
            ax.scatter(
                x_vals[hilbert_has_pick], hilbert_pick_ms[hilbert_has_pick],
                s=8, color="lime", marker="o", label="Hilbert-based pick", zorder=5,
            )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Trace number")
        ax.set_ylabel("TWT (ms)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        return fig

    fig_stored_tn = make_qc_figure(shot["offset"], "RAW Offset")
    fig_recomputed_tn = make_qc_figure(offset_recomputed, "Recomputed Offset")

    _p_stored_tn = qc_plot_dir / f"fldr{current_fldr}_raw_tracenum.png"
    _p_recomputed_tn = qc_plot_dir / f"fldr{current_fldr}_recomputed_tracenum.png"
    fig_stored_tn.savefig(_p_stored_tn, dpi=130)
    fig_recomputed_tn.savefig(_p_recomputed_tn, dpi=130)
    plt.close(fig_stored_tn)
    plt.close(fig_recomputed_tn)

    mo.vstack([
        mo.md(f"Showing all **{shot['traces'].shape[1]} traces** for this FFID."),
        mo.hstack([mo.image(src=str(_p_stored_tn), width=520), mo.image(src=str(_p_recomputed_tn), width=520)]),
        mo.hstack([
            mo.download(
                data=_p_stored_tn.read_bytes(), filename=_p_stored_tn.name,
                mimetype="image/png", label="💾 Raw",
            ),
            mo.download(
                data=_p_recomputed_tn.read_bytes(), filename=_p_recomputed_tn.name,
                mimetype="image/png", label="💾 Recomputed",
            ),
        ]),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    QC: If the red reference line tracks the
    leading edge of the real data in the **left** panel (as-stored
    offset), the header geometry is already consistent — no fix
    needed. If it only tracks in the **right** panel after
    enabling a toggle, that toggle is your directional fix —
    confirm it by repeating with other shots from different parts
    of the survey.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Automated whole-survey geometry audit (find bad shots automatically)

    Simple, direct comparison, no curve-fitting: for every trace, take
    the **LMO** time implied by its offset and the reference velocity
    above (`t_LMO = |offset| / V` — exactly the red reference line on
    the plots above) and compare it to that trace's **Hilbert-based
    energy pick** (same picker as the overlay checkbox above). The
    difference between the two, averaged per shot, is the misfit
    reported below. A shot whose picks systematically disagree with
    its own LMO by a lot is the one worth checking for a geometry
    problem.
    """)
    return


@app.cell
def _(mo):
    run_audit_button = mo.ui.run_button(
        label="▶ Run full-survey geometry audit (~2 min, reads every trace once)"
    )
    run_audit_button
    return (run_audit_button,)


@app.cell
def _(mo):
    get_audit, set_audit = mo.state(None)
    return get_audit, set_audit


@app.cell
def _(
    Path,
    REEL_BYTES,
    build_dtypes,
    get_shot_list,
    hilbert,
    ibm_bits_to_float,
    mo,
    np,
    probe_sgy,
    run_audit_button,
    set_audit,
    sta_lta_threshold,
    su_path,
    velocity,
):
    # Read the shot list from its state getter directly, NOT from the
    # `shot_list` variable used elsewhere in the notebook -- that variable
    # comes from a cell that calls mo.stop() when the scan hasn't been run
    # yet, and depending on it here means this whole cell (including its
    # own button-click handling) never runs at all in that case -- no
    # spinner, no message, no error, just silently nothing, which is
    # exactly what made this look broken. Reading the getter directly
    # keeps this button independent of that gate, with its own message.
    _shot_list = get_shot_list()
    if run_audit_button.value and _shot_list is None:
        mo.stop(True, mo.md(
            "**Can't run the audit yet.** Click **Scan whole survey for "
            "shot list** above first (in the \"Browse every shot\" "
            "section) -- the audit needs that shot list before it can "
            "start."
        ))

    if run_audit_button.value:
        _info = probe_sgy(Path(su_path.value))
        with mo.status.progress_bar(total=int(_info["n_traces"]), title="Running full-survey geometry audit (Hilbert picks vs. LMO)", subtitle=f"reading and picking {_info['n_traces']:,} traces", completion_title="Done", show_rate=True, show_eta=True, remove_on_exit=True) as _bar:
            _ns, _order, _n_traces = _info["ns"], _info["order"], _info["n_traces"]
            _dt_s = _info["dt_us"] * 1e-6
            _, _full_dtype = build_dtypes(_order, _ns)
            _format_code = _info["format_code"]
            _v = velocity.value

            # Same STA/LTA parameters (and the same adjustable threshold)
            # as the pick overlay above.
            _short_win = max(2, int(round(0.010 / _dt_s)))
            _long_win = max(_short_win + 1, int(round(0.100 / _dt_s)))
            _threshold = sta_lta_threshold.value
            _min_consecutive = 5

            _n_src = len(_shot_list)
            _sum_absmisfit = np.zeros(_n_src)
            _n_picks = np.zeros(_n_src)
            _n_large = np.zeros(_n_src)
            _LARGE_MS = 50.0

            def _sustained(exceed, k):
                if k <= 1:
                    return exceed
                out = exceed.copy()
                n = exceed.shape[1]
                for j in range(1, k):
                    shifted = np.zeros_like(exceed)
                    if n - j > 0:
                        shifted[:, : n - j] = exceed[:, j:]
                    out &= shifted
                return out

            def _pick_hilbert(samples):
                energy = np.abs(hilbert(samples, axis=1)) ** 2
                csum = np.concatenate([np.zeros((energy.shape[0], 1)), np.cumsum(energy, axis=1)], axis=1)
                idx = np.arange(_long_win - 1, energy.shape[1])
                sta = (csum[:, idx + 1] - csum[:, idx + 1 - _short_win]) / _short_win
                lta = (csum[:, idx + 1] - csum[:, idx + 1 - _long_win]) / _long_win
                ratio = sta / (lta + 1e-30)
                exceed = _sustained(ratio > _threshold, _min_consecutive)
                has_pick = exceed.any(axis=1)
                first_pos = np.argmax(exceed, axis=1)
                pick_sample = np.where(has_pick, idx[first_pos], -1)
                return np.where(has_pick, pick_sample * _dt_s, np.nan), has_pick

            _n_done = 0
            _n_picked = 0
            with open(su_path.value, "rb") as _f:
                _f.seek(REEL_BYTES)
                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_f, dtype=_full_dtype, count=_take)
                    _h = _chunk["header"]
                    _fldr_chunk = _h["fldr"]
                    _offset_chunk = np.abs(_h["offset"].astype(np.float64))
                    if _format_code == 1:
                        _samples = ibm_bits_to_float(_chunk["samples"].view(_order + "u4")).astype(np.float64)
                    else:
                        _samples = _chunk["samples"].astype(np.float64)

                    _t_pick, _has_pick = _pick_hilbert(_samples)

                    if _has_pick.any():
                        _si = np.searchsorted(_shot_list, _fldr_chunk[_has_pick])
                        _t_lmo_ms = (_offset_chunk[_has_pick] / _v) * 1000.0
                        _misfit_ms = np.abs(_t_pick[_has_pick] * 1000.0 - _t_lmo_ms)
                        _sum_absmisfit += np.bincount(_si, weights=_misfit_ms, minlength=_n_src)
                        _n_picks += np.bincount(_si, minlength=_n_src)
                        _n_large += np.bincount(_si, weights=(_misfit_ms > _LARGE_MS).astype(float), minlength=_n_src)
                        _n_picked += int(_has_pick.sum())

                    _n_done += _take
                    _remaining -= _take
                    _bar.update(increment=_take)

            with np.errstate(divide="ignore", invalid="ignore"):
                _mean_absmisfit = np.where(_n_picks > 0, _sum_absmisfit / _n_picks, np.nan)
                _frac_large = np.where(_n_picks > 0, _n_large / _n_picks, np.nan)

            set_audit({
                "fldr": _shot_list.copy(),
                "velocity_used": _v,
                "mean_absmisfit_ms": _mean_absmisfit,
                "frac_large": _frac_large,
                "n_picks": _n_picks,
                "n_picked_total": _n_picked,
                "n_traces": _n_traces,
            })
    return


@app.cell
def _(get_audit, mo):
    audit = get_audit()
    if audit is None:
        mo.stop(True, mo.md(
            "*Click **Run full-survey geometry audit** above -- this reads "
            "every trace once and takes about two minutes on this survey.*"
        ))
    return (audit,)


@app.cell
def _(mo):
    misfit_threshold = mo.ui.slider(20, 1000, value=300, step=20, label="Flag shots with mean misfit above (ms)")
    min_picks_lmo = mo.ui.slider(5, 200, value=20, step=5, label="Minimum picks required to trust a shot")
    mo.hstack([misfit_threshold, min_picks_lmo])
    return min_picks_lmo, misfit_threshold


@app.cell
def _(audit, min_picks_lmo, misfit_threshold, mo, np):
    effective_threshold = misfit_threshold.value
    _valid = audit["n_picks"] >= min_picks_lmo.value
    _misfit_safe = np.nan_to_num(np.where(_valid, audit["mean_absmisfit_ms"], np.nan), nan=0.0)
    _flag_mask = _valid & (_misfit_safe > effective_threshold)
    _order_idx = np.argsort(-_misfit_safe)
    _flagged_idx = [i for i in _order_idx if _flag_mask[i]]
    _median_misfit = np.nanmedian(np.where(_valid, audit["mean_absmisfit_ms"], np.nan))

    _rows = [
        {
            "FFID": int(audit["fldr"][i]),
            "Mean misfit (ms)": round(float(audit["mean_absmisfit_ms"][i]), 1),
            "Fraction large (>50ms)": f"{100 * audit['frac_large'][i]:.1f}%",
            "N picks": int(audit["n_picks"][i]),
        }
        for i in _flagged_idx
    ]
    # Named + selectable so a later cell can read which rows the user
    # checked off and queue them to be dropped from a corrected export.
    flagged_table = mo.ui.table(_rows, page_size=15, selection="multi")

    _threshold_desc = f"{effective_threshold:.0f} ms (manual slider)"

    if len(_flagged_idx) == 0:
        _out = mo.md(
            f"**No shots flagged** above {_threshold_desc} mean misfit "
            f"(with ≥{min_picks_lmo.value} picks, {int(_valid.sum())} shots "
            f"scored; survey median misfit is {_median_misfit:.0f} ms for "
            f"context)."
        )
    else:
        _out = mo.vstack([
            mo.md(
                f"**{len(_flagged_idx)} shot(s) flagged** out of "
                f"{int(_valid.sum())} scored (mean misfit above "
                f"{_threshold_desc}, worst first; survey median is "
                f"{_median_misfit:.0f} ms). Check each flagged shot's "
                f"**offset range** before concluding it's a geometry error "
                f"(see caveat above) -- drag the shot-index slider to its "
                f"FFID to inspect it with the toggles and pick overlay, or "
                f"**select rows below** to drop those shots entirely from "
                f"a corrected export (see the drop-shots section further "
                f"down)."
            ),
            flagged_table,
        ])
    _out
    return effective_threshold, flagged_table


@app.cell
def _(mo):
    mo.md("""
    ---
    ### Automatic geometry-correction search (for the currently loaded shot)

    It calculate the misfit between its LMO and Hilbert-based picks.
    Ranked worst-to-best; the lowest-misfit combination is the
    automatic suggestion.

    **This is a suggestion, not a verdict** — confirm it visually with
    the toggles above before queuing it (a lower misfit can also come
    from the wrong reference velocity or an odd offset range).
    """)
    return


@app.cell
def _(hilbert_has_pick, hilbert_pick_ms, mo, np, shot, velocity):
    if shot is None or hilbert_pick_ms is None:
        mo.stop(True, mo.md("*Load a shot above -- Hilbert picks are computed automatically once one is loaded.*"))

    _sx0, _sy0 = shot["sx"], shot["sy"]
    _gx0, _gy0 = shot["gx"], shot["gy"]
    _v = velocity.value
    _mask = hilbert_has_pick
    _t_pick_ms = hilbert_pick_ms[_mask]

    _combo_keys = ["swap_source_xy", "swap_receiver_xy", "negate_sx", "negate_sy", "negate_gx", "negate_gy"]
    _results = []
    for _bits in range(64):
        _combo = {k: bool((_bits >> i) & 1) for i, k in enumerate(_combo_keys)}
        _sx, _sy, _gx, _gy = _sx0.copy(), _sy0.copy(), _gx0.copy(), _gy0.copy()
        if _combo["swap_source_xy"]:
            _sx, _sy = _sy, _sx
        if _combo["swap_receiver_xy"]:
            _gx, _gy = _gy, _gx
        if _combo["negate_sx"]:
            _sx = -_sx
        if _combo["negate_sy"]:
            _sy = -_sy
        if _combo["negate_gx"]:
            _gx = -_gx
        if _combo["negate_gy"]:
            _gy = -_gy
        _offset = np.sqrt((_gx - _sx) ** 2 + (_gy - _sy) ** 2)
        _t_lmo_ms = (_offset[_mask] / _v) * 1000.0
        _misfit = float(np.mean(np.abs(_t_pick_ms - _t_lmo_ms))) if _mask.any() else float("nan")
        _label = "as-stored (no change)" if _bits == 0 else "+".join(k for k in _combo_keys if _combo[k])
        _results.append({"combo": {"type": "toggle", **_combo}, "label": _label, "misfit_ms": _misfit})

    _results.sort(key=lambda r: r["misfit_ms"])
    geometry_search_results = _results
    return (geometry_search_results,)


@app.cell
def _(current_fldr, geometry_search_results, go, mo):
    baseline_result = next(r for r in geometry_search_results if r["label"] == "as-stored (no change)")
    best_result = geometry_search_results[0]
    # Require a real (>=20%) improvement, not just numerical noise, before
    # calling a different combination "better" than as-stored.
    combo_improved = (
        best_result["label"] != "as-stored (no change)"
        and best_result["misfit_ms"] < baseline_result["misfit_ms"] * 0.8
    )

    _labels = [r["label"] for r in geometry_search_results]
    _misfits = [r["misfit_ms"] for r in geometry_search_results]
    _colors = [
        "#888888" if r["label"] == "as-stored (no change)"
        else "#2CA02C" if r is best_result
        else "#4C78A8"
        for r in geometry_search_results
    ]

    fig_combo = go.Figure(go.Bar(
        x=list(range(1, len(geometry_search_results) + 1)),
        y=_misfits,
        marker=dict(color=_colors),
        customdata=_labels,
        hovertemplate="%{customdata}<br>misfit=%{y:.1f} ms<extra></extra>",
    ))
    fig_combo.update_layout(
        title=(
            f"FFID {current_fldr} "
        ),
        xaxis_title="Rank (best to worst)", yaxis_title="Mean misfit (ms)",
        width=800, height=450,
    )

    if combo_improved:
        _msg = mo.md(
            f"**Best combination found: `{best_result['label']}`** -- misfit "
            f"drops from {baseline_result['misfit_ms']:.1f} ms (as-stored) "
            f"to {best_result['misfit_ms']:.1f} ms for FFID {current_fldr}."
        )
    else:
        _msg = mo.md(
            f"**As-stored geometry already looks best** for FFID "
            f"{current_fldr} -- baseline misfit "
            f"{baseline_result['misfit_ms']:.1f} ms, best alternative found "
            f"{best_result['misfit_ms']:.1f} ms (`{best_result['label']}`) "
            f"-- not enough of an improvement to suggest a fix."
        )

    mo.vstack([_msg, mo.ui.plotly(
        fig_combo,
        config={
            "displayModeBar": True,
            "toImageButtonOptions": {"filename": f"fldr{current_fldr}_toggle_search", "format": "png", "scale": 2},
        },
    )])
    return best_result, combo_improved


@app.cell
def _(best_result, combo_improved, current_fldr, mo):
    if combo_improved:
        _label = f"✅ Queue [{best_result['label']}] as the correction for FFID {current_fldr}"
    else:
        _label = f"No improvement found for FFID {current_fldr} -- nothing to queue"
    apply_best_button = mo.ui.run_button(label=_label)
    apply_best_button
    return (apply_best_button,)


@app.cell
def _(
    apply_best_button,
    best_result,
    combo_improved,
    current_fldr,
    get_corrections,
    mo,
    set_corrections,
):
    # Guard combo_improved again here (not just in the button's label) --
    # otherwise clicking the button when the search found no real fix would
    # queue "as-stored (no change)" itself as a "correction" (all six
    # toggles False), a confusing no-op that changes nothing if exported.
    _msg = None
    if apply_best_button.value:
        if combo_improved:
            set_corrections({**get_corrections(), current_fldr: best_result["combo"]})
            _msg = mo.md(
                f"Queued `{best_result['label']}` as the correction for FFID "
                f"{current_fldr}. Scroll up to the corrections table / export "
                f"section to review and write it out to a new file."
            )
        else:
            _msg = mo.md(
                f"*No improvement was found for FFID {current_fldr} -- "
                f"nothing was queued.*"
            )
    _msg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Shift-misfit landscape (hover to explore)

    The discrete search above only tries ±1 station/±1 line. This
    sweeps **SX, SY, GX, GY** continuously across a range of shift
    amounts and plots the resulting misfit as one curve per field for
    whichever shot is loaded — hover anywhere on a curve to read the
    exact shift/misfit at that point. The visual minimum of a curve is
    that field's best-fitting shift; the dashed line is the as-stored
    baseline for comparison. If a curve is still falling at the edge
    of the sweep range (as it will for a shot with no real shift
    error), that's itself a sign there's no clean minimum to find —
    widen the range and it should just keep falling, not level off.
    """)
    return


@app.cell
def _(mo):
    shift_range = mo.ui.slider(500, 25000, value=3000, step=500, label="Shift sweep range (± ft)")
    shift_step = mo.ui.slider(25, 200, value=50, step=25, label="Shift sweep step (ft)")
    mo.hstack([shift_range, shift_step])
    return shift_range, shift_step


@app.cell
def _(
    current_fldr,
    go,
    hilbert_has_pick,
    hilbert_pick_ms,
    mo,
    np,
    shift_range,
    shift_step,
    shot,
    velocity,
):
    if shot is None or hilbert_pick_ms is None:
        mo.stop(True, mo.md("*Load a shot above -- Hilbert picks are computed automatically once one is loaded.*"))

    _sx0, _sy0 = shot["sx"], shot["sy"]
    _gx0, _gy0 = shot["gx"], shot["gy"]
    _v = velocity.value
    _mask = hilbert_has_pick
    _t_pick_ms = hilbert_pick_ms[_mask]

    if not _mask.any():
        mo.stop(True, mo.md(f"*No Hilbert picks for FFID {current_fldr} -- can't build the landscape.*"))

    _deltas = np.arange(-shift_range.value, shift_range.value + shift_step.value, shift_step.value)

    def _misfit_curve(field):
        _misfits = np.empty(len(_deltas))
        for _i, _d in enumerate(_deltas):
            _sx, _sy, _gx, _gy = _sx0, _sy0, _gx0, _gy0
            if field == "sx":
                _sx = _sx0 + _d
            elif field == "sy":
                _sy = _sy0 + _d
            elif field == "gx":
                _gx = _gx0 + _d
            elif field == "gy":
                _gy = _gy0 + _d
            _offset = np.sqrt((_gx - _sx) ** 2 + (_gy - _sy) ** 2)
            _t_lmo_ms = (_offset[_mask] / _v) * 1000.0
            _misfits[_i] = np.mean(np.abs(_t_pick_ms - _t_lmo_ms))
        return _misfits

    _offset0 = np.sqrt((_gx0 - _sx0) ** 2 + (_gy0 - _sy0) ** 2)
    _baseline_misfit = float(np.mean(np.abs(_t_pick_ms - (_offset0[_mask] / _v) * 1000.0)))

    fig_landscape = go.Figure()
    _colors = {"sx": "#4C78A8", "sy": "#54A24B", "gx": "#F58518", "gy": "#E45756"}
    _best_overall = None
    for _field in ["sx", "sy", "gx", "gy"]:
        _misfits = _misfit_curve(_field)
        fig_landscape.add_trace(go.Scatter(
            x=_deltas, y=_misfits, mode="lines", name=_field.upper(),
            line=dict(color=_colors[_field]),
            hovertemplate=f"{_field.upper()} shift=%{{x:.0f}} ft<br>misfit=%{{y:.1f}} ms<extra></extra>",
        ))
        _i_min = int(np.argmin(_misfits))
        if _best_overall is None or _misfits[_i_min] < _best_overall[2]:
            _best_overall = (_field, float(_deltas[_i_min]), float(_misfits[_i_min]))

    fig_landscape.add_hline(
        y=_baseline_misfit, line_dash="dash", line_color="gray",
        annotation_text=f"as-stored ({_baseline_misfit:.0f} ms)", annotation_position="top left",
    )

    fig_landscape.update_layout(
        title=(
            f"FFID {current_fldr}: Misfit vs Shift"
        ),
        xaxis_title="Shift amount (ft)", yaxis_title="Mean misfit (ms)",
        width=800, height=500,
        legend_title="Field",
    )

    mo.ui.plotly(
        fig_landscape,
        config={
            "displayModeBar": True,
            "toImageButtonOptions": {"filename": f"fldr{current_fldr}_shift_landscape", "format": "png", "scale": 2},
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Drop shots from the corrected export

    Rather than (or in addition to) fixing a shot's geometry with a
    toggle combination, you can simply **exclude** it entirely from a
    corrected export — useful when a flagged shot's problem isn't a
    fixable sign/swap error (the automatic search above will tell you
    that). Select rows in the flagged-shots table above, then queue
    them for removal below. Dropped shots are only removed from the
    **exported copy** — the original `.su` file is never touched, and
    nothing is dropped until you actually run the export further down.
    """)
    return


@app.cell
def _(mo):
    get_dropped, set_dropped = mo.state(frozenset())
    return get_dropped, set_dropped


@app.cell
def _(flagged_table, mo):
    _n_selected = len(flagged_table.value)
    drop_selected_button = mo.ui.run_button(
        label=f"🗑 Queue {_n_selected} selected shot(s) to drop"
        if _n_selected else "🗑 Queue selected shot(s) to drop (select rows in the table above first)"
    )
    drop_selected_button
    return (drop_selected_button,)


@app.cell
def _(drop_selected_button, flagged_table, get_dropped, mo, set_dropped):
    _msg = None
    if drop_selected_button.value:
        _selected_fldrs = {int(row["FFID"]) for row in flagged_table.value}
        if _selected_fldrs:
            set_dropped(get_dropped() | frozenset(_selected_fldrs))
        else:
            _msg = mo.md("*No rows selected in the flagged-shots table -- nothing queued.*")
    _msg
    return


@app.cell
def _(get_dropped, mo):
    dropped_fldrs = get_dropped()
    clear_dropped_button = mo.ui.run_button(label="Clear dropped-shots list")

    if not dropped_fldrs:
        _out = mo.md("*No shots queued to drop yet.*")
    else:
        _out = mo.vstack([
            mo.md(
                f"**{len(dropped_fldrs)} shot(s) queued to drop:** "
                + ", ".join(str(f) for f in sorted(dropped_fldrs))
            ),
            clear_dropped_button,
        ])
    _out
    return clear_dropped_button, dropped_fldrs


@app.cell
def _(clear_dropped_button, set_dropped):
    if clear_dropped_button.value:
        set_dropped(frozenset())
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Acquisition geometry map

    Plan-view (map) of the whole survey: every receiver station
    (from `GX`/`GY`, small gray dots) and every shot point (from
    `SX`/`SY`, one point per unique `FLDR`). Hover over a shot point
    to see its FFID.
    The currently loaded shot (from the slider above) is highlighted
    in red. The scan runs automatically (a few seconds, header
    fields only) -- no button to click.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### IL/XL

    Off by default. Switch to IL/XL below and this asks for a grid
    corner table on the spot -- nothing is read from or written to any
    header field, it's purely a plotting convenience computed from
    whatever corner points you enter.
    """)
    return


@app.cell
def _(mo):
    coord_display = mo.ui.radio(
        options=["X/Y (ft)", "IL/XL"],
        value="X/Y (ft)", inline=True,
        label="Plot coordinates (all geometry plots)",
    )
    coord_display
    return (coord_display,)


@app.cell
def _(Path, json):
    ILXL_TABLE_PATH = Path(__file__).resolve().parent / "ilxl_corner_table.json"
    _ILXL_BLANK_TABLE = [
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
    ]

    def load_ilxl_table():
        if ILXL_TABLE_PATH.exists():
            try:
                return json.loads(ILXL_TABLE_PATH.read_text())
            except Exception:
                pass
        return list(_ILXL_BLANK_TABLE)

    def save_ilxl_table(rows):
        ILXL_TABLE_PATH.write_text(json.dumps(rows, indent=2))

    return ILXL_TABLE_PATH, load_ilxl_table, save_ilxl_table


@app.cell
def _(coord_display, load_ilxl_table, mo):
    ilxl_corner_table = mo.ui.data_editor(
        data=load_ilxl_table(),
        label="Grid corner table (IL, XL, X, Y) -- enter at least 2 known corner points",
    )
    ilxl_save_button = mo.ui.run_button(label="💾 Save this corner table (auto-loads next time)")
    _out = mo.vstack([
        mo.md("Enter this survey's known IL/XL &harr; X/Y corner points:"),
        ilxl_corner_table,
        ilxl_save_button,
    ]) if coord_display.value == "IL/XL" else mo.md("")
    _out
    return ilxl_corner_table, ilxl_save_button


@app.cell
def _(ILXL_TABLE_PATH, coord_display, ilxl_corner_table, ilxl_save_button, mo, save_ilxl_table):
    _msg = None
    if ilxl_save_button.value:
        save_ilxl_table(ilxl_corner_table.value)
        _msg = mo.callout(
            f"Saved to {ILXL_TABLE_PATH.name} -- will auto-load next time IL/XL is opened.",
            kind="success",
        )
    _display = _msg if coord_display.value == "IL/XL" else None
    _display
    return


@app.cell
def _(coord_display, ilxl_corner_table, mo, np):
    _rows = ilxl_corner_table.value
    _xl = np.array([r["XL"] for r in _rows], dtype=np.float64)
    _il = np.array([r["IL"] for r in _rows], dtype=np.float64)
    _x = np.array([r["X"] for r in _rows], dtype=np.float64)
    _y = np.array([r["Y"] for r in _rows], dtype=np.float64)

    _valid = len(_rows) >= 2 and _x.max() != _x.min() and _y.max() != _y.min()
    if _valid:
        # Least-squares line fit: XL is linear in X alone, IL linear in
        # Y alone, for an axis-aligned grid.
        _mx, _cx = np.polyfit(_x, _xl, 1)
        _my, _cy = np.polyfit(_y, _il, 1)
        _bin_x = 1.0 / _mx
        _bin_y = 1.0 / _my
        ilxl_x0 = (1001.0 - _cx) * _bin_x
        ilxl_y0 = (1001.0 - _cy) * _bin_y
        ilxl_bin_ft = (_bin_x + _bin_y) / 2.0
        _msg = mo.md(
            f"Derived grid: **X0 = {ilxl_x0:,.1f}**, **Y0 = {ilxl_y0:,.1f}**, "
            f"**bin size = {ilxl_bin_ft:.2f} ft**."
        )
    else:
        # Placeholder so to_display_xy below always has something to
        # divide by (X/Y mode never uses these anyway); IL/XL mode with
        # no real corners yet just shows a warning instead of crashing.
        ilxl_x0, ilxl_y0, ilxl_bin_ft = 0.0, 0.0, 1.0
        _msg = mo.callout(
            "Enter at least 2 corner rows with distinct X values and "
            "distinct Y values above to compute IL/XL.",
            kind="warn",
        )

    _display = _msg if coord_display.value == "IL/XL" else None
    _display
    return ilxl_bin_ft, ilxl_x0, ilxl_y0


@app.cell
def _(coord_display, ilxl_bin_ft, ilxl_x0, ilxl_y0):
    def to_display_xy(x, y):
        if coord_display.value == "IL/XL":
            return 1001 + (x - ilxl_x0) / ilxl_bin_ft, 1001 + (y - ilxl_y0) / ilxl_bin_ft
        return x, y
    return (to_display_xy,)


@app.cell
def _(Path, REEL_BYTES, build_header_scan_dtype, mo, np, probe_sgy, su_path):
    _info = probe_sgy(Path(su_path.value))
    with mo.status.progress_bar(total=int(_info["n_traces"]), title="Scanning whole survey headers for acquisition geometry (SX, SY, GX, GY, FLDR)", subtitle=f"reading {_info['n_traces']:,} traces", completion_title="Done", show_rate=True, show_eta=True, remove_on_exit=True) as _bar:
        _, _full_dtype = build_header_scan_dtype(_info["order"], _info["ns"])
        _n = _info["n_traces"]
        _sx = np.empty(_n, dtype=np.int64)
        _sy = np.empty(_n, dtype=np.int64)
        _gx = np.empty(_n, dtype=np.int64)
        _gy = np.empty(_n, dtype=np.int64)
        _fldr = np.empty(_n, dtype=np.int64)
        _done = 0
        with open(su_path.value, "rb") as _f:
            _f.seek(REEL_BYTES)
            _remaining = _n
            while _remaining > 0:
                _take = min(20000, _remaining)
                _chunk = np.fromfile(_f, dtype=_full_dtype, count=_take)
                _h = _chunk["header"]
                _sx[_done:_done + _take] = _h["sx"]
                _sy[_done:_done + _take] = _h["sy"]
                _gx[_done:_done + _take] = _h["gx"]
                _gy[_done:_done + _take] = _h["gy"]
                _fldr[_done:_done + _take] = _h["fldr"]
                _done += _take
                _remaining -= _take
                _bar.update(increment=_take)

    header_scan = {"sx": _sx, "sy": _sy, "gx": _gx, "gy": _gy, "fldr": _fldr}
    return (header_scan,)


@app.cell
def _(header_scan, mo, np):
    # One (sx, sy) per unique FLDR -- source position is constant across
    # all traces of a shot, so the first occurrence of each FLDR is enough.
    _fldr = header_scan["fldr"]
    _order_idx = np.argsort(_fldr, kind="stable")
    _fldr_sorted = _fldr[_order_idx]
    _first_mask = np.concatenate(([True], _fldr_sorted[1:] != _fldr_sorted[:-1]))
    _first_idx = _order_idx[_first_mask]

    map_shot_fldr = header_scan["fldr"][_first_idx]
    map_shot_sx = header_scan["sx"][_first_idx]
    map_shot_sy = header_scan["sy"][_first_idx]

    # Unique receiver stations (dedup -- each is hit by many shots).
    _recv_xy = np.unique(
        np.column_stack([header_scan["gx"], header_scan["gy"]]), axis=0
    )
    map_recv_gx = _recv_xy[:, 0]
    map_recv_gy = _recv_xy[:, 1]

    mo.md(
        f"**{len(map_shot_fldr):,} shot points** and "
        f"**{len(map_recv_gx):,} unique receiver stations** in this survey."
    )
    return map_recv_gx, map_recv_gy, map_shot_fldr, map_shot_sx, map_shot_sy


@app.cell
def _(
    coord_display,
    current_fldr,
    go,
    map_recv_gx,
    map_recv_gy,
    map_shot_fldr,
    map_shot_sx,
    map_shot_sy,
    mo,
    np,
    to_display_xy,
):
    fig = go.Figure()
    _is_ilxl = coord_display.value == "IL/XL"
    _x_label, _y_label = ("XL", "IL") if _is_ilxl else ("X", "Y")

    _recv_x, _recv_y = to_display_xy(map_recv_gx, map_recv_gy)
    _shot_x, _shot_y = to_display_xy(map_shot_sx, map_shot_sy)

    fig.add_trace(go.Scattergl(
        x=_recv_x, y=_recv_y, mode="markers",
        marker=dict(size=3, color="lightgray"),
        name="Receiver stations", hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergl(
        x=_shot_x, y=_shot_y, mode="markers",
        marker=dict(size=6, color="steelblue"),
        name="Shot points",
        customdata=map_shot_fldr,
        hovertemplate=f"FFID %{{customdata}}<br>{_x_label}=%{{x:.1f}}<br>{_y_label}=%{{y:.1f}}<extra></extra>",
    ))
    _cur_idx = np.where(map_shot_fldr == current_fldr)[0]
    if len(_cur_idx):
        _i = _cur_idx[0]
        _cur_x = float(_shot_x[_i])
        _cur_y = float(_shot_y[_i])
        fig.add_trace(go.Scattergl(
            x=[_cur_x], y=[_cur_y], mode="markers",
            marker=dict(size=16, color="red", symbol="star"),
            name=f"Current shot (FFID {current_fldr})",
            hovertemplate=f"Current shot: FFID {current_fldr}<br>{_x_label}=%{{x:.1f}}<br>{_y_label}=%{{y:.1f}}<extra></extra>",
        ))

    # Size the canvas from the X/Y (feet) extent ALWAYS -- regardless of
    # which coordinate system is currently displayed -- since this width
    # /height is reused as-is by the corrected map below (always X/Y),
    # and by this cell's own IL/XL mode too (via scaleanchor's 1:1 unit
    # scaling, a canvas sized for the X/Y aspect ratio still renders
    # IL/XL correctly, just not necessarily filling every pixel).
    _all_x = np.concatenate([map_recv_gx, map_shot_sx])
    _all_y = np.concatenate([map_recv_gy, map_shot_sy])
    _x_range = _all_x.max() - _all_x.min()
    _y_range = _all_y.max() - _all_y.min()
    _max_dim = 900
    if _x_range >= _y_range:
        map_width, map_height = _max_dim, max(300, int(_max_dim * _y_range / _x_range))
    else:
        map_height, map_width = _max_dim, max(300, int(_max_dim * _x_range / _y_range))

    fig.update_layout(
        title="Acquisition geometry",
        xaxis_title=_x_label, yaxis_title=_y_label,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=map_width, height=map_height,
        # Legend outside the plot area (not overlaid on the data), placed
        # horizontally along the top instead of Plotly's default side
        # column.
        legend=dict(
            itemsizing="constant", orientation="h",
            yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
    )

    # Lazy (callable, not computed eagerly): kaleido's PNG export takes
    # ~1.5-3s per call, and this cell re-runs on every slider drag (the
    # current-shot star moves) -- computing it unconditionally on every
    # render would make browsing the survey noticeably laggier. Deferred
    # until the button is actually clicked instead.
    def _make_fig_png():
        return fig.to_image(format="png", width=map_width, height=map_height, scale=2)

    mo.vstack([
        mo.ui.plotly(
            fig,
            config={
                "displayModeBar": True,
                "toImageButtonOptions": {"filename": "acquisition_geometry", "format": "png", "scale": 2},
            },
        ),
        mo.download(data=_make_fig_png, filename="acquisition_geometry.png", mimetype="image/png", label="💾 Save geometry map"),
    ])
    return map_height, map_width


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Spatial misfit map (find bad shots by location, not just FFID)

    Every shot plotted at its actual position, colored by its audit
    misfit (Hilbert pick vs. LMO, same numbers as the audit table
    above) — a heat map showing *where* bad shots sit, not just which
    FFIDs. If bad shots cluster in one area (survey edge, a specific
    terrain feature, etc.) that's a very different story than if
    they're scattered randomly. Run the audit above first; shots
    without enough picks to trust are shown as small gray dots instead
    of a color.
    """)
    return


@app.cell
def _(audit, map_shot_fldr, map_shot_sx, map_shot_sy, min_picks_lmo, mo, np):
    if audit is None:
        mo.stop(True, mo.md(
            "*Run the full-survey geometry audit above first -- this map "
            "needs its per-shot misfit values.*"
        ))

    # audit["fldr"] is shot_list (sorted unique FLDRs); map_shot_fldr comes
    # from an independent whole-survey header scan -- searchsorted joins
    # them by FLDR rather than assuming matching order/coverage.
    _pos = np.searchsorted(audit["fldr"], map_shot_fldr)
    _pos = np.clip(_pos, 0, len(audit["fldr"]) - 1)
    _matched = audit["fldr"][_pos] == map_shot_fldr
    _valid_pick_count = np.where(_matched, audit["n_picks"][_pos], 0)
    _misfit = np.where(_matched, audit["mean_absmisfit_ms"][_pos], np.nan)

    _is_scored = _matched & (_valid_pick_count >= min_picks_lmo.value)

    misfit_map_fldr_scored = map_shot_fldr[_is_scored]
    misfit_map_sx_scored = map_shot_sx[_is_scored]
    misfit_map_sy_scored = map_shot_sy[_is_scored]
    misfit_map_values_scored = _misfit[_is_scored]
    misfit_map_fldr_unscored = map_shot_fldr[~_is_scored]
    misfit_map_sx_unscored = map_shot_sx[~_is_scored]
    misfit_map_sy_unscored = map_shot_sy[~_is_scored]
    return (
        misfit_map_fldr_scored,
        misfit_map_fldr_unscored,
        misfit_map_sx_scored,
        misfit_map_sx_unscored,
        misfit_map_sy_scored,
        misfit_map_sy_unscored,
        misfit_map_values_scored,
    )


@app.cell
def _(
    coord_display,
    effective_threshold,
    go,
    map_height,
    map_recv_gx,
    map_recv_gy,
    map_width,
    misfit_map_fldr_scored,
    misfit_map_fldr_unscored,
    misfit_map_sx_scored,
    misfit_map_sx_unscored,
    misfit_map_sy_scored,
    misfit_map_sy_unscored,
    misfit_map_values_scored,
    mo,
    np,
    to_display_xy,
):
    fig_misfit = go.Figure()
    _is_ilxl = coord_display.value == "IL/XL"
    _x_label, _y_label = ("XL", "IL") if _is_ilxl else ("X", "Y")

    _recv_x, _recv_y = to_display_xy(map_recv_gx, map_recv_gy)
    _un_x, _un_y = to_display_xy(misfit_map_sx_unscored, misfit_map_sy_unscored)
    _sc_x, _sc_y = to_display_xy(misfit_map_sx_scored, misfit_map_sy_scored)

    fig_misfit.add_trace(go.Scattergl(
        x=_recv_x, y=_recv_y, mode="markers",
        marker=dict(size=3, color="lightgray"),
        name="Receiver stations", hoverinfo="skip",
    ))

    if len(misfit_map_fldr_unscored):
        fig_misfit.add_trace(go.Scattergl(
            x=_un_x, y=_un_y, mode="markers",
            marker=dict(size=5, color="#BBBBBB"),
            name="Not enough picks to score",
            customdata=misfit_map_fldr_unscored,
            hovertemplate="FFID %{customdata}<br>not scored<extra></extra>",
        ))

    # Split at the audit's own flagging threshold: shots below it get the
    # continuous color gradient (capped at the threshold so a couple of
    # 1000+ ms outliers can't stretch the scale until every ordinary shot
    # looks identical); shots AT OR ABOVE it get pulled into their own
    # trace and drawn as red triangles so they're unmistakable at a
    # glance, not just a slightly darker dot in the same gradient.
    _cmax = max(float(effective_threshold), 1.0)
    _is_flagged = misfit_map_values_scored >= effective_threshold
    _ok_x, _ok_y = _sc_x[~_is_flagged], _sc_y[~_is_flagged]
    _ok_fldr, _ok_val = misfit_map_fldr_scored[~_is_flagged], misfit_map_values_scored[~_is_flagged]
    _flag_x, _flag_y = _sc_x[_is_flagged], _sc_y[_is_flagged]
    _flag_fldr, _flag_val = misfit_map_fldr_scored[_is_flagged], misfit_map_values_scored[_is_flagged]

    fig_misfit.add_trace(go.Scattergl(
        x=_ok_x, y=_ok_y, mode="markers",
        marker=dict(
            size=8, color=_ok_val,
            colorscale="YlOrRd", cmin=0, cmax=_cmax,
            colorbar=dict(title="Mean misfit (ms)"),
        ),
        name="Shot points (misfit)",
        customdata=np.stack([_ok_fldr, _ok_val], axis=1),
        hovertemplate="FFID %{customdata[0]:.0f}<br>misfit=%{customdata[1]:.1f} ms<extra></extra>",
    ))

    if len(_flag_x):
        fig_misfit.add_trace(go.Scattergl(
            x=_flag_x, y=_flag_y, mode="markers",
            marker=dict(size=13, color="#d1272f", symbol="triangle-up", line=dict(width=1, color="#5c0a0e")),
            name=f"Flagged (≥ {effective_threshold:.0f} ms)",
            customdata=np.stack([_flag_fldr, _flag_val], axis=1),
            hovertemplate="FFID %{customdata[0]:.0f}<br>misfit=%{customdata[1]:.1f} ms (flagged)<extra></extra>",
        ))

    fig_misfit.update_layout(
        title=f"Spatial misfit map",
        xaxis_title=_x_label, yaxis_title=_y_label,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=map_width, height=map_height,
        legend=dict(
            itemsizing="constant", orientation="h",
            yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
    )

    def _make_fig_misfit_png():
        return fig_misfit.to_image(format="png", width=map_width, height=map_height, scale=2)

    # Explicit counts so a "no colored shots" result is self-explanatory
    # instead of a silently empty-looking plot -- if this reads 0 scored,
    # the "Minimum picks required to trust a shot" slider in the audit
    # section above is almost certainly set higher than this survey's
    # shots can satisfy (or the audit hasn't been (re)run since the
    # header scan) rather than anything wrong with this map itself.
    _n_scored = len(misfit_map_sx_scored)
    _n_unscored = len(misfit_map_sx_unscored)
    mo.vstack([
        mo.md(
            f"**{_n_scored:,} shot(s) colored by misfit**, "
            f"**{_n_unscored:,} shown gray** (not enough picks to trust, "
            f"per the \"Minimum picks\" slider above)."
            + (" *No shots matched between the audit and the geometry scan "
               "-- try re-running the audit.*" if _n_scored == 0 and _n_unscored == 0 else "")
        ),
        mo.ui.plotly(
            fig_misfit,
            config={
                "displayModeBar": True,
                "toImageButtonOptions": {"filename": "spatial_misfit_map", "format": "png", "scale": 2},
            },
        ),
        mo.download(data=_make_fig_misfit_png, filename="spatial_misfit_map.png", mimetype="image/png", label="💾 Save misfit map"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Corrected acquisition geometry (preview of what would be exported)

    Same map, but reflecting the corrections and drops queued above:
    shots with a queued toggle combination are plotted at their
    **corrected** position (orange), shots queued to be **dropped** are
    marked with a red X at their original position (excluded from the
    export entirely), and everything else is unchanged (blue) — a
    preview of exactly what **Write corrected copy** above would
    produce, without having to run the full export first.
    """)
    return


@app.cell
def _(corrections, dropped_fldrs, map_shot_fldr, map_shot_sx, map_shot_sy, np):
    _drop_arr = np.array(sorted(dropped_fldrs), dtype=np.int64) if dropped_fldrs else np.array([], dtype=np.int64)
    _corr_fldr_set = set(corrections.keys())

    _is_dropped = np.isin(map_shot_fldr, _drop_arr) if _drop_arr.size else np.zeros(len(map_shot_fldr), dtype=bool)
    _is_corrected = np.isin(map_shot_fldr, np.array(sorted(_corr_fldr_set), dtype=np.int64)) if _corr_fldr_set else np.zeros(len(map_shot_fldr), dtype=bool)
    _is_corrected &= ~_is_dropped
    _is_unaffected = ~_is_dropped & ~_is_corrected

    # Only source-side corrections (negate_sx/negate_sy/swap_source_xy, or
    # a grid shift on the sx/sy field) move a shot's own point on this map
    # -- receiver-side corrections only affect that shot's receivers,
    # which aren't tracked per-shot here.
    _new_sx = map_shot_sx.copy()
    _new_sy = map_shot_sy.copy()
    for _fldr, _combo in corrections.items():
        _idx = np.where(map_shot_fldr == _fldr)[0]
        if len(_idx) == 0:
            continue
        _sx, _sy = _new_sx[_idx], _new_sy[_idx]
        if _combo.get("type") == "shift":
            _deltas = _combo["deltas"]
            _sx = _sx + _deltas.get("sx", 0.0)
            _sy = _sy + _deltas.get("sy", 0.0)
        else:
            if _combo.get("swap_source_xy"):
                _sx, _sy = _sy, _sx
            if _combo.get("negate_sx"):
                _sx = -_sx
            if _combo.get("negate_sy"):
                _sy = -_sy
        _new_sx[_idx] = _sx
        _new_sy[_idx] = _sy

    unaffected_fldr = map_shot_fldr[_is_unaffected]
    unaffected_sx = map_shot_sx[_is_unaffected]
    unaffected_sy = map_shot_sy[_is_unaffected]
    corrected_shot_fldr = map_shot_fldr[_is_corrected]
    corrected_shot_sx = _new_sx[_is_corrected]
    corrected_shot_sy = _new_sy[_is_corrected]
    dropped_shot_fldr = map_shot_fldr[_is_dropped]
    dropped_shot_sx = map_shot_sx[_is_dropped]
    dropped_shot_sy = map_shot_sy[_is_dropped]
    return (
        corrected_shot_fldr,
        corrected_shot_sx,
        corrected_shot_sy,
        dropped_shot_fldr,
        dropped_shot_sx,
        dropped_shot_sy,
        unaffected_fldr,
        unaffected_sx,
        unaffected_sy,
    )


@app.cell
def _(
    coord_display,
    corrected_shot_fldr,
    corrected_shot_sx,
    corrected_shot_sy,
    corrections,
    dropped_fldrs,
    dropped_shot_fldr,
    dropped_shot_sx,
    dropped_shot_sy,
    go,
    map_height,
    map_recv_gx,
    map_recv_gy,
    map_width,
    mo,
    to_display_xy,
    unaffected_fldr,
    unaffected_sx,
    unaffected_sy,
):
    fig_corrected = go.Figure()
    _is_ilxl = coord_display.value == "IL/XL"
    _x_label, _y_label = ("XL", "IL") if _is_ilxl else ("X", "Y")

    # Computed position reflects a queued correction correctly (unlike
    # a stored header value, which can't) -- so "corrected" shots show
    # their actual new IL/XL here, not a stale pre-correction one.
    _recv_x, _recv_y = to_display_xy(map_recv_gx, map_recv_gy)
    _un_x, _un_y = to_display_xy(unaffected_sx, unaffected_sy)
    _corr_x, _corr_y = to_display_xy(corrected_shot_sx, corrected_shot_sy)
    _drop_x, _drop_y = to_display_xy(dropped_shot_sx, dropped_shot_sy)

    fig_corrected.add_trace(go.Scattergl(
        x=_recv_x, y=_recv_y, mode="markers",
        marker=dict(size=3, color="lightgray"),
        name="Receiver stations", hoverinfo="skip",
    ))
    fig_corrected.add_trace(go.Scattergl(
        x=_un_x, y=_un_y, mode="markers",
        marker=dict(size=6, color="steelblue"),
        name="Shot points (unchanged)",
        customdata=unaffected_fldr,
        hovertemplate=f"FFID %{{customdata}}<br>{_x_label}=%{{x:.1f}}<br>{_y_label}=%{{y:.1f}}<extra></extra>",
    ))
    if len(_corr_x):
        fig_corrected.add_trace(go.Scattergl(
            x=_corr_x, y=_corr_y, mode="markers",
            marker=dict(size=9, color="orange"),
            name="Shot points (corrected position)",
            customdata=corrected_shot_fldr,
            hovertemplate=f"FFID %{{customdata}} (corrected)<br>{_x_label}=%{{x:.1f}}<br>{_y_label}=%{{y:.1f}}<extra></extra>",
        ))
    if len(_drop_x):
        fig_corrected.add_trace(go.Scattergl(
            x=_drop_x, y=_drop_y, mode="markers",
            marker=dict(size=11, color="red", symbol="x"),
            name="Dropped shots (excluded from export)",
            customdata=dropped_shot_fldr,
            hovertemplate=f"FFID %{{customdata}} (dropped)<br>{_x_label}=%{{x:.1f}}<br>{_y_label}=%{{y:.1f}}<extra></extra>",
        ))

    # Reuse the exact same canvas size as the acquisition-geometry map
    # above (not recomputed from this map's own data) so the two render
    # at identical resolution and are directly comparable side by side.
    fig_corrected.update_layout(
        title=(
            f"Corrected acquisition geometry -- {len(corrections)} "
            f"shot(s) repositioned, {len(dropped_fldrs)} dropped"
        ),
        xaxis_title=_x_label, yaxis_title=_y_label,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=map_width, height=map_height,
        # Legend outside the plot area (not overlaid on the data), placed
        # horizontally along the top instead of Plotly's default side
        # column.
        legend=dict(
            itemsizing="constant", orientation="h",
            yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
    )

    # Lazy, same reasoning as the acquisition-geometry map above -- avoid
    # paying kaleido's ~1.5-3s export cost on every reactive re-render.
    def _make_fig_corrected_png():
        return fig_corrected.to_image(format="png", width=map_width, height=map_height, scale=2)

    mo.vstack([
        mo.ui.plotly(
            fig_corrected,
            config={
                "displayModeBar": True,
                "toImageButtonOptions": {"filename": "corrected_acquisition_geometry", "format": "png", "scale": 2},
            },
        ),
        mo.download(
            data=_make_fig_corrected_png, filename="corrected_acquisition_geometry.png",
            mimetype="image/png", label="💾 Save corrected geometry map",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
