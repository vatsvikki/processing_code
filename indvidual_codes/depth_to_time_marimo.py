import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Depth ↔ time conversion of a seismic image

    **0.** Direction: depth → time, or time → depth &nbsp;→&nbsp; **1.** Input file, input parameters & corner points
    &nbsp;→&nbsp; **2.** Velocity (a velocity file, a constant or a velocity function) &nbsp;→&nbsp; **3.** Output
    parameters &nbsp;→&nbsp; **4.** IL / XL sections before and after (as the app's CDP Stack) + 💾 save figures
    &nbsp;→&nbsp; **5.** Convert and write.

    Depth z and two-way time t are linked by the interval velocity: **t(z) = 2 ∫₀ᶻ dz / V(z)**. Depth → time moves
    every depth sample to its time t(z); time → depth moves every time sample to the depth z(t) where t(z) = t. Every
    step has an **ⓘ What the parameters mean** panel. The answers are saved per file (`depth_to_time_settings.json`).
    """)
    return


@app.cell
def _():
    import datetime
    import json
    import os
    import re
    import struct
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    return Path, datetime, json, mo, np, os, plt, re, struct, time


@app.cell
def _(np, os, struct):
    # ---- SEG-Y reading / writing (no external SEG-Y package needed) ----------------------------------------------
    BPS = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}                    # bytes per sample of the sample-format codes read here

    def ibm_to_float(words):
        """IBM 32-bit floats (as uint32) -> float64."""
        u = np.asarray(words, dtype=np.uint32).astype(np.uint64)
        sign = np.where(u >> 31, -1.0, 1.0)
        expo = ((u >> 24) & 0x7F).astype(np.int64) - 64
        frac = (u & 0xFFFFFF).astype(np.float64) / float(1 << 24)
        return sign * frac * np.power(16.0, expo)

    def float_to_ibm(values):
        """float -> IBM 32-bit floats (uint32, to be written big-endian)."""
        v = np.asarray(values, dtype=np.float64)
        out = np.zeros(v.shape, dtype=np.uint32)
        nz = v != 0
        a = np.abs(v[nz])
        e = np.floor(np.log(a) / np.log(16.0)).astype(np.int64) + 1
        f = a / np.power(16.0, e)
        big = f >= 1.0                                     # (rounding at the edges)
        e[big] += 1
        f[big] /= 16.0
        m = np.round(f * (1 << 24)).astype(np.uint64)
        over = m >= (1 << 24)
        m[over] >>= 4
        e[over] += 1
        e = np.clip(e + 64, 0, 127).astype(np.uint64)
        sign = (v[nz] < 0).astype(np.uint64) << 31
        out[nz] = (sign | (e << 24) | (m & 0xFFFFFF)).astype(np.uint32)
        return out

    def segy_open(path):
        """Headers and layout of a SEG-Y file: text header lines, binary header values, trace count."""
        path = os.path.expanduser(str(path).strip())
        with open(path, "rb") as f:
            text = f.read(3200)
            binh = f.read(400)
        lines = [text[i:i + 80].decode("cp037", errors="replace") for i in range(0, 3200, 80)]
        if not any(l.strip().startswith("C") for l in lines):    # an ASCII text header
            lines = [text[i:i + 80].decode("ascii", errors="replace") for i in range(0, 3200, 80)]
        order = ">"
        fmt = struct.unpack(">H", binh[24:26])[0]
        if fmt not in BPS:
            fmt_le = struct.unpack("<H", binh[24:26])[0]
            if fmt_le in BPS:
                order, fmt = "<", fmt_le
            else:
                raise ValueError(f"sample format code {fmt} is not supported (IBM 1, IEEE 5, int32 2, int16 3, int8 8)")
        dt = struct.unpack(order + "H", binh[16:18])[0]
        ns = struct.unpack(order + "H", binh[20:22])[0]
        n_ext = max(struct.unpack(order + "h", binh[304:306])[0], 0)
        data_start = 3600 + 3200 * n_ext
        trace_bytes = 240 + ns * BPS[fmt]
        size = os.path.getsize(path)
        ntr = (size - data_start) // trace_bytes
        if ntr <= 0:
            raise ValueError("no traces found (check that this is a SEG-Y file)")
        return {"path": path, "lines": lines, "binary": binh, "order": order, "fmt": fmt, "dt": dt, "ns": ns,
                "n_ext": n_ext, "data_start": data_start, "trace_bytes": trace_bytes, "ntr": ntr, "size": size,
                "leftover": (size - data_start) % trace_bytes}

    def records(seg):
        """All trace records (240-byte header + samples) as a read-only array [ntr, trace_bytes]."""
        return np.memmap(seg["path"], dtype=np.uint8, mode="r", offset=seg["data_start"],
                         shape=(seg["ntr"], seg["trace_bytes"]))

    def header_word(rec, byte, kind="i4", order=">"):
        """One trace-header word of the records `rec` (byte = 1-based start byte)."""
        n = np.dtype(kind).itemsize
        return np.ascontiguousarray(rec[:, byte - 1:byte - 1 + n]).view(order + kind).ravel().astype(np.int64)

    def samples_of(seg, rec):
        """Samples of the records `rec` as float64 [n, ns]."""
        raw = np.ascontiguousarray(rec[:, 240:])
        o, fmt = seg["order"], seg["fmt"]
        if fmt == 1:
            return ibm_to_float(raw.view(o + "u4"))
        if fmt == 5:
            return raw.view(o + "f4").astype(np.float64)
        if fmt == 2:
            return raw.view(o + "i4").astype(np.float64)
        if fmt == 3:
            return raw.view(o + "i2").astype(np.float64)
        return raw.view("i1").astype(np.float64)

    def encode_samples(values, out_fmt):
        """float samples [n, ns] -> big-endian bytes [n, 4 * ns] (IBM or IEEE)."""
        if out_fmt == "IBM":
            words = float_to_ibm(values).astype(">u4")
        else:
            words = np.asarray(values, dtype=">f4")
        return np.ascontiguousarray(words).view(np.uint8).reshape(len(values), -1)

    return encode_samples, header_word, records, samples_of, segy_open


@app.cell
def _(np):
    # ---- velocity -> time-depth relation (tested in migration_velocity_marimo.py), and the depth-to-time mapping ----
    # ---- the velocity conversion (pure numpy) ---------------------------------------------------------------------
    # Internal units: metres and seconds, time = two-way time (TWT).
    VEL_UNITS = {"ft/s": 0.3048, "m/s": 1.0, "kft/s": 304.8, "km/s": 1000.0}        # -> m/s
    LEN_UNITS = {"ft": 0.3048, "m": 1.0}                                            # -> m
    TYPES = ["Interval", "RMS", "Average"]
    DOMAINS = ["Time (TWT, ms)", "Depth"]

    def _smooth(v, n):
        if n <= 1:
            return v
        k = np.ones(int(n)) / int(n)
        pad = int(n) // 2
        vp = np.pad(v, ((0, 0), (pad, int(n) - 1 - pad)), mode="edge")
        return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, vp)

    def to_interval(v, axis, vtype, domain, smooth=0, vmin_clip=1.0):
        """Velocity [ntr, ns] (m/s) on its axis (TWT s, or depth m) -> (interval velocity, TWT, depth) at every
        sample - the interval velocity of sample i holds between samples i-1 and i (an interval-velocity input is a
        sampled field: the velocity of each interval is the mean of its two ends - harmonic in depth). Also the count of samples where
        Dix / the derivative gave an impossible (negative) value, clipped."""
        v = np.maximum(np.asarray(v, np.float64), vmin_clip)
        q = np.asarray(axis, np.float64)
        dq = np.diff(q)
        bad = 0
        if domain == DOMAINS[0]:                               # time domain: q = TWT
            t = np.broadcast_to(q, v.shape)
            if vtype == "Interval":                            # a sampled field: the mean of the interval's ends
                vint = v.copy()
                vint[:, 1:] = 0.5 * (v[:, :-1] + v[:, 1:])
            elif vtype == "RMS":                               # Dix: Vint^2 = d(Vrms^2 t) / dt
                vint2 = np.empty_like(v)
                vint2[:, 0] = v[:, 0] ** 2
                vint2[:, 1:] = np.diff(v ** 2 * t, axis=1) / dq
                bad = int((vint2 <= 0).sum())
                vint = np.sqrt(np.maximum(vint2, vmin_clip ** 2))
            else:                                              # average: z = Vavg t / 2 -> Vint = d(Vavg t) / dt
                vint = np.empty_like(v)
                vint[:, 0] = v[:, 0]
                vint[:, 1:] = np.diff(v * t, axis=1) / dq
                bad = int((vint <= 0).sum())
                vint = np.maximum(vint, vmin_clip)
            vint = _smooth(vint, smooth)
            z = np.empty_like(v)
            z[:, 0] = vint[:, 0] * q[0] / 2
            z[:, 1:] = z[:, :1] + np.cumsum(vint[:, 1:] * dq / 2, axis=1)
            return vint, np.array(t), z, bad
        z = np.broadcast_to(q, v.shape)                        # depth domain: q = depth
        if vtype == "Interval":                                # a sampled field: the harmonic mean of the interval's
            vint = v.copy()                                    # ends (= the trapezoid rule on slowness)
            vint[:, 1:] = 2.0 / (1.0 / v[:, :-1] + 1.0 / v[:, 1:])
        elif vtype == "Average":                               # t = 2 z / Vavg -> Vint = 2 dz / dt
            tt = 2 * z / v
            vint = np.empty_like(v)
            vint[:, 0] = v[:, 0]
            dt_ = np.diff(tt, axis=1)
            bad = int((dt_ <= 0).sum())
            vint[:, 1:] = 2 * dq / np.maximum(dt_, 1e-9)
            vint = np.minimum(np.maximum(vint, vmin_clip), 1e5)
        else:                                                  # RMS against depth: layer stripping
            vint = np.empty_like(v)
            vint[:, 0] = v[:, 0]
            t_prev = 2 * q[0] / v[:, 0]
            acc = v[:, 0] ** 2 * t_prev                        # = integral of Vint^2 dt so far
            for i in range(1, v.shape[1]):
                c = 2 * dq[i - 1]
                a = v[:, i] ** 2
                b = acc - a * t_prev
                vi = (-b + np.sqrt(b * b + 4 * c * c * a)) / (2 * c)
                vint[:, i] = vi
                t_prev = t_prev + c / vi
                acc = acc + c * vi
        vint = _smooth(vint, smooth)
        t = np.empty_like(v)
        t[:, 0] = 2 * q[0] / vint[:, 0]
        t[:, 1:] = t[:, :1] + np.cumsum(2 * dq / vint[:, 1:], axis=1)
        return vint, t, np.array(z), bad

    BELOW = ["Hold the last velocity", "Continue the gradient of the last part", "Velocity table (depth, interval velocity)"]

    def time_at_depths(T, Z, vint, z_img, below, grad_len=1000.0, vcap=0.0, table=None):
        """TWT (s) at the image depths z_img (m) for every trace, from the velocity traces' TWT / depth / interval
        velocity at their samples (all in s, m, m/s). Below a velocity trace's last depth the interval velocity is
        held, continued with the gradient of its last grad_len metres (capped at vcap > 0), or read from the table
        [(depth m, Vint m/s), ...]. Returns (TWT [n, nz], the depth where each velocity trace ends)."""
        n = len(T)
        out = np.empty((n, len(z_img)))
        z_end = Z[:, -1].copy()
        for r in range(n):
            zr, tr, vr = Z[r], T[r], vint[r]
            inside = z_img <= zr[-1]
            out[r, inside] = np.interp(z_img[inside], np.concatenate([[0.0], zr]), np.concatenate([[0.0], tr]))
            if inside.all():
                continue
            zs = np.concatenate([[zr[-1]], z_img[~inside]])
            mid = 0.5 * (zs[:-1] + zs[1:])
            v_last = vr[-1]
            if below == BELOW[1]:
                v_up = np.interp(zr[-1] - grad_len, zr, vr)
                k = (v_last - v_up) / max(grad_len, 1e-6)
                v = np.maximum(v_last + k * (mid - zr[-1]), 0.5 * v_last)
            elif below == BELOW[2] and table is not None and len(table):
                v = np.interp(mid, table[:, 0], table[:, 1])
            else:
                v = np.full(len(mid), v_last)
            if vcap > 0:
                v = np.minimum(v, vcap)
            out[r, ~inside] = tr[-1] + np.cumsum(2 * np.diff(zs) / v)
        return out, z_end

    def depth_to_time(amp, t_img, _out_t, antialias=True, over=4):
        """Amplitudes [n, nz] at their TWT t_img [n, nz] -> [n, len(_out_t)] on the regular time axis _out_t (s).
        anti-alias: built on a 4x finer time axis, low-passed below 0.8 x the output Nyquist, then decimated."""
        n = len(amp)
        live = np.flatnonzero(np.abs(amp).max(axis=1) > 0)
        out = np.zeros((n, len(_out_t)))
        if not len(live):
            return out
        if not antialias:
            for r in live:
                out[r] = np.interp(_out_t, t_img[r], amp[r], left=0.0, right=0.0)
            return out
        dt = float(_out_t[1] - _out_t[0])
        fine = _out_t[0] + np.arange((len(_out_t) - 1) * over + 1) * (dt / over)
        a = np.empty((len(live), len(fine)))
        for j, r in enumerate(live):
            a[j] = np.interp(fine, t_img[r], amp[r], left=0.0, right=0.0)
        nfft = int(2 ** np.ceil(np.log2(len(fine) * 1.1)))
        f = np.fft.rfftfreq(nfft, dt / over)
        nyq = 0.5 / dt
        w = np.clip((0.9 * nyq - f) / (0.2 * nyq), 0.0, 1.0)      # 1 below 0.7 Nyquist, 0 above 0.9, linear between
        spec = np.fft.rfft(a, n=nfft, axis=1) * w
        out[live] = np.fft.irfft(spec, n=nfft, axis=1)[:, :len(fine):over]
        return out

    return BELOW, DOMAINS, LEN_UNITS, TYPES, VEL_UNITS, depth_to_time, time_at_depths, to_interval


@app.cell
def _(np, re):
    # ---- corner points: text-header parsing and the (IL, XL) <-> (X, Y) affine fit -------------------------------
    def corners_from_text(lines):
        """Grid corners written in the text header: a line with "CORNER", then "IL XL X Y" rows. [] if none."""
        out, on = [], False
        num = r"-?\d+(?:\.\d+)?"
        for line in lines:
            body = re.sub(r"^C\s*\d+\s?", "", line)
            if "CORNER" in body.upper():
                on = True
            if not on:
                continue
            nums = re.findall(num, body.split(":")[-1])
            if len(nums) >= 4 and re.fullmatch(r"[\s\d.\-]*", body.split(":")[-1]):
                out.append([float(v) for v in nums[-4:]])
            elif out:
                break
        return out[:8] if len(out) >= 3 else []

    def fit_affine(rows):
        """X = a0 + a1 IL + a2 XL, Y = b0 + b1 IL + b2 XL from >= 3 corner rows (least squares) - any grid rotation.
        Returns the fit with bin sizes, azimuths and residuals, or None."""
        pts = []
        for r in rows or []:
            try:
                il, xl, x, y = (float(r[k]) for k in ("IL", "XL", "X", "Y"))
            except (KeyError, TypeError, ValueError):
                continue
            pts.append((il, xl, x, y))
        if len(pts) < 3:
            return None
        p = np.array(pts)
        G = np.column_stack([np.ones(len(p)), p[:, 0], p[:, 1]])
        if np.linalg.matrix_rank(G) < 3:
            return None                                       # all corners on one line
        a, *_ = np.linalg.lstsq(G, p[:, 2], rcond=None)
        b, *_ = np.linalg.lstsq(G, p[:, 3], rcond=None)
        res = np.hypot(G @ a - p[:, 2], G @ b - p[:, 3])
        return {"a": a, "b": b, "rms": float(np.sqrt(np.mean(res ** 2))), "max": float(res.max()),
                "il_bin": float(np.hypot(a[1], b[1])), "xl_bin": float(np.hypot(a[2], b[2])),
                "il_az": float(np.degrees(np.arctan2(a[1], b[1])) % 360),     # direction of increasing IL (from north)
                "xl_az": float(np.degrees(np.arctan2(a[2], b[2])) % 360)}

    def ilxl_to_xy(fit, il, xl):
        a, b = fit["a"], fit["b"]
        return a[0] + a[1] * il + a[2] * xl, b[0] + b[1] * il + b[2] * xl

    def xy_to_ilxl(fit, x, y):
        """The inverse of ilxl_to_xy: (X, Y) -> fractional (IL, XL)."""
        a, b = fit["a"], fit["b"]
        m = np.array([[a[1], a[2]], [b[1], b[2]]])
        rhs = np.vstack([np.asarray(x, float) - a[0], np.asarray(y, float) - b[0]])
        il, xl = np.linalg.solve(m, rhs)
        return il, xl

    def corner_rows_of(table_value):
        rows = table_value.to_dict("records") if hasattr(table_value, "to_dict") else list(table_value or [])
        return rows

    return corner_rows_of, corners_from_text, fit_affine, ilxl_to_xy, xy_to_ilxl


@app.cell
def _(Path, json):
    SETTINGS_FILE = Path(__file__).resolve().with_name("depth_to_time_settings.json")

    def load_settings():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def save_settings(path, values):
        allv = load_settings()
        allv["last_file"] = path
        allv[path] = values
        SETTINGS_FILE.write_text(json.dumps(allv, indent=2, default=str))

    def explain(text):
        """The (i) panel of a step: what its parameters mean."""
        import marimo as _mo
        return _mo.accordion({"ⓘ  What the parameters mean": _mo.md(text)})

    return explain, load_settings, save_settings


@app.cell(hide_code=True)
def _(explain, load_settings, mo):
    # ---- 0. direction and 1. input file -------------------------------------------------------------------------------
    DIRS = ["Depth → time", "Time → depth"]
    _all = load_settings()
    _last = _all.get(_all.get("last_file", ""), {})
    direction = mo.ui.radio(DIRS, value=_last.get("direction", DIRS[0]) if _last.get("direction") in DIRS else DIRS[0],
                            label="**Convert**", inline=True)
    img_box = mo.ui.text(value=_all.get("last_file", ""), label="Input SEG-Y file", full_width=True)
    img_btn = mo.ui.run_button(label="📂 Load")
    mo.vstack([mo.md("## 0. What do you want to do?"), direction,
               explain("""
**Convert**

- **Depth → time** - the input is a depth image (e.g. a PSDM / RTM stack, vertical axis in ft or m); the output has
  two-way time (ms) as its vertical axis. Each depth sample z goes to the time t(z) = 2 ∫ dz / V.
- **Time → depth** - the input is a time image (a stack, a PSTM, vertical axis in ms of two-way time); the output has
  depth as its vertical axis. Each time sample t goes to the depth z where t(z) = t.

Both directions use the same velocity (step 2): the interval velocity in depth (or anything that converts to it).
Converting depth → time → depth with the same velocity gives back the input (to the sampling).
"""),
               mo.md("## 1. Input file and input parameters"), img_box, img_btn])
    return DIRS, direction, img_box, img_btn


@app.cell(hide_code=True)
def _(DIRS, direction, explain, img_box, img_btn, load_settings, mo, re, segy_open):
    mo.stop(not img_btn.value and not img_box.value.strip(), mo.md("_Type the file path and press **Load**._"))
    _err = ""
    try:
        img = segy_open(img_box.value)
    except Exception as _e:
        img, _err = None, f"{type(_e).__name__}: {_e}"
    mo.stop(img is None, mo.callout(mo.md(f"**Could not read the file** - {_err}"), kind="danger"))
    _all = load_settings()
    S = _all.get(img["path"], {})
    if not S:                    # a new file: start from the velocity / units / figure choices used for the last file
        _keep = ("vsource", "v_unit", "v_const", "v_depth_tab", "v_time_tab", "v_path", "vfile", "in_len", "in_il",
                 "in_xl", "fig_dir", "fig_fmt", "fig_dpi", "write_xy", "out_fmt", "antialias")
        S = {k: v for k, v in _all.get(_all.get("last_file", ""), {}).items() if k in _keep}
    d2t = direction.value == DIRS[0]
    _same = S.get("direction", DIRS[0]) == direction.value          # saved input parameters belong to one direction
    _txt = " ".join(img["lines"]).upper()
    _ft = bool(re.search(r"\b(FT|FEET|FOOT)\b|\d'", _txt))
    _what = next((l[3:].strip() for l in img["lines"] if "DESCRIPTION" in l.upper()), "")
    in_len = mo.ui.dropdown(["ft", "m"], value=S.get("in_len", "ft" if _ft else "m"),
                            label="Depth unit" + ("" if d2t else " (of the output)"))
    in_dz = mo.ui.number(value=S.get("in_dz", img["dt"] / 1000.0) if _same else img["dt"] / 1000.0, start=0, step=0.001,
                         label="Depth sample interval" if d2t else "Time sample interval (ms)")
    in_z0 = mo.ui.number(value=S.get("in_z0", 0.0) if _same else 0.0, step=0.001,
                         label="Depth of the first sample" if d2t else "Time of the first sample (ms)")
    in_il = mo.ui.number(value=S.get("in_il", 189), start=1, stop=237, label="Inline byte")
    in_xl = mo.ui.number(value=S.get("in_xl", 193), start=1, stop=237, label="Crossline byte")
    _warn = None
    if d2t and not re.search(r"DEPTH|\bFT\b|FEET|PSDM|RTM", _txt) and re.search(r"\bMS\b|TIME|PSTM", _txt):
        _warn = "The text header speaks of time - is this really a **depth** image? (else choose Time → depth)"
    if not d2t and re.search(r"DEPTH|PSDM|RTM", _txt):
        _warn = "The text header speaks of depth - is this really a **time** image? (else choose Depth → time)"
    mo.vstack([
        mo.md(f"**{img['ntr']:,} traces**, {img['ns']} samples, sample-interval field {img['dt']}"
              + (f" · *{_what}*" if _what else "")),
        mo.callout(mo.md(_warn), kind="warn") if _warn else mo.md(""),
        mo.accordion({"Text header": mo.md("```\n" + "\n".join(l.rstrip() for l in img["lines"]) + "\n```")}),
        mo.hstack([in_dz, in_z0, in_len], justify="start", gap=1, wrap=True),
        mo.hstack([in_il, in_xl], justify="start", gap=1, wrap=True),
        explain(f"""
- **{'Depth' if d2t else 'Time'} sample interval** - the vertical step between two samples of the input
  ({'in the depth unit' if d2t else 'in ms of two-way time'}). Filled from the binary header's sample-interval field ÷
  1000 (SEG-Y stores µs for time; for depth most software stores the step × 1000, e.g. 20000 = 20 ft).
- **{'Depth' if d2t else 'Time'} of the first sample** - where the first sample is (usually 0; non-zero for a datum
  shift or a cut record).
- **Depth unit** - ft or m{' of the input depth axis' if d2t else ' of the output depth axis'}; also the unit of the
  velocity file's depth axis and of the depth / velocity table.
- **Inline / crossline byte** - where the trace headers keep the IL / XL numbers (SEG-Y rev 1: 189 / 193). Used to
  pick the sections to plot, to match the velocity traces and with the corner points.
- **Corner points** - (IL, XL) ↔ (X, Y) of at least 3 grid corners (not on one line), read from the text header when
  it has them. They give the bin size and grid direction, place the sections on the map, match the image to a velocity
  file through X / Y, and can be written as CDP X / Y into the output.
"""),
    ])
    return S, d2t, img, in_dz, in_il, in_len, in_xl, in_z0


@app.cell(hide_code=True)
def _(LEN_UNITS, S, corners_from_text, d2t, header_word, img, in_dz, in_il, in_len, in_xl, in_z0, mo, np, records):
    # the grid of the file, its vertical axis and its corner points
    _r = records(img)
    img_ilv = header_word(_r, int(in_il.value), "i4", img["order"])
    img_xlv = header_word(_r, int(in_xl.value), "i4", img["order"])
    in_axis = float(in_z0.value) + np.arange(img["ns"]) * float(in_dz.value)           # depth unit, or ms
    in_axis_si = in_axis * LEN_UNITS[in_len.value] if d2t else in_axis / 1000.0         # m, or s
    in_ylabel = f"Depth ({in_len.value})" if d2t else "Time (ms)"
    _text = corners_from_text(img["lines"])
    corner_table = mo.ui.data_editor(
        S.get("corners") or [{"IL": r[0], "XL": r[1], "X": r[2], "Y": r[3]} for r in _text]
        or [{"IL": "", "XL": "", "X": "", "Y": ""}] * 4, label="Corner points (IL, XL ↔ X, Y)")
    mo.vstack([
        mo.md(f"{'Depth' if d2t else 'Time'} **{in_axis[0]:g} – {in_axis[-1]:,.0f} "
              f"{in_len.value if d2t else 'ms'}** · inline {img_ilv.min()} – {img_ilv.max()} · crossline "
              f"{img_xlv.min()} – {img_xlv.max()}"),
        mo.md("**Corner points**" + (" (read from the text header - check them)" if _text else " (type them)")),
        corner_table,
    ])
    return corner_table, img_ilv, img_xlv, in_axis, in_axis_si, in_ylabel


@app.cell(hide_code=True)
def _(corner_rows_of, corner_table, fit_affine, ilxl_to_xy, img_ilv, img_xlv, mo, np, plt):
    # the corner points: their fit (bin sizes, directions) and the survey map
    corners = corner_rows_of(corner_table.value)
    corner_fit = fit_affine(corners)
    if corner_fit is None:
        _out = mo.callout(mo.md("**Corner points:** give at least 3 corners (IL, XL, X, Y) that are not on one line - "
                                "they place the survey on the map and can be written into the output headers."),
                          kind="warn")
    else:
        _fig, _ax = plt.subplots(figsize=(6.5, 4.8))
        _ci = np.array([img_ilv.min(), img_ilv.min(), img_ilv.max(), img_ilv.max(), img_ilv.min()], float)
        _cx = np.array([img_xlv.min(), img_xlv.max(), img_xlv.max(), img_xlv.min(), img_xlv.min()], float)
        _ox, _oy = ilxl_to_xy(corner_fit, _ci, _cx)
        _ax.plot(_ox, _oy, color="#e34948", lw=1.6, label="survey (inline / crossline range)")
        for _r in corners:
            try:
                _ax.plot(float(_r["X"]), float(_r["Y"]), "o", color="#2a78d6")
                _ax.annotate(f"IL {int(float(_r['IL']))}\nXL {int(float(_r['XL']))}", (float(_r["X"]), float(_r["Y"])),
                             fontsize=7, xytext=(3, 3), textcoords="offset points")
            except (KeyError, TypeError, ValueError):
                pass
        _ax.set_aspect("equal")
        _ax.ticklabel_format(useOffset=False, style="plain")
        _ax.tick_params(labelsize=7)
        _ax.legend(fontsize=7)
        _ax.set_title("Corner points", fontsize=10)
        _fig.tight_layout()
        _out = mo.vstack([mo.callout(mo.md(
            f"Corner points OK: bin **{corner_fit['il_bin']:.1f}** (inline step) × **{corner_fit['xl_bin']:.1f}** "
            f"(crossline step), inline direction {corner_fit['il_az']:.1f}° from north, misfit {corner_fit['rms']:.2f}"),
            kind="success"), _fig])
    _out
    return corner_fit, corners


@app.cell(hide_code=True)
def _(S, VEL_UNITS, explain, in_len, mo):
    # ---- 2. velocity ------------------------------------------------------------------------------------------------
    VSOURCES = ["Velocity file (SEG-Y)", "Constant velocity", "Velocity function: depth - interval velocity",
                "Velocity function: time (TWT ms) - RMS velocity"]
    _saved = S.get("vsource", VSOURCES[0])
    _saved = VSOURCES[0] if str(_saved).startswith("Velocity model SEG-Y") else _saved
    vsource = mo.ui.dropdown(VSOURCES, value=_saved if _saved in VSOURCES else VSOURCES[0],
                             label="Velocity for the conversion")
    v_unit = mo.ui.dropdown(list(VEL_UNITS), value=S.get("v_unit", "ft/s" if in_len.value == "ft" else "m/s"),
                            label="Velocity unit")
    v_const = mo.ui.number(value=S.get("v_const", 10000.0), start=0, step=0.001, label="Velocity")
    _dz = [{"depth": 0, "velocity": 5000}, {"depth": 6000, "velocity": 8500}, {"depth": 15000, "velocity": 12000},
           {"depth": 40000, "velocity": 16000}]
    v_depth_tab = mo.ui.data_editor(S.get("v_depth_tab", _dz),
                                    label=f"depth ({in_len.value}) - interval velocity (linear between rows)")
    _tz = [{"time": 0, "velocity": 5000}, {"time": 2000, "velocity": 7000}, {"time": 5000, "velocity": 10000},
           {"time": 8000, "velocity": 12000}]
    v_time_tab = mo.ui.data_editor(S.get("v_time_tab", _tz), label="TWT (ms) - RMS velocity (linear between rows)")
    v_path = mo.ui.text(value=S.get("v_path", ""), label="Velocity file (SEG-Y)", full_width=True)
    v_btn = mo.ui.run_button(label="📂 Load velocity file")
    mo.vstack([mo.md("## 2. Velocity for the conversion"), mo.hstack([vsource, v_unit], justify="start", gap=1),
               explain("""
**Velocity for the conversion** - the velocity that links depth and time (the one the depth migration used, for an
image from a depth migration - for a TTI / VTI model its vertical velocity):

- **Velocity file (SEG-Y)** - a velocity model, one trace per IL / XL. Its own parameters (below, after loading):
  - *Velocity type* - **Interval** (velocity of each layer / sample: tomography, FWI and migration models),
    **RMS** (stacking / time-migration velocities; turned into interval velocity with Dix), **Average** (z / t).
  - *Vertical domain* - is the model sampled in **depth** or in **time** (TWT)?
  - *Sample interval / first sample / depth unit* - its vertical axis (from its header; check them).
  - *Inline / crossline byte* - where its IL / XL are.
  - *Match image and velocity traces by* - **corner points** (image IL/XL → X/Y with the image's corners → velocity
    IL/XL with the velocity's corners; right also when the two grids are numbered differently) or the **same
    inline / crossline numbers** (only when both files use the same grid).
  - *Below the velocity file's last sample* - the image may go deeper than the model: **hold the last velocity**, or
    **continue the gradient** of its last part (over the depth given). Below the model the result is only as good as
    this choice (gray in the velocity sections of step 4).
  - *Velocity file corner points* - its (IL, XL) ↔ (X, Y), read from its text header.
- **Constant velocity** - one velocity everywhere: t = 2 z / V (quick look only).
- **Velocity function: depth - interval velocity** - a table you type: interval velocity at some depths, linear in
  between, held above the first and below the last row; the same for every trace.
- **Velocity function: time - RMS velocity** - a table of TWT (ms) and RMS velocity (e.g. from velocity analysis),
  turned into interval velocity with Dix; the same for every trace.
- **Velocity unit** - the unit of the velocity values (file, constant or table): ft/s, m/s, kft/s, km/s.

A velocity file is checked: a file with negative values (a seismic image, not a velocity) is refused.
"""),])
    return VSOURCES, v_btn, v_const, v_depth_tab, v_path, v_time_tab, v_unit, vsource


@app.cell(hide_code=True)
def _(BELOW, DOMAINS, S, TYPES, VSOURCES, corners_from_text, in_len, mo, re, segy_open, v_btn, v_path, vsource):
    # the velocity file and its parameters (asked only when the velocity comes from a file)
    vel_file, vel_err = None, ""
    if vsource.value == VSOURCES[0] and v_path.value.strip():
        try:
            vel_file = segy_open(v_path.value)
        except Exception as _e:
            vel_err = f"{type(_e).__name__}: {_e}"
    _lines = vel_file["lines"] if vel_file else []
    _txt = " ".join(_lines).upper()
    _dom = DOMAINS[0] if re.search(r"\bTIME\b|\bMS\b", _txt) and not re.search(r"\bDEPTH\b|\bFT\b|\bFEET\b", _txt) \
        else DOMAINS[1]
    _typ = "RMS" if "RMS" in _txt else ("Average" if "AVERAGE" in _txt else "Interval")
    _s = S.get("vfile", {}) if S.get("v_path") == v_path.value else {}
    v_type = mo.ui.dropdown(TYPES, value=_s.get("type", _typ), label="Velocity type")
    v_domain = mo.ui.dropdown(DOMAINS, value=_s.get("domain", _dom), label="Vertical domain")
    v_dz = mo.ui.number(value=_s.get("dz", (vel_file["dt"] / 1000.0) if vel_file else 0.0), start=0, step=0.001,
                        label="Sample interval (ms for time / depth unit)")
    v_z0 = mo.ui.number(value=_s.get("z0", 0.0), step=0.001, label="First sample at")
    v_len = mo.ui.dropdown(["ft", "m"], value=_s.get("len", in_len.value), label="Depth unit")
    v_il = mo.ui.number(value=_s.get("il", 189), start=1, stop=237, label="Inline byte")
    v_xl = mo.ui.number(value=_s.get("xl", 193), start=1, stop=237, label="Crossline byte")
    VMATCH = ["Corner points (image IL/XL -> X/Y -> velocity IL/XL)", "Same inline / crossline numbers"]
    v_match = mo.ui.dropdown(VMATCH, value=_s.get("match", VMATCH[0]), label="Match image and velocity traces by")
    v_below = mo.ui.dropdown(BELOW[:2], value=_s.get("below", BELOW[0]) if _s.get("below") in BELOW[:2] else BELOW[0],
                             label="Below the velocity file's last sample")
    v_grad = mo.ui.number(value=_s.get("grad", 1000.0), start=0, step=0.001,
                          label="(gradient taken over the last ... depth units)")
    _text = corners_from_text(_lines)
    v_corner_table = mo.ui.data_editor(
        _s.get("corners") or [{"IL": r[0], "XL": r[1], "X": r[2], "Y": r[3]} for r in _text]
        or [{"IL": "", "XL": "", "X": "", "Y": ""}] * 4, label="Velocity file corner points (IL, XL ↔ X, Y)")
    v_corner_src = "read from its text header - check them" if _text and not _s.get("corners") else (
        "saved" if _s.get("corners") else "type them")
    return VMATCH, v_below, v_corner_src, v_corner_table, v_domain, v_dz, v_grad, v_il, v_len, v_match, v_type, v_xl, \
        v_z0, vel_err, vel_file


@app.cell(hide_code=True)
def _(VSOURCES, mo, v_below, v_btn, v_const, v_corner_src, v_corner_table, v_depth_tab, v_domain, v_dz, v_grad, v_il,
      v_len, v_match, v_path, v_time_tab, v_type, v_xl, v_z0, vel_err, vel_file, vsource):
    # only the inputs of the chosen velocity
    if vsource.value == VSOURCES[0]:
        _info = []
        if vel_err:
            _info.append(mo.callout(mo.md(f"**Could not read the velocity file** - {vel_err}"), kind="danger"))
        elif vel_file is not None:
            _what = next((l[3:].strip() for l in vel_file["lines"] if "DESCRIPTION" in l.upper()), "")
            _info += [mo.md(f"**{vel_file['ntr']:,} traces**, {vel_file['ns']} samples, sample-interval field "
                            f"{vel_file['dt']}" + (f" · *{_what}*" if _what else "")),
                      mo.accordion({"Velocity file text header": mo.md(
                          "```\n" + "\n".join(l.rstrip() for l in vel_file["lines"]) + "\n```")}),
                      mo.md("**Velocity file parameters** (guessed from its header - check them)"),
                      mo.hstack([v_type, v_domain, v_dz, v_z0, v_len], justify="start", gap=1, wrap=True),
                      mo.hstack([v_il, v_xl, v_match], justify="start", gap=1, wrap=True),
                      mo.hstack([v_below, v_grad], justify="start", gap=1, wrap=True),
                      mo.md(f"**Velocity file corner points** ({v_corner_src})"), v_corner_table]
        else:
            _info.append(mo.md("_Type the velocity file path and press **Load velocity file**._"))
        _out = mo.vstack([v_path, v_btn] + _info)
    else:
        _out = {VSOURCES[1]: v_const, VSOURCES[2]: v_depth_tab, VSOURCES[3]: v_time_tab}[vsource.value]
    _out
    return


@app.cell(hide_code=True)
def _(DOMAINS, LEN_UNITS, VEL_UNITS, VMATCH, VSOURCES, corner_fit, corner_rows_of, fit_affine, header_word, ilxl_to_xy,
      img_ilv, img_xlv, in_len, mo, np, records, samples_of, time_at_depths, to_interval, v_below, v_const,
      v_corner_table, v_depth_tab, v_domain, v_dz, v_grad, v_il, v_len, v_match, v_time_tab, v_type, v_unit, v_xl, v_z0,
      vel_file, vsource, xy_to_ilxl):
    # the depth -> TWT relation of any image traces at any depths: twt_at(trace indices, depths m) -> [n, nz] s, and
    # the interval velocity there: vint_at(...) -> [n, nz] in the velocity unit (NaN below the end of a velocity file)
    def _rows(tab, a, b):
        rows = tab.value.to_dict("records") if hasattr(tab.value, "to_dict") else list(tab.value or [])
        out = []
        for r in rows:
            try:
                out.append((float(r[a]), float(r[b])))
            except (KeyError, TypeError, ValueError):
                pass
        return np.array(sorted(out)) if out else np.zeros((0, 2))

    vu = VEL_UNITS[v_unit.value]
    vmsg, vel_near, vel_axis_si, _kind, _tab, _rmsT, _rmsZ, _rmsV = "", None, None, "", None, None, None, None
    if vsource.value == VSOURCES[1]:
        _vc = float(v_const.value) * vu
        mo.stop(_vc <= 0, mo.callout(mo.md("The velocity must be > 0."), kind="danger"))
        _kind, vmsg = "const", f"constant {float(v_const.value):g} {v_unit.value}"
    elif vsource.value == VSOURCES[2]:
        _tab = _rows(v_depth_tab, "depth", "velocity")
        mo.stop(len(_tab) < 1 or (_tab[:, 1] <= 0).any(), mo.callout(mo.md("Give at least one row, velocities > 0."),
                                                                      kind="danger"))
        _tab = np.column_stack([_tab[:, 0] * LEN_UNITS[in_len.value], _tab[:, 1] * vu])
        _kind, vmsg = "depth table", f"depth function of {len(_tab)} rows"
    elif vsource.value == VSOURCES[3]:
        _t = _rows(v_time_tab, "time", "velocity")
        mo.stop(len(_t) < 1 or (_t[:, 1] <= 0).any(), mo.callout(mo.md("Give at least one row, velocities > 0."),
                                                                  kind="danger"))
        _tq = np.arange(0.0, max(_t[-1, 0], 30000.0) + 1, 2.0)
        _rv, _rT, _rZ, _ = to_interval((np.interp(_tq, _t[:, 0], _t[:, 1]) * vu)[None], _tq / 1000.0, "RMS", DOMAINS[0])
        _rmsV, _rmsT, _rmsZ = _rv[0], _rT[0], _rZ[0]
        _kind, vmsg = "rms table", f"RMS function of {len(_t)} rows (Dix to interval)"
    else:
        mo.stop(vel_file is None, mo.callout(mo.md("Load the velocity file (step 2)."), kind="warn"))
        _r = records(vel_file)
        _pick = np.linspace(0, vel_file["ntr"] - 1, min(vel_file["ntr"], 200)).astype(np.int64)
        _sv = samples_of(vel_file, _r[_pick])
        _live = _sv[_sv != 0]
        _neg = float((_live < 0).mean()) if _live.size else 0.0
        mo.stop(_neg > 0.01, mo.callout(mo.md(
            f"**This is not a velocity file** - {_neg:.0%} of its samples are negative (a velocity never is). It is "
            "probably a seismic image. Load the velocity model."), kind="danger"))
        _vil = header_word(_r, int(v_il.value), "i4", vel_file["order"])
        _vxl = header_word(_r, int(v_xl.value), "i4", vel_file["order"])
        from scipy.spatial import cKDTree as _KD
        if v_match.value == VMATCH[0]:
            _vfit = fit_affine(corner_rows_of(v_corner_table.value))
            mo.stop(corner_fit is None or _vfit is None, mo.callout(mo.md(
                "Matching by corner points needs valid corner points for the image (step 1) and the velocity file "
                "(step 2) - or match by the same inline / crossline numbers."), kind="danger"))
            _x, _y = ilxl_to_xy(corner_fit, img_ilv.astype(float), img_xlv.astype(float))
            _qi, _qx = xy_to_ilxl(_vfit, _x, _y)
            _dist, vel_near = _KD(np.column_stack([_vil, _vxl]).astype(float)).query(np.column_stack([_qi, _qx]))
        else:
            _dist, vel_near = _KD(np.column_stack([_vil, _vxl]).astype(float)).query(
                np.column_stack([img_ilv, img_xlv]).astype(float))
        _is_t = v_domain.value == DOMAINS[0]
        _ax = float(v_z0.value) + np.arange(vel_file["ns"]) * float(v_dz.value or vel_file["dt"] / 1000.0)
        vel_axis_si = _ax / 1000.0 if _is_t else _ax * LEN_UNITS[v_len.value]
        _lo, _hi = (np.percentile(_live, [1, 99]) if _live.size else (0, 0))
        _kind = "file"
        vmsg = (f"velocity file: {v_type.value} in {'time' if _is_t else 'depth'}, {_ax[0]:g} – {_ax[-1]:g} "
                f"{'ms' if _is_t else v_len.value}, values {_lo:,.0f} – {_hi:,.0f} {v_unit.value} · "
                f"{float((_dist < 0.5).mean()):.0%} of the image traces on a velocity trace"
                + (" (⚠ part of the image is outside the velocity file - the nearest trace is used)"
                   if _dist.max() > 1.5 else ""))

    def _file_vint(idx):
        v = samples_of(vel_file, records(vel_file)[vel_near[idx]]) * vu
        return to_interval(v, vel_axis_si, v_type.value, v_domain.value)

    def twt_at(idx, zs):
        """TWT (s) at the depths zs (m, increasing) for the image traces idx."""
        zs = np.asarray(zs, float)
        if _kind == "const":
            return np.broadcast_to(2 * zs / _vc, (len(idx), len(zs)))
        if _kind == "depth table":
            z = np.concatenate([[0.0], zs]) if zs[0] > 0 else zs
            v = np.interp(z, _tab[:, 0], _tab[:, 1])
            t = np.concatenate([[0.0], np.cumsum(np.diff(z) * (1 / v[:-1] + 1 / v[1:]))])[-len(zs):]
            return np.broadcast_to(t, (len(idx), len(zs)))
        if _kind == "rms table":
            return np.broadcast_to(np.interp(zs, _rmsZ, _rmsT), (len(idx), len(zs)))
        vint, t, z, _ = _file_vint(idx)
        return time_at_depths(t, z, vint, zs, v_below.value, float(v_grad.value) * LEN_UNITS[v_len.value])[0]

    def vint_at(idx, zs):
        """Interval velocity (velocity unit) at the depths zs (m) - NaN below the end of a velocity file."""
        zs = np.asarray(zs, float)
        if _kind == "const":
            return np.full((len(idx), len(zs)), _vc / vu)
        if _kind == "depth table":
            return np.broadcast_to(np.interp(zs, _tab[:, 0], _tab[:, 1]) / vu, (len(idx), len(zs)))
        if _kind == "rms table":
            j = np.clip(np.searchsorted(_rmsZ, zs), 0, len(_rmsV) - 1)
            return np.broadcast_to(_rmsV[j] / vu, (len(idx), len(zs)))
        vint, t, z, _ = _file_vint(idx)
        out = np.empty((len(idx), len(zs)))
        for r in range(len(idx)):
            j = np.clip(np.searchsorted(z[r], zs), 0, vint.shape[1] - 1)
            out[r] = np.where(zs <= z[r, -1], vint[r, j], np.nan)
        return out / vu

    def model_end(idx):
        """Depth (m) where the velocity file ends for the traces idx (inf for a constant / a table)."""
        if _kind != "file":
            return np.full(len(idx), np.inf)
        return _file_vint(idx)[2][:, -1]

    mo.callout(mo.md(f"Velocity: **{vmsg}**"), kind="info")
    return model_end, twt_at, vint_at, vmsg


@app.cell(hide_code=True)
def _(LEN_UNITS, S, d2t, explain, img, in_axis_si, in_len, mo, np, twt_at):
    # ---- 3. output parameters -----------------------------------------------------------------------------------------
    _mid = np.array([img["ntr"] // 2])
    _same = S.get("direction") == ("Depth → time" if d2t else "Time → depth")
    if d2t:
        _deep_t = float(twt_at(_mid, in_axis_si)[0, -1] * 1000)
        _dflt_step, _dflt_end = 4.0, float(np.ceil(_deep_t / 100.0) * 100)
    else:                                                  # the depth of the last input time, from the velocity
        _zp = np.arange(0.0, 80000.0, 10.0)
        _tp = twt_at(_mid, _zp)[0]
        _zend = float(np.interp(in_axis_si[-1], _tp, _zp)) / LEN_UNITS[in_len.value]
        _dflt_step = 10.0 if in_len.value == "ft" else 5.0
        _dflt_end = float(np.ceil(_zend / 500.0) * 500)
    out_step = mo.ui.number(value=S.get("out_step", _dflt_step) if _same else _dflt_step, start=0, step=0.001,
                            label="Output time sample interval (ms)" if d2t else f"Output depth sample interval ({in_len.value})")
    out_end = mo.ui.number(value=S.get("out_end", _dflt_end) if _same else _dflt_end, start=0, step=0.001,
                           label="Output record length (ms)" if d2t else f"Output maximum depth ({in_len.value})")
    out_fmt = mo.ui.dropdown(["IBM", "IEEE"], value=S.get("out_fmt", "IBM"), label="Sample format")
    antialias = mo.ui.checkbox(value=S.get("antialias", True), label="anti-alias filter")
    mo.vstack([mo.md("## 3. Output parameters"),
               mo.callout(mo.md(f"The deepest input sample is at about **{_deep_t:,.0f} ms** TWT (middle trace)"
                                if d2t else f"The last input time is at about **{_zend:,.0f} {in_len.value}** depth "
                                            "(middle trace)"), kind="info"),
               mo.hstack([out_step, out_end, out_fmt, antialias], justify="start", gap=1, wrap=True),
               explain(f"""
- **Output {'time' if d2t else 'depth'} sample interval** - the vertical step of the output
  ({'ms of two-way time' if d2t else in_len.value}). Smaller = finer and larger files. A good choice keeps about the
  same resolution: Δt ≈ 2 Δz / V ({'e.g. 20 ft at 10 000 ft/s ≈ 4 ms' if d2t else 'e.g. 4 ms at 10 000 ft/s ≈ 20 ft'}).
- **Output {'record length (ms)' if d2t else 'maximum depth'}** - where the output stops; filled with
  {'the time of the deepest input sample' if d2t else 'the depth of the last input sample'} (middle of the survey,
  rounded up). Longer = zeros below the data; shorter = the deep part is cut.
- **Sample format** - IBM float (classic SEG-Y, read by everything) or IEEE float (exact, rev 1 and later).
- **Anti-alias filter** - when the output sampling is coarser than the input's (after the stretch), frequencies above
  the new Nyquist are removed first (the data are built on a 4× finer axis, low-passed at 0.7–0.9 of the output Nyquist,
  then decimated). Keep it on unless the output is much finer than the input.
"""),
               ])
    return antialias, out_end, out_fmt, out_step


@app.cell(hide_code=True)
def _(LEN_UNITS, d2t, in_len, np, out_end, out_step):
    # the output vertical axis
    out_axis = np.arange(0.0, float(out_end.value) + 1e-9, float(out_step.value))     # ms, or depth unit
    out_axis_si = out_axis / 1000.0 if d2t else out_axis * LEN_UNITS[in_len.value]       # s, or m
    out_ylabel = "Time (ms)" if d2t else f"Depth ({in_len.value})"
    return out_axis, out_axis_si, out_ylabel


@app.cell
def _(d2t, depth_to_time, in_axis_si, np, out_axis_si, twt_at, vint_at):
    # the conversion of a block of traces, both directions: convert_block(idx, amp) -> output [n, len(out_axis)];
    # velocity_on_output(idx) -> interval velocity on the output axis (NaN below the end of a velocity file)
    def convert_block(idx, amp, antialias=True):
        if d2t:                                            # every depth sample -> its TWT, then onto the time axis
            pos = np.array(twt_at(idx, in_axis_si))
            return depth_to_time(amp, pos, out_axis_si, antialias)
        twt = np.array(twt_at(idx, out_axis_si))           # TWT of every output depth
        pos = np.empty((len(idx), len(in_axis_si)))
        for r in range(len(idx)):                          # every time sample -> its depth z(t)
            pos[r] = np.interp(in_axis_si, twt[r], out_axis_si)
            past = in_axis_si > twt[r, -1]
            if past.any():                                 # (beyond the output: keep increasing, it lands outside)
                slope = (out_axis_si[-1] - out_axis_si[-2]) / max(twt[r, -1] - twt[r, -2], 1e-12)
                pos[r, past] = out_axis_si[-1] + (in_axis_si[past] - twt[r, -1]) * slope
        return depth_to_time(amp, pos, out_axis_si, antialias)

    def velocity_on_output(idx):
        if not d2t:
            return np.array(vint_at(idx, out_axis_si))
        vz = np.array(vint_at(idx, in_axis_si))
        tz = np.array(twt_at(idx, in_axis_si))
        out = np.full((len(idx), len(out_axis_si)), np.nan)
        for r in range(len(idx)):
            ok = np.isfinite(vz[r])
            if ok.any():
                out[r] = np.where(out_axis_si <= tz[r][ok][-1], np.interp(out_axis_si, tz[r][ok], vz[r][ok]), np.nan)
        return out

    return convert_block, velocity_on_output


@app.cell
def _(np):
    # ---- the figures of the app's CDP Stack (functions/plotting.py: plot_stack_sections, plot_velocity_sections,
    # plot_overlay_sections) - same layout, sizes, colour maps, clip, titles - with the vertical axis as a parameter
    from matplotlib.figure import Figure as _Figure
    import matplotlib as _mpl

    def _two_sections(fig_height):
        fig = _Figure(figsize=(12.0, float(fig_height)), facecolor="white")
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [60, 40]}, sharey=True)
        return fig, axes

    def _extent(vary_vals, y_max):
        a, b = float(vary_vals[0]), float(vary_vals[-1])
        if a == b:
            a, b = a - 0.5, b + 0.5
        return [a, b, y_max, 0]

    def _empty(ax, fixed_label, fixed_val):
        ax.text(0.5, 0.5, f"No traces at {fixed_label}={int(fixed_val)}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{fixed_label}={int(fixed_val)} (empty)")

    def _inset_colorbar(fig, ax, im):
        cax = ax.inset_axes([0.55, 0.06, 0.4, 0.035])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)

    def _limits(axes, y_max, xlim_il=None, xlim_xl=None, ylim=None):
        """The axis windows: x of the IL section (its XL range), x of the XL section (its IL range), the vertical
        range (shared) - each None = all."""
        axes[0].set_ylim(*(ylim[::-1] if ylim else (y_max, 0)))
        if xlim_il:
            axes[0].set_xlim(*xlim_il)
        if xlim_xl:
            axes[1].set_xlim(*xlim_xl)

    def plot_sections(il_result, xl_result, il_value, xl_value, y_max, y_label, *, clip_pct=98.0, fig_height=7.5,
                      xlim_il=None, xlim_xl=None, ylim=None):
        """IL section (fixed IL, varying XL) left, XL section right, same vertical axis - as the app's stack."""
        fig, axes = _two_sections(fig_height)
        for ax, result, fixed_label, fixed_val, vary_label in ((axes[0], il_result, "IL", il_value, "XL"),
                                                                (axes[1], xl_result, "XL", xl_value, "IL")):
            if result is None:
                _empty(ax, fixed_label, fixed_val)
                continue
            vary_vals, section = result
            vclip = np.percentile(np.abs(section), clip_pct) or 1.0
            ax.imshow(section.T, aspect="auto", cmap="gray", vmin=-vclip, vmax=vclip, extent=_extent(vary_vals, y_max))
            ax.set_title(f"{fixed_label}={int(fixed_val)}, {len(vary_vals)} CDPs")
            ax.set_xlabel(vary_label)
        _limits(axes, y_max, xlim_il, xlim_xl, ylim)
        axes[0].set_ylabel(y_label)
        fig.tight_layout()
        return fig

    def plot_velocity_sections(il_result, xl_result, il_value, xl_value, y_max, y_label, kind="interval", *,
                               fig_height=7.5, xlim_il=None, xlim_xl=None, ylim=None):
        """Velocity along the same two lines (gray = beyond the velocity model's own range)."""
        cmap = _mpl.colormaps["viridis"].copy()
        cmap.set_bad(color="lightgray")
        fig, axes = _two_sections(fig_height)
        im = None
        for ax, result, fixed_label, fixed_val, vary_label in ((axes[0], il_result, "IL", il_value, "XL"),
                                                                (axes[1], xl_result, "XL", xl_value, "IL")):
            if result is None:
                _empty(ax, fixed_label, fixed_val)
                continue
            vary_vals, section = result
            im = ax.imshow(section.T, aspect="auto", cmap=cmap, extent=_extent(vary_vals, y_max))
            ax.set_title(f"{fixed_label}={int(fixed_val)} {kind} velocity")
            ax.set_xlabel(vary_label)
        _limits(axes, y_max, xlim_il, xlim_xl, ylim)
        axes[0].set_ylabel(y_label)
        if im is not None:
            _inset_colorbar(fig, axes[0], im)
        fig.tight_layout()
        return fig

    def plot_overlay_sections(il_result, xl_result, il_value, xl_value, y_max, y_label, kind="interval", *,
                              clip_pct=98.0, fig_height=7.5, xlim_il=None, xlim_xl=None, ylim=None):
        """The velocity in colour (half transparent) over the grayscale section; where the model has no data the
        section shows through."""
        cmap = _mpl.colormaps["viridis"].copy()
        cmap.set_bad(alpha=0.0)
        fig, axes = _two_sections(fig_height)
        im = None
        for ax, result, fixed_label, fixed_val, vary_label in ((axes[0], il_result, "IL", il_value, "XL"),
                                                                (axes[1], xl_result, "XL", xl_value, "IL")):
            if result is None:
                _empty(ax, fixed_label, fixed_val)
                continue
            vary_vals, sec, vel = result
            extent = _extent(vary_vals, y_max)
            vclip = np.percentile(np.abs(sec), clip_pct) or 1.0
            ax.imshow(sec.T, aspect="auto", cmap="gray", vmin=-vclip, vmax=vclip, extent=extent)
            im = ax.imshow(vel.T, aspect="auto", cmap=cmap, alpha=0.5, extent=extent)
            ax.set_title(f"{fixed_label}={int(fixed_val)} + {kind} vel, {len(vary_vals)} CDPs")
            ax.set_xlabel(vary_label)
        _limits(axes, y_max, xlim_il, xlim_xl, ylim)
        axes[0].set_ylabel(y_label)
        if im is not None:
            _inset_colorbar(fig, axes[0], im)
        fig.tight_layout()
        return fig

    return plot_overlay_sections, plot_sections, plot_velocity_sections


@app.cell(hide_code=True)
def _(d2t, explain, img, img_ilv, img_xlv, mo, np, records):
    # ---- 4. plot: an IL section and an XL section, before and after (as the app's CDP Stack) ------------------------
    # default lines: through the middle of the LIVE data - every 4th inline / crossline looked at (1/16 of the traces,
    # 3 samples each), so the large file is not read in full
    _r = records(img)
    _bps = (img["trace_bytes"] - 240) // img["ns"]
    _sub = np.flatnonzero((img_ilv % 4 == 0) & (img_xlv % 4 == 0))
    if len(_sub) < 50:
        _sub = np.arange(0, img["ntr"], max(1, img["ntr"] // 20000))
    _blk = np.asarray(_r[_sub])
    _live = np.zeros(len(_sub), bool)
    for _k in (img["ns"] // 4, img["ns"] // 2, 3 * img["ns"] // 4):
        _live |= _blk[:, 240 + _bps * _k:240 + _bps * (_k + 1)].any(axis=1)
    if not _live.any():
        _live[:] = True

    def _mid(v, steps):
        m = float(np.median(v[_sub][_live]))
        return int(steps[np.argmin(np.abs(steps - m))])

    _ilu, _xlu = np.unique(img_ilv), np.unique(img_xlv)
    sec_il = mo.ui.slider(steps=[int(x) for x in _ilu], value=_mid(img_ilv, _ilu), show_value=True, include_input=True,
                          full_width=True, debounce=True, label="Fixed IL (inline section)")
    sec_xl = mo.ui.slider(steps=[int(x) for x in _xlu], value=_mid(img_xlv, _xlu), show_value=True, include_input=True,
                          full_width=True, debounce=True, label="Fixed XL (crossline section)")
    clip_pct = mo.ui.number(value=98.0, start=80, stop=100, step=0.5, label="Amplitude clip percentile")
    fig_height = mo.ui.number(value=7.0, start=3, stop=14, step=0.5, label="Figure height, inches")
    # axis windows (0 / 0 = all): x of each panel and the vertical range of the input / output figures
    _num = lambda lab: mo.ui.number(value=0, step=1, label=lab)
    lim_xl0, lim_xl1 = _num("IL section: from XL"), _num("to XL")
    lim_il0, lim_il1 = _num("XL section: from IL"), _num("to IL")
    lim_in0, lim_in1 = _num(f"Input figure: from ({'depth' if d2t else 'ms'})"), _num("to")
    lim_out0, lim_out1 = _num(f"Output figures: from ({'ms' if d2t else 'depth'})"), _num("to")
    mo.vstack([mo.md("## 4. Sections before and after"), sec_il, sec_xl,
               mo.hstack([clip_pct, fig_height], justify="start", gap=1),
               mo.md("**Axis limits** (0 and 0 = the whole axis)"),
               mo.hstack([lim_xl0, lim_xl1, lim_il0, lim_il1], justify="start", gap=1, wrap=True),
               mo.hstack([lim_in0, lim_in1, lim_out0, lim_out1], justify="start", gap=1, wrap=True),
               explain("""
- **Fixed IL** - the inline section shown (all crosslines of that inline), left panel of every figure.
- **Fixed XL** - the crossline section shown (all inlines of that crossline), right panel. Both start in the middle of
  the live data.
- **Amplitude clip percentile** - the gray scale is clipped at this percentile of |amplitude| of each section (98 =
  the strongest 2 % saturate); lower = weak events stand out more. The same clip rule as the app's CDP Stack.
- **Figure height** - height of the figures in inches; the width is 12 inches. 7 (the default) gives the same size as
  the CDP Stack sections of the app's Flow (its figure height 7 → 12 × 7.5 inches, 110 dpi).
- **Axis limits** - the window of the figures (0 and 0 = the whole axis): **IL section: from / to XL** = the horizontal
  range of the left panel (the inline section runs along the crosslines); **XL section: from / to IL** = the horizontal
  range of the right panel; **Input figure: from / to** = the vertical range of the input sections (depth or ms);
  **Output figures: from / to** = the vertical range of the output, velocity and overlay sections. The figure size stays
  the same - the window is stretched to fill it. Saved figures show the same window.

The figures: the **input** sections, the **output** sections, the **interval velocity** used along the two lines on the
output axis (gray = below the velocity file: there the conversion uses the 'below' choice of step 2), the output with
the velocity on top, and the depth ↔ time curve with the two lines on the survey map. **💾 Save figures** (below them)
writes them as files.
"""),
               ])
    return clip_pct, fig_height, lim_il0, lim_il1, lim_in0, lim_in1, lim_out0, lim_out1, lim_xl0, lim_xl1, sec_il, sec_xl


@app.cell(hide_code=True)
def _(antialias, clip_pct, convert_block, corner_fit, d2t, fig_height, ilxl_to_xy, img, img_ilv, img_xlv, in_axis,
      in_axis_si, in_len, in_ylabel, lim_il0, lim_il1, lim_in0, lim_in1, lim_out0, lim_out1, lim_xl0, lim_xl1, mo,
      model_end, np, out_axis, out_axis_si, out_ylabel, plot_overlay_sections, plot_sections, plot_velocity_sections,
      plt, records, samples_of, sec_il, sec_xl, time, twt_at, v_unit, velocity_on_output):
    _t0 = time.perf_counter()

    def _line(fixed, value, along):
        sel = np.flatnonzero(fixed == int(value))
        if not len(sel):
            return None
        sel = sel[np.argsort(along[sel])]
        amp = samples_of(img, records(img)[sel])
        out = convert_block(sel, amp, antialias.value)
        return along[sel], amp, out, velocity_on_output(sel), sel

    _il = _line(img_ilv, sec_il.value, img_xlv)
    _xl = _line(img_xlv, sec_xl.value, img_ilv)
    _secs = time.perf_counter() - _t0
    _cp, _fh = float(clip_pct.value), float(fig_height.value) + 0.5

    def _rng(a, b):                                          # (from, to) or None when both are 0 / empty
        a, b = float(a.value or 0), float(b.value or 0)
        return None if a == 0 and b == 0 else (min(a, b), max(a, b))

    _lim = dict(xlim_il=_rng(lim_xl0, lim_xl1), xlim_xl=_rng(lim_il0, lim_il1))
    _lim_in, _lim_out = dict(_lim, ylim=_rng(lim_in0, lim_in1)), dict(_lim, ylim=_rng(lim_out0, lim_out1))
    _pick = lambda r, i: None if r is None else (r[0], r[i])
    _in_name, _out_name = ("Depth", "Time") if d2t else ("Time", "Depth")
    _f_in = plot_sections(_pick(_il, 1), _pick(_xl, 1), sec_il.value, sec_xl.value, float(in_axis[-1]), in_ylabel,
                          clip_pct=_cp, fig_height=_fh, **_lim_in)
    _f_out = plot_sections(_pick(_il, 2), _pick(_xl, 2), sec_il.value, sec_xl.value, float(out_axis[-1]), out_ylabel,
                           clip_pct=_cp, fig_height=_fh, **_lim_out)
    _f_vel = plot_velocity_sections(_pick(_il, 3), _pick(_xl, 3), sec_il.value, sec_xl.value, float(out_axis[-1]),
                                    out_ylabel, fig_height=_fh, **_lim_out)
    _ov = lambda r: None if r is None else (r[0], r[2], r[3])
    _f_ov = plot_overlay_sections(_ov(_il), _ov(_xl), sec_il.value, sec_xl.value, float(out_axis[-1]), out_ylabel,
                                  clip_pct=_cp, fig_height=_fh, **_lim_out)
    # depth <-> time used (middle trace of each line) and where the lines are
    _f_tz, (_a1, _a2) = plt.subplots(1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [40, 60]})
    _du = 0.3048 if in_len.value == "ft" else 1.0
    _zs = in_axis_si if d2t else out_axis_si
    for _r, _lab in ((_il, f"IL={sec_il.value}"), (_xl, f"XL={sec_xl.value}")):
        if _r is not None:
            _k = _r[4][len(_r[4]) // 2]
            _a1.plot(np.asarray(twt_at(np.array([_k]), _zs))[0] * 1000, _zs / _du, label=f"{_lab} (middle trace)")
            _zend = float(model_end(np.array([_k]))[0])
            if np.isfinite(_zend) and _zend < _zs[-1]:
                _a1.axhline(_zend / _du, color="gray", ls="--", lw=0.8)
    _a1.invert_yaxis()
    _a1.set_xlabel("Time (ms)")
    _a1.set_ylabel(f"Depth ({in_len.value})")
    _a1.set_title("Depth ↔ time used (dashed: end of the velocity file)")
    _a1.grid(alpha=0.3)
    _a1.legend(fontsize=8)
    if corner_fit is not None:
        _ci = np.array([img_ilv.min(), img_ilv.min(), img_ilv.max(), img_ilv.max(), img_ilv.min()], float)
        _cx = np.array([img_xlv.min(), img_xlv.max(), img_xlv.max(), img_xlv.min(), img_xlv.min()], float)
        _ox, _oy = ilxl_to_xy(corner_fit, _ci, _cx)
        _a2.plot(_ox, _oy, color="gray", lw=1)
        for _r, _c, _lab in ((_il, "tab:red", f"IL={sec_il.value}"), (_xl, "tab:blue", f"XL={sec_xl.value}")):
            if _r is not None:
                _lx, _ly = ilxl_to_xy(corner_fit, img_ilv[_r[4]].astype(float), img_xlv[_r[4]].astype(float))
                _a2.plot(_lx, _ly, color=_c, lw=2, label=_lab)
        _a2.set_aspect("equal")
        _a2.ticklabel_format(useOffset=False, style="plain")
        _a2.legend(fontsize=8)
        _a2.set_title("Sections on the survey (corner points)")
    else:
        _a2.axis("off")
    _f_tz.tight_layout()

    def _png(fig, dpi=110):                                  # as the app: PNG at 110 dpi (keeps every output small)
        import io as _io
        from matplotlib.backends.backend_agg import FigureCanvasAgg as _Canvas
        _Canvas(fig)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        return buf.getvalue()

    _tag = f"IL {sec_il.value} / XL {sec_xl.value}"
    sec_figs = [(f"{_in_name} sections (input) {_tag}", _png(_f_in), _f_in),
                (f"{_out_name} sections (output) {_tag}", _png(_f_out), _f_out),
                (f"Velocity sections {_tag}", _png(_f_vel), _f_vel),
                (f"{_out_name} + velocity {_tag}", _png(_f_ov), _f_ov),
                ("Depth - time and location", _png(_f_tz), _f_tz)]
    mo.md(f"IL {sec_il.value} and XL {sec_xl.value} converted in {_secs:.1f} s · velocity in {v_unit.value}; gray "
          "in the velocity sections = below the velocity file (the 'below' choice of step 2)")
    return (sec_figs,)


@app.cell(hide_code=True)
def _(mo, sec_figs):
    mo.vstack([mo.md(f"#### {sec_figs[0][0]}"), mo.image(sec_figs[0][1], style={"width": "100%", "height": "auto"})])
    return


@app.cell(hide_code=True)
def _(mo, sec_figs):
    mo.vstack([mo.md(f"#### {sec_figs[1][0]}"), mo.image(sec_figs[1][1], style={"width": "100%", "height": "auto"})])
    return


@app.cell(hide_code=True)
def _(mo, sec_figs):
    mo.vstack([mo.md(f"#### {sec_figs[2][0]}"), mo.image(sec_figs[2][1], style={"width": "100%", "height": "auto"})])
    return


@app.cell(hide_code=True)
def _(mo, sec_figs):
    mo.vstack([mo.md(f"#### {sec_figs[3][0]}"), mo.image(sec_figs[3][1], style={"width": "100%", "height": "auto"})])
    return


@app.cell(hide_code=True)
def _(mo, sec_figs):
    mo.vstack([mo.md(f"#### {sec_figs[4][0]}"), mo.image(sec_figs[4][1], style={"width": "100%", "height": "auto"})])
    return


@app.cell(hide_code=True)
def _(Path, S, explain, img, mo):
    # ---- 💾 save the figures (as the app's Save figures) ----------------------------------------------------------------
    fig_dir = mo.ui.text(value=S.get("fig_dir", str(Path(__file__).resolve().with_name("output"))),
                         label="Folder for the figures", full_width=True)
    fig_fmt = mo.ui.dropdown(["png", "pdf", "svg"], value=S.get("fig_fmt", "png"), label="Format")
    fig_dpi = mo.ui.number(value=S.get("fig_dpi", 150), start=50, stop=600, step=50, label="PNG dpi")
    fig_btn = mo.ui.run_button(label="💾  Save figures")
    mo.vstack([mo.md("#### 💾 Save figures"), mo.hstack([fig_btn, fig_fmt, fig_dpi], justify="start", align="end", gap=1.2),
               fig_dir, explain(f"""
- **💾 Save figures** - writes the five figures above, as shown (the IL / XL and settings on screen), into the folder:
  `{Path(img['path']).stem}_<figure>_IL<il>_XL<xl>.<format>`. An existing file of the same name is replaced.
- **Format** - png (image), pdf or svg (vector graphics: sharp at any zoom, for reports).
- **PNG dpi** - resolution of PNG files (150 = sharp on screen, 300 = print).
- **Folder** - created if needed (default: `output` next to this notebook, as in the app).
""")])
    return fig_btn, fig_dir, fig_dpi, fig_fmt


@app.cell(hide_code=True)
def _(Path, fig_btn, fig_dir, fig_dpi, fig_fmt, img, mo, re, sec_figs):
    mo.stop(not fig_btn.value)
    _dir = Path(fig_dir.value.strip() or ".").expanduser()
    _dir.mkdir(parents=True, exist_ok=True)
    _written = []
    for _title, _png_bytes, _fig in sec_figs:
        _slug = re.sub(r"[^A-Za-z0-9]+", "_", _title).strip("_").lower()
        _p = _dir / f"{Path(img['path']).stem}_{_slug}.{fig_fmt.value}"
        _fig.savefig(_p, format=fig_fmt.value, dpi=int(fig_dpi.value or 150), facecolor="white")
        _written.append(str(_p))
    mo.callout(mo.md(f"Saved {len(_written)} figure(s):  \n" + "  \n".join(f"`{_w}`" for _w in _written)),
               kind="success")
    return


@app.cell(hide_code=True)
def _(Path, S, d2t, explain, img, mo):
    # ---- 5. write ---------------------------------------------------------------------------------------------------
    _p = Path(img["path"])
    _dflt = str(_p.with_name(f"{_p.stem}_{'time' if d2t else 'depth'}.sgy"))
    _saved = S.get("out_path", "")
    out_path = mo.ui.text(value=_saved if _saved and _saved.endswith("_time.sgy" if d2t else "_depth.sgy") else _dflt,
                          label="Output SEG-Y file", full_width=True)
    write_xy = mo.ui.checkbox(value=S.get("write_xy", False),
                              label="write CDP X / Y (bytes 181 / 185, scalar -100) from the corner points")
    overwrite = mo.ui.checkbox(value=False, label="overwrite if it exists")
    write_btn = mo.ui.run_button(label="💾 Convert and write")
    mo.vstack([mo.md("## 5. Convert and write"), out_path, write_xy, mo.hstack([overwrite, write_btn], justify="start", gap=1),
               explain(f"""
- **Output SEG-Y file** - full path of the {'time' if d2t else 'depth'} image (on the machine the notebook runs on);
  never the input file. Written as `<name>.part` and renamed when complete.
- **Write CDP X / Y** - put the X / Y of every trace, computed from its IL / XL with the corner points, into bytes
  181 / 185 (scalar -100 in byte 71: two decimals). Off = the input's trace headers are kept as they are.
- **Overwrite** - replace an existing output file of that name.
- **Convert and write** - converts every trace (the whole file, in blocks, with a progress bar) with the settings
  above; the binary header gets the new sample count and interval ({'µs' if d2t else 'the depth step × 1000'}), the
  text header says what was done. Your answers are saved for this file.
""")])
    return out_path, overwrite, write_btn, write_xy


@app.cell(hide_code=True)
def _(DIRS, S, antialias, convert_block, corner_fit, corners, d2t, datetime, direction, encode_samples, fig_dir,
      fig_dpi, fig_fmt, ilxl_to_xy, img, img_ilv, img_xlv, in_dz, in_il, in_len, in_xl, in_z0, mo, np, os, out_axis,
      out_end, out_fmt, out_path, out_step, overwrite, records, samples_of, save_settings, time, v_below, v_const,
      v_corner_table, v_depth_tab, v_domain, v_dz, v_grad, v_il, v_len, v_match, v_path, v_time_tab, v_type, v_unit,
      v_xl, v_z0, vmsg, vsource, write_btn, write_xy):
    mo.stop(not write_btn.value)
    _dest = os.path.abspath(os.path.expanduser(out_path.value.strip()))
    mo.stop(_dest == img["path"], mo.callout(mo.md("The output must be a new file."), kind="danger"))
    mo.stop(os.path.exists(_dest) and not overwrite.value,
            mo.callout(mo.md(f"`{_dest}` exists - tick **overwrite** or change the name."), kind="warn"))
    mo.stop(write_xy.value and corner_fit is None, mo.callout(mo.md("Writing X / Y needs valid corner points."),
                                                             kind="danger"))

    def _tab(t):
        return t.value.to_dict("records") if hasattr(t.value, "to_dict") else list(t.value or [])

    save_settings(img["path"], {
        "direction": direction.value, "in_len": in_len.value, "in_dz": in_dz.value, "in_z0": in_z0.value,
        "in_il": in_il.value, "in_xl": in_xl.value, "corners": corners, "vsource": vsource.value, "v_unit": v_unit.value,
        "v_const": v_const.value, "v_depth_tab": _tab(v_depth_tab), "v_time_tab": _tab(v_time_tab),
        "v_path": v_path.value,
        "vfile": {"type": v_type.value, "domain": v_domain.value, "dz": v_dz.value, "z0": v_z0.value, "len": v_len.value,
                  "il": v_il.value, "xl": v_xl.value, "match": v_match.value, "below": v_below.value,
                  "grad": v_grad.value, "corners": _tab(v_corner_table)},
        "out_step": out_step.value, "out_end": out_end.value, "out_fmt": out_fmt.value, "antialias": antialias.value,
        "out_path": out_path.value, "write_xy": write_xy.value, "fig_dir": fig_dir.value, "fig_fmt": fig_fmt.value,
        "fig_dpi": fig_dpi.value})
    _ns = len(out_axis)
    _dtf = int(round(float(out_step.value) * 1000))          # µs for time; the depth step x 1000 for depth
    mo.stop(_ns > 65535 or _dtf > 65535, mo.callout(mo.md("Too many samples / too long a sample interval for SEG-Y."),
                                                    kind="danger"))
    import shutil as _shutil
    _need = img["ntr"] * (240 + 4 * _ns) + 3600
    _free = _shutil.disk_usage(os.path.dirname(_dest) or ".").free
    mo.stop(_free < _need * 1.02, mo.callout(mo.md(f"Not enough disk space: needs {_need / 1e9:.1f} GB, "
                                                   f"{_free / 1e9:.1f} GB free."), kind="danger"))
    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _unit = "MS" if d2t else in_len.value.upper()
    _new = [f"{'DEPTH TO TIME' if d2t else 'TIME TO DEPTH'} CONVERSION {_now}",
            f"INPUT: {os.path.basename(img['path'])}"[:76], f"VELOCITY: {vmsg.upper()}"[:76],
            f"{'TWT' if d2t else 'DEPTH'}, DZ {float(out_step.value):g} {_unit}, {_ns} SAMPLES, {out_fmt.value} FLOAT"
            + ("" if d2t else f", DT FIELD = DZ X 1000"), "ORIGINAL TEXT HEADER:"]
    _old = [l[3:].rstrip() if l[:1] == "C" else l.rstrip() for l in img["lines"]]
    _cards = (_new + _old)[:39] + ["END TEXTUAL HEADER"]
    _text = "".join(f"C{_i + 1:2d} {_c}"[:80].ljust(80) for _i, _c in enumerate(_cards)).replace("–", "-").replace(
        "—", "-").encode("cp037", errors="replace")
    _bin = bytearray(img["binary"])
    if img["order"] == "<":
        for _o in (12, 16, 18, 20, 22, 24, 304):
            _bin[_o:_o + 2] = _bin[_o:_o + 2][::-1]
    for _o, _val in ((16, _dtf), (18, _dtf), (20, _ns), (22, _ns), (24, 1 if out_fmt.value == "IBM" else 5), (304, 0)):
        _bin[_o:_o + 2] = int(_val).to_bytes(2, "big")
    _ri = records(img)
    _t0 = time.perf_counter()
    _part = _dest + ".part"
    with open(_part, "wb") as _f:
        _f.write(_text)
        _f.write(bytes(_bin))
        with mo.status.progress_bar(total=img["ntr"], title=direction.value, show_eta=True, show_rate=True) as _bar:
            for _a in range(0, img["ntr"], 2000):
                _sel = np.arange(_a, min(_a + 2000, img["ntr"]))
                _block = np.array(_ri[_sel])
                _amp = samples_of(img, _block)
                _live = np.abs(_amp).max(axis=1) > 0
                _out = np.zeros((len(_sel), _ns))
                if _live.any():
                    _out[_live] = convert_block(_sel[_live], _amp[_live], antialias.value)
                _hdr = _block[:, :240].copy()
                if img["order"] == "<":
                    _hdr = _hdr.reshape(len(_sel), 60, 4)[:, :, ::-1].reshape(len(_sel), 240).copy()
                _hdr[:, 114:116] = np.frombuffer(_ns.to_bytes(2, "big"), np.uint8)
                _hdr[:, 116:118] = np.frombuffer(_dtf.to_bytes(2, "big"), np.uint8)
                _hdr[:, 108:110] = 0
                if write_xy.value:
                    _x, _y = ilxl_to_xy(corner_fit, img_ilv[_sel].astype(float), img_xlv[_sel].astype(float))
                    _hdr[:, 70:72] = np.frombuffer((-100).to_bytes(2, "big", signed=True), np.uint8)
                    _hdr[:, 180:184] = np.round(_x * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                    _hdr[:, 184:188] = np.round(_y * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                np.hstack([_hdr, encode_samples(_out, out_fmt.value)]).tofile(_f)
                _bar.update(increment=len(_sel))
    os.replace(_part, _dest)
    mo.callout(mo.md(f"✅ Wrote **{img['ntr']:,} traces** × {_ns} samples ({'TWT' if d2t else 'depth'} every "
                     f"{float(out_step.value):g} {'ms' if d2t else in_len.value}) to `{_dest}` "
                     f"({os.path.getsize(_dest) / 1e9:.2f} GB) in {time.perf_counter() - _t0:.0f} s. Answers saved."),
               kind="success")
    return


if __name__ == "__main__":
    app.run()
