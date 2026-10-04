import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Depth image → time image

    **1.** Input file, input parameters & corner points &nbsp;→&nbsp; **2.** Velocity file (or a constant / a
    velocity function) with its parameters & corner points &nbsp;→&nbsp;
    **3.** Output parameters &nbsp;→&nbsp; **4.** Plot IL / XL sections in depth and time &nbsp;→&nbsp; **5.** Convert and write.

    Every depth sample goes to its two-way time **t = 2 ∫ dz / V(z)**, so the conversion needs a velocity - give a
    constant, type a velocity function, or use a velocity SEG-Y (step 2). The answers are saved per file
    (`depth_to_time_settings.json`).
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

    return load_settings, save_settings


@app.cell(hide_code=True)
def _(load_settings, mo):
    # ---- 1. input file ------------------------------------------------------------------------------------------------
    img_box = mo.ui.text(value=load_settings().get("last_file", ""), label="Depth image (SEG-Y)", full_width=True)
    img_btn = mo.ui.run_button(label="📂 Load")
    mo.vstack([mo.md("## 1. Input file and input parameters"), img_box, img_btn])
    return img_box, img_btn


@app.cell(hide_code=True)
def _(img_box, img_btn, load_settings, mo, re, segy_open):
    mo.stop(not img_btn.value and not img_box.value.strip(), mo.md("_Type the file path and press **Load**._"))
    _err = ""
    try:
        img = segy_open(img_box.value)
    except Exception as _e:
        img, _err = None, f"{type(_e).__name__}: {_e}"
    mo.stop(img is None, mo.callout(mo.md(f"**Could not read the file** - {_err}"), kind="danger"))
    S = load_settings().get(img["path"], {})
    _txt = " ".join(img["lines"]).upper()
    _ft = bool(re.search(r"\b(FT|FEET|FOOT)\b|\d'", _txt))
    _what = next((l[3:].strip() for l in img["lines"] if "DESCRIPTION" in l.upper()), "")
    in_len = mo.ui.dropdown(["ft", "m"], value=S.get("in_len", "ft" if _ft else "m"), label="Depth unit")
    in_dz = mo.ui.number(value=S.get("in_dz", img["dt"] / 1000.0), start=0, step=0.001, label="Depth sample interval")
    in_z0 = mo.ui.number(value=S.get("in_z0", 0.0), step=0.001, label="Depth of the first sample")
    in_il = mo.ui.number(value=S.get("in_il", 189), start=1, stop=237, label="Inline byte")
    in_xl = mo.ui.number(value=S.get("in_xl", 193), start=1, stop=237, label="Crossline byte")
    mo.vstack([
        mo.md(f"**{img['ntr']:,} traces**, {img['ns']} samples, sample-interval field {img['dt']}"
              + (f" · *{_what}*" if _what else "")),
        mo.accordion({"Text header": mo.md("```\n" + "\n".join(l.rstrip() for l in img["lines"]) + "\n```")}),
        mo.hstack([in_dz, in_z0, in_len], justify="start", gap=1, wrap=True),
        mo.hstack([in_il, in_xl], justify="start", gap=1, wrap=True),
    ])
    return S, img, in_dz, in_il, in_len, in_xl, in_z0


@app.cell(hide_code=True)
def _(LEN_UNITS, S, corners_from_text, header_word, img, in_dz, in_il, in_len, in_xl, in_z0, mo, np, records):
    # the grid of the file and its corner points
    _r = records(img)
    img_ilv = header_word(_r, int(in_il.value), "i4", img["order"])
    img_xlv = header_word(_r, int(in_xl.value), "i4", img["order"])
    depth_axis = float(in_z0.value) + np.arange(img["ns"]) * float(in_dz.value)
    depth_m = depth_axis * LEN_UNITS[in_len.value]
    _text = corners_from_text(img["lines"])
    corner_table = mo.ui.data_editor(
        S.get("corners") or [{"IL": r[0], "XL": r[1], "X": r[2], "Y": r[3]} for r in _text]
        or [{"IL": "", "XL": "", "X": "", "Y": ""}] * 4, label="Corner points (IL, XL ↔ X, Y)")
    mo.vstack([
        mo.md(f"Depth **{depth_axis[0]:g} – {depth_axis[-1]:,.0f} {in_len.value}** · inline {img_ilv.min()} – "
              f"{img_ilv.max()} · crossline {img_xlv.min()} – {img_xlv.max()}"),
        mo.md("**Corner points**" + (" (read from the text header - check them)" if _text else " (type them)")),
        corner_table,
    ])
    return corner_table, depth_axis, depth_m, img_ilv, img_xlv


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
def _(S, VEL_UNITS, in_len, mo):
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
    mo.vstack([mo.md("## 2. Velocity for the conversion"), mo.hstack([vsource, v_unit], justify="start", gap=1)])
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
def _(DOMAINS, LEN_UNITS, VEL_UNITS, VMATCH, VSOURCES, corner_fit, corner_rows_of, depth_m, fit_affine, header_word,
      ilxl_to_xy, img_ilv, img_xlv, in_len, mo, np, records, samples_of, time_at_depths, to_interval, v_below, v_const,
      v_corner_table, v_depth_tab, v_domain, v_dz, v_grad, v_il, v_len, v_match, v_time_tab, v_type, v_unit, v_xl, v_z0,
      vel_file, vsource, xy_to_ilxl):
    # TWT (s) at every depth sample, for a set of image traces: twt_for(trace_indices) -> [n, ns]
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
    vmsg, twt_one, vel_near, vel_axis_si, vint_one = "", None, None, None, None
    if vsource.value == VSOURCES[1]:
        _v = float(v_const.value) * vu
        mo.stop(_v <= 0, mo.callout(mo.md("The velocity must be > 0."), kind="danger"))
        twt_one = 2 * depth_m / _v
        vint_one = np.full(len(depth_m), _v)
        vmsg = f"constant {float(v_const.value):g} {v_unit.value}"
    elif vsource.value == VSOURCES[2]:
        _t = _rows(v_depth_tab, "depth", "velocity")
        mo.stop(len(_t) < 1 or (_t[:, 1] <= 0).any(), mo.callout(mo.md("Give at least one row, velocities > 0."),
                                                                  kind="danger"))
        _z = np.concatenate([[0.0], depth_m]) if depth_m[0] > 0 else depth_m
        _v = np.interp(_z, _t[:, 0] * LEN_UNITS[in_len.value], _t[:, 1] * vu)
        _tt = np.concatenate([[0.0], np.cumsum(np.diff(_z) * (1 / _v[:-1] + 1 / _v[1:]))])    # 2 * trapezoid of 1/v
        twt_one = _tt[-len(depth_m):]
        vint_one = np.interp(depth_m, _t[:, 0] * LEN_UNITS[in_len.value], _t[:, 1] * vu)
        vmsg = f"depth function of {len(_t)} rows"
    elif vsource.value == VSOURCES[3]:
        _t = _rows(v_time_tab, "time", "velocity")
        mo.stop(len(_t) < 1 or (_t[:, 1] <= 0).any(), mo.callout(mo.md("Give at least one row, velocities > 0."),
                                                                  kind="danger"))
        _tq = np.arange(0.0, max(_t[-1, 0], 20000.0) + 1, 2.0)
        _vr = np.interp(_tq, _t[:, 0], _t[:, 1]) * vu
        _vint, _T, _Z, _bad = to_interval(_vr[None], _tq / 1000.0, "RMS", DOMAINS[0])
        twt_one = np.interp(depth_m, _Z[0], _T[0])
        vint_one = _vint[0][np.clip(np.searchsorted(_Z[0], depth_m), 0, _vint.shape[1] - 1)]
        vmsg = f"RMS function of {len(_t)} rows (Dix to interval)"
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
        vmsg = (f"velocity file: {v_type.value} in {'time' if _is_t else 'depth'}, {_ax[0]:g} – {_ax[-1]:g} "
                f"{'ms' if _is_t else v_len.value}, values {_lo:,.0f} – {_hi:,.0f} {v_unit.value} · "
                f"{float((_dist < 0.5).mean()):.0%} of the image traces on a velocity trace"
                + (" (⚠ part of the image is outside the velocity file - the nearest trace is used)"
                   if _dist.max() > 1.5 else ""))

    def twt_for(idx):
        """TWT (s) at every depth sample of the image traces idx."""
        if twt_one is not None:
            return np.broadcast_to(twt_one, (len(idx), len(depth_m)))
        v = samples_of(vel_file, records(vel_file)[vel_near[idx]]) * vu
        vint, t, z, _ = to_interval(v, vel_axis_si, v_type.value, v_domain.value)
        return time_at_depths(t, z, vint, depth_m, v_below.value, float(v_grad.value) * LEN_UNITS[v_len.value])[0]

    def vint_for(idx):
        """Interval velocity (velocity unit) at every depth sample of the image traces idx - NaN below the end of a
        velocity file (there the time comes from the 'below' choice, not from the file)."""
        if vint_one is not None:
            return np.broadcast_to(vint_one / vu, (len(idx), len(depth_m)))
        v = samples_of(vel_file, records(vel_file)[vel_near[idx]]) * vu
        vint, t, z, _ = to_interval(v, vel_axis_si, v_type.value, v_domain.value)
        out = np.empty((len(idx), len(depth_m)))
        for r in range(len(idx)):
            j = np.clip(np.searchsorted(z[r], depth_m), 0, vint.shape[1] - 1)
            out[r] = np.where(depth_m <= z[r, -1], vint[r, j], np.nan)
        return out / vu

    def vel_along(idx):
        """The velocity file's values for the image traces idx (None for a constant / a typed function)."""
        if vel_file is None or vel_near is None:
            return None
        return samples_of(vel_file, records(vel_file)[vel_near[idx]])

    _t_end = float(twt_for(np.array([len(img_ilv) // 2]))[0, -1] * 1000)
    mo.callout(mo.md(f"Velocity: **{vmsg}** · the deepest sample ({depth_m[-1] / LEN_UNITS[in_len.value]:,.0f} "
                     f"{in_len.value}) goes to **{_t_end:,.0f} ms** TWT"), kind="info")
    return twt_for, vel_along, vel_axis_si, vint_for, vmsg


@app.cell(hide_code=True)
def _(S, img, mo, np, twt_for):
    # ---- 3. output parameters -----------------------------------------------------------------------------------------
    _tmax = float(np.ceil(twt_for(np.array([img["ntr"] // 2]))[0, -1] * 10) * 100)
    out_dt = mo.ui.number(value=S.get("out_dt", 4.0), start=0, step=0.001, label="Time sample interval (ms)")
    out_tmax = mo.ui.number(value=S.get("out_tmax", _tmax), start=0, step=0.001, label="Record length (ms)")
    out_fmt = mo.ui.dropdown(["IBM", "IEEE"], value=S.get("out_fmt", "IBM"), label="Sample format")
    antialias = mo.ui.checkbox(value=S.get("antialias", True), label="anti-alias filter")
    mo.vstack([mo.md("## 3. Output parameters"),
               mo.hstack([out_dt, out_tmax, out_fmt, antialias], justify="start", gap=1, wrap=True)])
    return antialias, out_dt, out_fmt, out_tmax


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

    def plot_sections(il_result, xl_result, il_value, xl_value, y_max, y_label, *, clip_pct=98.0, fig_height=7.5):
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
        axes[0].set_ylim(y_max, 0)
        axes[0].set_ylabel(y_label)
        fig.tight_layout()
        return fig

    def plot_velocity_sections(il_result, xl_result, il_value, xl_value, y_max, y_label, kind="interval", *,
                               fig_height=7.5):
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
        axes[0].set_ylim(y_max, 0)
        axes[0].set_ylabel(y_label)
        if im is not None:
            _inset_colorbar(fig, axes[0], im)
        fig.tight_layout()
        return fig

    def plot_overlay_sections(il_result, xl_result, il_value, xl_value, y_max, y_label, kind="interval", *,
                              clip_pct=98.0, fig_height=7.5):
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
        axes[0].set_ylim(y_max, 0)
        axes[0].set_ylabel(y_label)
        if im is not None:
            _inset_colorbar(fig, axes[0], im)
        fig.tight_layout()
        return fig

    return plot_overlay_sections, plot_sections, plot_velocity_sections


@app.cell(hide_code=True)
def _(img, img_ilv, img_xlv, mo, np, records):
    # ---- 4. plot: an IL section and an XL section, in depth and in time (as the app's stack) -------------------------
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
    mo.vstack([mo.md("## 4. Plot - depth and time"), sec_il, sec_xl,
               mo.hstack([clip_pct, fig_height], justify="start", gap=1)])
    return clip_pct, fig_height, sec_il, sec_xl


@app.cell(hide_code=True)
def _(antialias, clip_pct, corner_fit, depth_axis, depth_to_time, fig_height, ilxl_to_xy, img, img_ilv, img_xlv,
      in_len, mo, np, out_dt, out_tmax, plot_overlay_sections, plot_sections, plot_velocity_sections, plt, records,
      samples_of, sec_il, sec_xl, time, twt_for, v_unit, vint_for):
    _t0 = time.perf_counter()
    _out_t = np.arange(0.0, float(out_tmax.value) + 1e-9, float(out_dt.value)) / 1000.0

    def _line(fixed, value, along):
        sel = np.flatnonzero(fixed == int(value))
        if not len(sel):
            return None
        sel = sel[np.argsort(along[sel])]
        amp = samples_of(img, records(img)[sel])
        twt = np.array(twt_for(sel))
        tim = depth_to_time(amp, twt, _out_t, antialias.value)
        vz = np.array(vint_for(sel))                        # interval velocity at the depths (NaN below the model)
        vt = np.full((len(sel), len(_out_t)), np.nan)        # ... and at the output times
        for r in range(len(sel)):
            ok = np.isfinite(vz[r])
            if ok.any():
                tend = twt[r][ok][-1]
                vt[r] = np.where(_out_t <= tend, np.interp(_out_t, twt[r][ok], vz[r][ok]), np.nan)
        return along[sel], amp, tim, vt, twt, sel

    _il = _line(img_ilv, sec_il.value, img_xlv)
    _xl = _line(img_xlv, sec_xl.value, img_ilv)
    _secs = time.perf_counter() - _t0
    _cp, _fh = float(clip_pct.value), float(fig_height.value) + 0.5
    _pick = lambda r, i: None if r is None else (r[0], r[i])
    _zmax, _tmax = float(depth_axis[-1]), float(_out_t[-1] * 1000)
    _f_depth = plot_sections(_pick(_il, 1), _pick(_xl, 1), sec_il.value, sec_xl.value, _zmax, f"Depth ({in_len.value})",
                             clip_pct=_cp, fig_height=_fh)
    _f_time = plot_sections(_pick(_il, 2), _pick(_xl, 2), sec_il.value, sec_xl.value, _tmax, "Time (ms)",
                            clip_pct=_cp, fig_height=_fh)
    _f_vel = plot_velocity_sections(_pick(_il, 3), _pick(_xl, 3), sec_il.value, sec_xl.value, _tmax, "Time (ms)",
                                    fig_height=_fh)
    _ov = lambda r: None if r is None else (r[0], r[2], r[3])
    _f_ov = plot_overlay_sections(_ov(_il), _ov(_xl), sec_il.value, sec_xl.value, _tmax, "Time (ms)", clip_pct=_cp,
                                  fig_height=_fh)
    # the depth -> time relation used, and where the two lines are
    _f_tz, (_a1, _a2) = plt.subplots(1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [40, 60]})
    for _r, _lab in ((_il, f"IL={sec_il.value}"), (_xl, f"XL={sec_xl.value}")):
        if _r is not None:
            _k = len(_r[0]) // 2
            _a1.plot(_r[4][_k] * 1000, depth_axis, label=f"{_lab} (middle trace)")
    _a1.invert_yaxis()
    _a1.set_xlabel("Time (ms)")
    _a1.set_ylabel(f"Depth ({in_len.value})")
    _a1.set_title("Depth → time used")
    _a1.grid(alpha=0.3)
    _a1.legend(fontsize=8)
    if corner_fit is not None:
        _ci = np.array([img_ilv.min(), img_ilv.min(), img_ilv.max(), img_ilv.max(), img_ilv.min()], float)
        _cx = np.array([img_xlv.min(), img_xlv.max(), img_xlv.max(), img_xlv.min(), img_xlv.min()], float)
        _ox, _oy = ilxl_to_xy(corner_fit, _ci, _cx)
        _a2.plot(_ox, _oy, color="gray", lw=1)
        for _r, _c, _lab in ((_il, "tab:red", f"IL={sec_il.value}"), (_xl, "tab:blue", f"XL={sec_xl.value}")):
            if _r is not None:
                _lx, _ly = ilxl_to_xy(corner_fit, img_ilv[_r[5]].astype(float), img_xlv[_r[5]].astype(float))
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
    sec_figs = [(f"Depth sections {_tag}", _png(_f_depth)), (f"Time sections {_tag}", _png(_f_time)),
                (f"Velocity sections {_tag}", _png(_f_vel)), (f"Time + velocity {_tag}", _png(_f_ov)),
                ("Depth → time and location", _png(_f_tz))]
    plt.close(_f_tz)
    mo.md(f"IL {sec_il.value} and XL {sec_xl.value} converted in {_secs:.1f} s · velocity in {v_unit.value}; gray "
          "in the velocity sections = below the velocity model (time from the 'below' choice)")
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
def _(Path, S, img, mo):
    # ---- 5. write ---------------------------------------------------------------------------------------------------
    _p = Path(img["path"])
    out_path = mo.ui.text(value=S.get("out_path", str(_p.with_name(f"{_p.stem}_time.sgy"))), label="Output SEG-Y file",
                          full_width=True)
    write_xy = mo.ui.checkbox(value=S.get("write_xy", False),
                              label="write CDP X / Y (bytes 181 / 185, scalar -100) from the corner points")
    overwrite = mo.ui.checkbox(value=False, label="overwrite if it exists")
    write_btn = mo.ui.run_button(label="💾 Convert and write")
    mo.vstack([mo.md("## 5. Convert and write"), out_path, write_xy, mo.hstack([overwrite, write_btn], justify="start", gap=1)])
    return out_path, overwrite, write_btn, write_xy


@app.cell(hide_code=True)
def _(antialias, corner_fit, corners, datetime, depth_to_time, encode_samples, ilxl_to_xy, img,
      img_ilv, img_xlv, in_dz, in_il, in_len, in_xl, in_z0, mo, np, os, out_dt, out_fmt, out_path, out_tmax, overwrite,
      records, samples_of, save_settings, time, twt_for, v_below, v_const, v_corner_table, v_depth_tab, v_domain,
      v_dz, v_grad, v_il, v_len, v_match, v_path, v_time_tab, v_type, v_unit, v_xl, v_z0, vmsg, vsource, write_btn,
      write_xy):
    mo.stop(not write_btn.value)
    _dest = os.path.abspath(os.path.expanduser(out_path.value.strip()))
    mo.stop(_dest == img["path"], mo.callout(mo.md("The output must be a new file."), kind="danger"))
    mo.stop(os.path.exists(_dest) and not overwrite.value,
            mo.callout(mo.md(f"`{_dest}` exists - tick **overwrite** or change the name."), kind="warn"))
    _corners, _fit = corners, corner_fit
    mo.stop(write_xy.value and _fit is None, mo.callout(mo.md("Writing X / Y needs valid corner points."), kind="danger"))

    def _tab(t):
        return t.value.to_dict("records") if hasattr(t.value, "to_dict") else list(t.value or [])

    save_settings(img["path"], {
        "in_len": in_len.value, "in_dz": in_dz.value, "in_z0": in_z0.value, "in_il": in_il.value, "in_xl": in_xl.value,
        "corners": _corners, "vsource": vsource.value, "v_unit": v_unit.value, "v_const": v_const.value,
        "v_depth_tab": _tab(v_depth_tab), "v_time_tab": _tab(v_time_tab), "v_path": v_path.value,
        "vfile": {"type": v_type.value, "domain": v_domain.value, "dz": v_dz.value, "z0": v_z0.value, "len": v_len.value,
                  "il": v_il.value, "xl": v_xl.value, "match": v_match.value, "below": v_below.value,
                  "grad": v_grad.value, "corners": _tab(v_corner_table)},
        "out_dt": out_dt.value, "out_tmax": out_tmax.value, "out_fmt": out_fmt.value, "antialias": antialias.value,
        "out_path": out_path.value, "write_xy": write_xy.value})
    _out_t = np.arange(0.0, float(out_tmax.value) + 1e-9, float(out_dt.value)) / 1000.0
    _ns, _dtf = len(_out_t), int(round(float(out_dt.value) * 1000))
    mo.stop(_ns > 65535 or _dtf > 65535, mo.callout(mo.md("Too many samples for SEG-Y."), kind="danger"))
    import shutil as _shutil
    _need = img["ntr"] * (240 + 4 * _ns) + 3600
    _free = _shutil.disk_usage(os.path.dirname(_dest) or ".").free
    mo.stop(_free < _need * 1.02, mo.callout(mo.md(f"Not enough disk space: needs {_need / 1e9:.1f} GB, "
                                                   f"{_free / 1e9:.1f} GB free."), kind="danger"))
    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _new = [f"DEPTH TO TIME CONVERSION {_now}", f"INPUT: {os.path.basename(img['path'])}"[:76],
            f"VELOCITY: {vmsg.upper()}"[:76],
            f"TWT, DT {float(out_dt.value):g} MS, {_ns} SAMPLES, {out_fmt.value} FLOAT", "ORIGINAL TEXT HEADER:"]
    _old = [l[3:].rstrip() if l[:1] == "C" else l.rstrip() for l in img["lines"]]
    _cards = (_new + _old)[:39] + ["END TEXTUAL HEADER"]
    _text = "".join(f"C{_i + 1:2d} {_c}"[:80].ljust(80) for _i, _c in enumerate(_cards)).replace("–", "-").replace("—", "-").encode(
        "cp037", errors="replace")              # (EBCDIC has no dashes / symbols such as "–": replaced)
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
        with mo.status.progress_bar(total=img["ntr"], title="Depth to time", show_eta=True, show_rate=True) as _bar:
            for _a in range(0, img["ntr"], 2000):
                _sel = np.arange(_a, min(_a + 2000, img["ntr"]))
                _block = np.array(_ri[_sel])
                _amp = samples_of(img, _block)
                _live = np.abs(_amp).max(axis=1) > 0
                _out = np.zeros((len(_sel), _ns))
                if _live.any():
                    _out[_live] = depth_to_time(_amp[_live], np.array(twt_for(_sel[_live])), _out_t, antialias.value)
                _hdr = _block[:, :240].copy()
                if img["order"] == "<":
                    _hdr = _hdr.reshape(len(_sel), 60, 4)[:, :, ::-1].reshape(len(_sel), 240).copy()
                _hdr[:, 114:116] = np.frombuffer(_ns.to_bytes(2, "big"), np.uint8)
                _hdr[:, 116:118] = np.frombuffer(_dtf.to_bytes(2, "big"), np.uint8)
                _hdr[:, 108:110] = 0
                if write_xy.value:
                    _x, _y = ilxl_to_xy(_fit, img_ilv[_sel].astype(float), img_xlv[_sel].astype(float))
                    _hdr[:, 70:72] = np.frombuffer((-100).to_bytes(2, "big", signed=True), np.uint8)
                    _hdr[:, 180:184] = np.round(_x * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                    _hdr[:, 184:188] = np.round(_y * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                np.hstack([_hdr, encode_samples(_out, out_fmt.value)]).tofile(_f)
                _bar.update(increment=len(_sel))
    os.replace(_part, _dest)
    mo.callout(mo.md(f"✅ Wrote **{img['ntr']:,} traces** × {_ns} samples (TWT every {float(out_dt.value):g} ms) to "
                     f"`{_dest}` ({os.path.getsize(_dest) / 1e9:.2f} GB) in {time.perf_counter() - _t0:.0f} s."),
               kind="success")
    return


if __name__ == "__main__":
    app.run()
