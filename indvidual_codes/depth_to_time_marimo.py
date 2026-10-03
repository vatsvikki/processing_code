import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Depth image → time (depth-to-time conversion)

    Converts a depth-migrated image (e.g. a PSDM / RTM stack) to two-way time with the migration velocity: every image
    trace gets the time-depth relation of the velocity trace at its position, and its amplitudes are mapped onto a
    regular time axis (with an anti-alias filter).

    **1.** Depth image + its corner points &nbsp;→&nbsp; **2.** Velocity model + its corner points (map, matching)
    &nbsp;→&nbsp; **3.** Velocity below the model &nbsp;→&nbsp; **4.** Output time axis &nbsp;→&nbsp;
    **5.** Plot any inline / crossline in depth and time &nbsp;→&nbsp; **6.** Write.
    Answers are saved per image file (`depth_to_time_settings.json`).

    This is a **depth-to-time stretch of the depth image**, not a new time migration: the imaging stays that of the
    depth migration (all its velocity / anisotropy work), only the vertical axis becomes time. Use the velocity the
    depth migration used (its **vertical** velocity for a TTI / VTI model) - a different velocity puts the events at
    the wrong times. Time is two-way time (TWT).
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
    # ---- 1. the depth image -----------------------------------------------------------------------------------------
    _saved = load_settings()
    img_box = mo.ui.text(value=_saved.get("last_file", ""), label="Depth image SEG-Y (PSDM / RTM)", full_width=True)
    img_btn = mo.ui.run_button(label="📂 Load image")
    mo.vstack([mo.md("## 1. The depth image"), img_box, img_btn])
    return img_box, img_btn


@app.cell(hide_code=True)
def _(img_box, img_btn, load_settings, mo, re, segy_open):
    mo.stop(not img_btn.value and not img_box.value.strip(), mo.md("_Type the image path and press **Load image**._"))
    try:
        img = segy_open(img_box.value)
        _err = None
    except Exception as _e:
        img, _err = None, f"{type(_e).__name__}: {_e}"
    mo.stop(img is None, mo.callout(mo.md(f"**Could not read the image** - {_err}"), kind="danger"))
    S = load_settings().get(img["path"], {})
    _txt = " ".join(img["lines"]).upper()
    _ft = bool(re.search(r"\b(FT|FEET|FOOT)\b|\d'", _txt))
    img_len = mo.ui.dropdown(["ft", "m"], value=S.get("img_len", "ft" if _ft else "m"), label="Depth unit")
    img_dz = mo.ui.number(value=S.get("img_dz", img["dt"] / 1000.0), start=0, step=0.001,
                          label="Depth sample interval")
    img_z0 = mo.ui.number(value=S.get("img_z0", 0.0), step=0.001, label="First sample at depth")
    img_il = mo.ui.number(value=S.get("img_il", 189), start=1, stop=237, label="Inline byte")
    img_xl = mo.ui.number(value=S.get("img_xl", 193), start=1, stop=237, label="Crossline byte")
    _what = next((l[3:].strip() for l in img["lines"] if "DESCRIPTION" in l.upper()), "")
    mo.vstack([
        mo.md(f"**{img['ntr']:,} traces** · {img['ns']} samples · sample-interval field {img['dt']} · "
              f"{img['size'] / 1e9:.2f} GB" + (f" · *{_what}*" if _what else "")),
        mo.accordion({"Text header": mo.md("```\n" + "\n".join(l.rstrip() for l in img["lines"]) + "\n```")}),
        mo.hstack([img_len, img_dz, img_z0], justify="start", gap=1, wrap=True),
        mo.hstack([img_il, img_xl], justify="start", gap=1, wrap=True),
    ])
    return S, img, img_dz, img_il, img_len, img_xl, img_z0


@app.cell(hide_code=True)
def _(S, corners_from_text, header_word, img, img_il, img_xl, mo, np, records):
    # the image's grid from its trace headers, and its corner points (text header -> saved -> trace-header corners)
    _r = records(img)
    img_ilv = header_word(_r, int(img_il.value), "i4", img["order"])
    img_xlv = header_word(_r, int(img_xl.value), "i4", img["order"])
    _sc = header_word(_r, 71, "i2", img["order"]).astype(float)
    _f = np.where(_sc < 0, -1.0 / np.where(_sc == 0, 1, _sc), np.where(_sc > 0, _sc, 1.0))
    img_hx = header_word(_r, 181, "i4", img["order"]) * _f
    img_hy = header_word(_r, 185, "i4", img["order"]) * _f

    def _hdr_corners():
        rows = []
        for a_ in (img_ilv.min(), img_ilv.max()):
            for b_ in (img_xlv.min(), img_xlv.max()):
                k = int(np.argmin((img_ilv - a_) ** 2 + (img_xlv - b_) ** 2))
                rows.append({"IL": int(img_ilv[k]), "XL": int(img_xlv[k]), "X": round(float(img_hx[k]), 2),
                             "Y": round(float(img_hy[k]), 2)})
        return rows

    _text = corners_from_text(img["lines"])
    _src = "saved for this image" if S.get("img_corners") else ("text header" if _text else "trace headers")
    img_corner_table = mo.ui.data_editor(
        S.get("img_corners") or [{"IL": r[0], "XL": r[1], "X": r[2], "Y": r[3]} for r in _text] or _hdr_corners(),
        label="Image corner points (IL, XL ↔ X, Y)")
    mo.vstack([mo.md(f"**Image grid** (trace headers): IL {img_ilv.min()} – {img_ilv.max()}, XL {img_xlv.min()} – "
                     f"{img_xlv.max()} · corner points from the **{_src}** - check / edit them (at least 3):"),
               img_corner_table])
    return img_corner_table, img_hx, img_hy, img_ilv, img_xlv


@app.cell(hide_code=True)
def _(DOMAINS, S, TYPES, VEL_UNITS, img_len, mo):
    MATCHES = ["Corner points (image IL/XL -> X/Y -> velocity IL/XL)", "Same inline / crossline",
               "Trace-header X / Y (bytes 181 / 185, scalar 71)"]
    # ---- 2. the velocity model ---------------------------------------------------------------------------------------
    vel_box = mo.ui.text(value=S.get("vel_path", ""), label="Velocity model SEG-Y (the one the depth migration used)",
                         full_width=True)
    vel_type = mo.ui.dropdown(TYPES, value=S.get("vel_type", "Interval"), label="Velocity type")
    vel_domain = mo.ui.dropdown(DOMAINS, value=S.get("vel_domain", DOMAINS[1]), label="Vertical domain")
    vel_dz = mo.ui.number(value=S.get("vel_dz", 0.0), start=0, step=0.001,
                          label="Sample interval (ms / depth unit; 0 = from its header)")
    vel_z0 = mo.ui.number(value=S.get("vel_z0", 0.0), step=0.001, label="First sample at")
    vel_len = mo.ui.dropdown(["ft", "m"], value=S.get("vel_len", img_len.value), label="Depth unit")
    vel_unit = mo.ui.dropdown(list(VEL_UNITS), value=S.get("vel_unit", "ft/s" if img_len.value == "ft" else "m/s"),
                              label="Velocity unit")
    vel_il = mo.ui.number(value=S.get("vel_il", 189), start=1, stop=237, label="Inline byte")
    vel_xl = mo.ui.number(value=S.get("vel_xl", 193), start=1, stop=237, label="Crossline byte")
    match = mo.ui.dropdown(MATCHES, value=S.get("match", MATCHES[0]) if S.get("match") in MATCHES else MATCHES[0],
                           label="Match image and velocity traces by")
    mo.vstack([mo.md("## 2. The velocity model"), vel_box,
               mo.hstack([vel_type, vel_domain, vel_dz, vel_z0], justify="start", gap=1, wrap=True),
               mo.hstack([vel_len, vel_unit, vel_il, vel_xl], justify="start", gap=1, wrap=True), match])
    return MATCHES, match, vel_box, vel_domain, vel_dz, vel_il, vel_len, vel_type, vel_unit, vel_xl, vel_z0


@app.cell(hide_code=True)
def _(DOMAINS, S, corners_from_text, header_word, mo, np, records, samples_of, segy_open, vel_box, vel_domain, vel_dz,
      vel_il, vel_len, vel_xl, vel_z0):
    # the velocity model's grid, values and corner points
    mo.stop(not vel_box.value.strip(), mo.md("_Type the velocity model path._"))
    _err = ""
    try:
        vel = segy_open(vel_box.value)
    except Exception as _e:
        vel = None
        _err = f"{type(_e).__name__}: {_e}"
    mo.stop(vel is None, mo.callout(mo.md(f"**Could not read the velocity model** - {_err}"), kind="danger"))
    _r = records(vel)
    vel_ilv = header_word(_r, int(vel_il.value), "i4", vel["order"])
    vel_xlv = header_word(_r, int(vel_xl.value), "i4", vel["order"])
    _sc = header_word(_r, 71, "i2", vel["order"]).astype(float)
    _f = np.where(_sc < 0, -1.0 / np.where(_sc == 0, 1, _sc), np.where(_sc > 0, _sc, 1.0))
    vel_hx = header_word(_r, 181, "i4", vel["order"]) * _f
    vel_hy = header_word(_r, 185, "i4", vel["order"]) * _f
    vel_dz_val = float(vel_dz.value) if vel_dz.value else vel["dt"] / 1000.0
    vel_axis = float(vel_z0.value) + np.arange(vel["ns"]) * vel_dz_val
    _unit = "ms" if vel_domain.value == DOMAINS[0] else vel_len.value
    _pick = np.linspace(0, vel["ntr"] - 1, min(vel["ntr"], 200)).astype(np.int64)
    _v = samples_of(vel, _r[_pick])
    _live = _v[_v != 0]
    _neg = float((_live < 0).mean()) if _live.size else 0.0
    _lo, _hi = (np.percentile(_live, [1, 99]) if _live.size else (0, 0))
    vel_ok = _neg <= 0.01
    _text = corners_from_text(vel["lines"])
    _src = "saved" if S.get("vel_corners") else ("text header" if _text else "none found - type them")
    vel_corner_table = mo.ui.data_editor(
        S.get("vel_corners") or [{"IL": r[0], "XL": r[1], "X": r[2], "Y": r[3]} for r in _text]
        or [{"IL": "", "XL": "", "X": "", "Y": ""}] * 3, label="Velocity model corner points (IL, XL ↔ X, Y)")
    mo.vstack([
        mo.callout(mo.md(f"Velocity model: **{vel['ntr']:,} traces**, {vel['ns']} samples, {vel_axis[0]:g} – "
                         f"{vel_axis[-1]:g} {_unit} · values {_lo:,.0f} – {_hi:,.0f} · grid IL {vel_ilv.min()} – "
                         f"{vel_ilv.max()}, XL {vel_xlv.min()} – {vel_xlv.max()}"), kind="info"),
        mo.callout(mo.md(f"**This does not look like a velocity** - {_neg:.0%} of its samples are negative. Load the "
                         "velocity model, not an image."), kind="danger") if not vel_ok else mo.md(""),
        mo.md(f"Velocity corner points from the **{_src}** - check / edit them:"), vel_corner_table,
    ])
    return vel, vel_axis, vel_corner_table, vel_hx, vel_hy, vel_ilv, vel_ok, vel_xlv


@app.cell(hide_code=True)
def _(MATCHES, corner_rows_of, fit_affine, ilxl_to_xy, img, img_corner_table, img_hx, img_hy, img_ilv, img_xlv, match,
      mo, np, plt, vel_corner_table, vel_hx, vel_hy, vel_ilv, vel_xlv, xy_to_ilxl):
    # ---- corner-point fits, the map, and which velocity trace each image trace uses ---------------------------------
    from scipy.spatial import cKDTree
    img_corners = corner_rows_of(img_corner_table.value)
    vel_corners = corner_rows_of(vel_corner_table.value)
    img_fit, vel_fit = fit_affine(img_corners), fit_affine(vel_corners)
    _msgs = []
    for _name, _fit, _hx, _hy, _il, _xl in (("Image", img_fit, img_hx, img_hy, img_ilv, img_xlv),
                                            ("Velocity", vel_fit, vel_hx, vel_hy, vel_ilv, vel_xlv)):
        if _fit is None:
            _msgs.append(f"**{_name}: the corner points do not define a grid** (need 3 corners not on one line).")
            continue
        _px, _py = ilxl_to_xy(_fit, _il.astype(float), _xl.astype(float))
        _mis = np.hypot(_px - _hx, _py - _hy)
        _msgs.append(f"**{_name}**: bin {_fit['il_bin']:.1f} × {_fit['xl_bin']:.1f}, inline direction "
                     f"{_fit['il_az']:.1f}° · corner misfit {_fit['rms']:.2f} · trace-header X/Y vs corners: median "
                     f"{np.median(_mis):.1f}, max {_mis.max():.1f}")
    # matching
    if match.value == MATCHES[0]:
        if img_fit is None or vel_fit is None:
            mo.stop(True, mo.callout(mo.md("  \n".join(_msgs) + "  \nMatching by corner points needs both corner "
                                                                    "tables."), kind="danger"))
        _x, _y = ilxl_to_xy(img_fit, img_ilv.astype(float), img_xlv.astype(float))
        _qi, _qx = xy_to_ilxl(vel_fit, _x, _y)
        _dist, vel_of = cKDTree(np.column_stack([vel_ilv, vel_xlv]).astype(float)).query(np.column_stack([_qi, _qx]))
        _unit = "velocity bins"
    elif match.value == MATCHES[1]:
        _dist, vel_of = cKDTree(np.column_stack([vel_ilv, vel_xlv]).astype(float)).query(
            np.column_stack([img_ilv, img_xlv]).astype(float))
        _unit = "IL/XL"
    else:
        _dist, vel_of = cKDTree(np.column_stack([vel_hx, vel_hy])).query(np.column_stack([img_hx, img_hy]))
        _unit = "coordinate units"
    _msgs.append(f"Matching **{match.value.split(' (')[0]}**: {float((_dist < 0.5).mean()):.0%} of the image traces "
                 f"on a velocity trace, largest distance {_dist.max():.2f} {_unit}"
                 + (" - ⚠ part of the image is outside the velocity model" if _dist.max() > 1.5 else ""))
    _fig, _ax = plt.subplots(figsize=(7.5, 5.5))
    for _fit, _il, _xl, _col, _lab in ((vel_fit, vel_ilv, vel_xlv, "#2a78d6", "velocity model grid"),
                                       (img_fit, img_ilv, img_xlv, "#e34948", "image grid")):
        if _fit is not None:
            _ci = np.array([_il.min(), _il.min(), _il.max(), _il.max(), _il.min()], float)
            _cx = np.array([_xl.min(), _xl.max(), _xl.max(), _xl.min(), _xl.min()], float)
            _ox, _oy = ilxl_to_xy(_fit, _ci, _cx)
            _ax.plot(_ox, _oy, color=_col, lw=1.8, label=_lab)
    for _rows, _col in ((img_corners, "#e34948"), (vel_corners, "#2a78d6")):
        for _r in _rows:
            try:
                _ax.plot(float(_r["X"]), float(_r["Y"]), "o", color=_col, ms=5)
                _ax.annotate(f"{int(float(_r['IL']))}/{int(float(_r['XL']))}", (float(_r["X"]), float(_r["Y"])),
                             fontsize=7, xytext=(3, 3), textcoords="offset points", color=_col)
            except (KeyError, TypeError, ValueError):
                pass
    _ax.set_aspect("equal")
    _ax.ticklabel_format(useOffset=False, style="plain")
    _ax.tick_params(labelsize=7)
    _ax.legend(fontsize=8)
    _ax.set_title("Grids from the corner points (labels: IL/XL)", fontsize=10)
    _fig.tight_layout()
    mo.vstack([mo.md("### Corner points and matching"), mo.callout(mo.md("  \n".join(_msgs)), kind="info"), _fig])
    return img_corners, img_fit, vel_corners, vel_of


@app.cell(hide_code=True)
def _(BELOW, S, mo, vel_len, vel_unit):
    # ---- 3. below the velocity model ----------------------------------------------------------------------------------
    below = mo.ui.dropdown(BELOW, value=S.get("below", BELOW[0]), label="Below the velocity model's last sample")
    grad_len = mo.ui.number(value=S.get("grad_len", 1000.0), start=0, step=0.001,
                            label=f"gradient of the last ({vel_len.value})")
    vcap = mo.ui.number(value=S.get("vcap", 0.0), start=0, step=0.001, label=f"never faster than ({vel_unit.value}; 0 = no cap)")
    _deflt = [{"depth": 6000, "velocity": 9000}, {"depth": 15000, "velocity": 13000}, {"depth": 40000, "velocity": 18000}]
    vtable = mo.ui.data_editor(S.get("vtable", _deflt),
                               label=f"Interval velocity below the model: depth ({vel_len.value}), velocity ({vel_unit.value})")
    mo.vstack([mo.md("## 3. Velocity below the model  \n<span style='opacity:.7'>Only used where the image is deeper "
                     "than the velocity model. Every time below the model depends on this choice - with the full "
                     "migration velocity it is not needed.</span>"),
               below, mo.hstack([grad_len, vcap], justify="start", gap=1, wrap=True), vtable])
    return below, grad_len, vcap, vtable


@app.cell(hide_code=True)
def _(LEN_UNITS, VEL_UNITS, below, grad_len, img_dz, img_len, img_z0, img, np, to_interval, time_at_depths, vcap,
      vel, vel_axis, vel_domain, vel_len, vel_type, vel_unit, vtable):
    # the depth -> time relation of a block of image traces (used by the preview and the writer)
    img_axis_m = (float(img_z0.value) + np.arange(img["ns"]) * float(img_dz.value)) * LEN_UNITS[img_len.value]
    _rows = vtable.value
    _rows = _rows.to_dict("records") if hasattr(_rows, "to_dict") else list(_rows or [])
    _tab = []
    for _r in _rows:
        try:
            _tab.append((float(_r["depth"]) * LEN_UNITS[vel_len.value], float(_r["velocity"]) * VEL_UNITS[vel_unit.value]))
        except (KeyError, TypeError, ValueError):
            pass
    vel_table = np.array(sorted(_tab)) if _tab else None

    def twt_of(vel_values):
        """TWT (s) at every image depth for these velocity traces [n, vel ns] (in the model's units)."""
        v = np.asarray(vel_values, np.float64) * VEL_UNITS[vel_unit.value]
        if vel_domain.value.startswith("Time"):
            ax = np.asarray(vel_axis) / 1000.0
        else:
            ax = np.asarray(vel_axis) * LEN_UNITS[vel_len.value]
        vint, t, z, bad = to_interval(v, ax, vel_type.value, vel_domain.value)
        return time_at_depths(t, z, vint, img_axis_m, below.value, float(grad_len.value) * LEN_UNITS[vel_len.value],
                              float(vcap.value) * VEL_UNITS[vel_unit.value], vel_table)

    return img_axis_m, twt_of


@app.cell(hide_code=True)
def _(LEN_UNITS, S, img, img_axis_m, img_len, mo, np, records, samples_of, twt_of, vel, vel_of):
    # ---- 4. the output time axis -------------------------------------------------------------------------------------
    # the deepest image sample's time over 60 velocity traces spread across the survey -> the default record length
    _pick = np.unique(vel_of[np.linspace(0, img["ntr"] - 1, 60).astype(np.int64)])
    _tt, _ = twt_of(samples_of(vel, records(vel)[_pick]))
    _t = _tt[[int(np.argmax(_tt[:, -1]))]]
    _tmax = float(np.ceil(_t[0, -1] * 1000 / 100.0) * 100)
    out_dt = mo.ui.number(value=S.get("out_dt", 2.0), start=0, step=0.001, label="Output sample interval (ms)")
    out_tmax = mo.ui.number(value=S.get("out_tmax", _tmax), start=0, step=0.001, label="Output record length (ms)")
    antialias = mo.ui.checkbox(value=S.get("antialias", True), label="anti-alias filter (recommended)")
    out_fmt = mo.ui.dropdown(["IBM", "IEEE"], value=S.get("out_fmt", "IBM"), label="Sample format")
    mo.vstack([mo.md(f"## 4. Output time axis  \n<span style='opacity:.7'>The image's deepest sample "
                     f"({img_axis_m[-1] / LEN_UNITS[img_len.value]:,.0f} {img_len.value}) is at up to "
                     f"**{_t[0, -1] * 1000:,.0f} ms** TWT (60 traces across the survey).</span>"),
               mo.hstack([out_dt, out_tmax, antialias, out_fmt], justify="start", gap=1, wrap=True)])
    return antialias, out_dt, out_fmt, out_tmax


@app.cell(hide_code=True)
def _(img_ilv, mo):
    # ---- 5. preview: any inline or crossline ------------------------------------------------------------------------
    pv_kind = mo.ui.radio(["Inline", "Crossline"], value="Inline", label="Plot", inline=True)
    mo.vstack([mo.md("## 5. Plot a line - depth and time"), pv_kind])
    return (pv_kind,)


@app.cell(hide_code=True)
def _(img_ilv, img_xlv, mo, np, pv_kind):
    _v = np.unique(img_ilv if pv_kind.value == "Inline" else img_xlv)
    pv_line = mo.ui.slider(steps=[int(x) for x in _v], value=int(_v[len(_v) // 2]), show_value=True,
                           include_input=True, full_width=True, debounce=True, label=f"{pv_kind.value} number")
    pv_line
    return (pv_line,)


@app.cell(hide_code=True)
def _(antialias, depth_to_time, ilxl_to_xy, img, img_axis_m, img_fit, img_ilv, img_len, img_xlv, mo, np, out_dt,
      out_tmax, plt, pv_kind, pv_line, records, samples_of, time, twt_of, vel, vel_of, vel_ok):
    mo.stop(not vel_ok, mo.callout(mo.md("The velocity file is not a velocity (step 2)."), kind="danger"))
    _inl = pv_kind.value == "Inline"
    _sel = np.flatnonzero((img_ilv if _inl else img_xlv) == int(pv_line.value))
    mo.stop(len(_sel) == 0, mo.callout(mo.md(f"{pv_kind.value} {pv_line.value} is not in the image."), kind="warn"))
    _along = img_xlv if _inl else img_ilv
    _sel = _sel[np.argsort(_along[_sel])]
    _t0 = time.perf_counter()
    _amp = samples_of(img, records(img)[_sel])
    _twt, _zend = twt_of(samples_of(vel, records(vel)[vel_of[_sel]]))
    _out_t = np.arange(0.0, float(out_tmax.value) + 1e-9, float(out_dt.value)) / 1000.0
    _tim = depth_to_time(_amp, _twt, _out_t, antialias.value)
    _secs = time.perf_counter() - _t0
    _live = np.flatnonzero(np.abs(_amp).max(axis=1) > 0)
    _k = int(_live[len(_live) // 2]) if len(_live) else len(_sel) // 2
    _du = 0.3048 if img_len.value == "ft" else 1.0
    _zax = img_axis_m / _du
    _clip = float(np.percentile(np.abs(_amp[_live]), 98)) if len(_live) else 1.0
    _x0, _x1 = _along[_sel][0], _along[_sel][-1]
    _xlab = "Crossline" if _inl else "Inline"
    _fig = plt.figure(figsize=(16, 7))
    _gs = _fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], height_ratios=[1, 0.8])
    _a0 = _fig.add_subplot(_gs[:, 0])
    _a1 = _fig.add_subplot(_gs[:, 1])
    _a2 = _fig.add_subplot(_gs[0, 2])
    _a3 = _fig.add_subplot(_gs[1, 2])
    _a0.imshow(_amp.T, aspect="auto", cmap="gray", vmin=-_clip, vmax=_clip, extent=[_x0, _x1, _zax[-1], _zax[0]])
    _a0.set_title(f"Depth - {pv_kind.value.lower()} {pv_line.value}", fontsize=10)
    _a0.set_ylabel(f"Depth ({img_len.value})", fontsize=9)
    _a1.imshow(_tim.T, aspect="auto", cmap="gray", vmin=-_clip, vmax=_clip, extent=[_x0, _x1, _out_t[-1] * 1000, 0])
    _a1.set_title(f"Time - {pv_kind.value.lower()} {pv_line.value}", fontsize=10)
    _a1.set_ylabel("TWT (ms)", fontsize=9)
    _zl = float(np.median(_zend)) / _du
    _tl = float(np.interp(np.median(_zend), img_axis_m, _twt[_k])) * 1000
    _deep = _zl < _zax[-1]
    if _deep:
        _a0.axhline(_zl, color="#e34948", lw=1, ls="--")
        _a1.axhline(_tl, color="#e34948", lw=1, ls="--")
    for _a in (_a0, _a1):
        _a.set_xlabel(_xlab, fontsize=9)
        _a.tick_params(labelsize=8)
    _a2.plot(_twt[_k] * 1000, _zax, color="#2a78d6")
    if _deep:
        _a2.axhspan(_zl, _zax[-1], color="#e34948", alpha=0.08)
    _a2.invert_yaxis()
    _a2.set_xlabel("TWT (ms)", fontsize=8)
    _a2.set_ylabel(f"Depth ({img_len.value})", fontsize=8)
    _a2.set_title(f"Time-depth at {_xlab.lower()} {_along[_sel][_k]}", fontsize=9)
    _a2.tick_params(labelsize=7)
    _a2.grid(alpha=0.3)
    # where the line is
    if img_fit is not None:
        _ci = np.array([img_ilv.min(), img_ilv.min(), img_ilv.max(), img_ilv.max(), img_ilv.min()], float)
        _cx = np.array([img_xlv.min(), img_xlv.max(), img_xlv.max(), img_xlv.min(), img_xlv.min()], float)
        _ox, _oy = ilxl_to_xy(img_fit, _ci, _cx)
        _a3.plot(_ox, _oy, color="#9aa3b2", lw=1.2)
        _lx, _ly = ilxl_to_xy(img_fit, img_ilv[_sel].astype(float), img_xlv[_sel].astype(float))
        _a3.plot(_lx, _ly, color="#e34948", lw=2.2)
        _a3.set_aspect("equal")
        _a3.ticklabel_format(useOffset=False, style="plain")
        _a3.tick_params(labelsize=6)
        _a3.set_title(f"{pv_kind.value} {pv_line.value} on the survey", fontsize=9)
    else:
        _a3.axis("off")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"{len(_sel)} traces ({len(_live)} live) converted in {_secs:.1f} s"
              + (f" · red dashed line = end of the velocity model ({_zl:,.0f} {img_len.value} ≈ {_tl:,.0f} ms)"
                 if _deep else "")),
        mo.callout(mo.md(f"⚠ Below **{_zl:,.0f} {img_len.value}** (≈ {_tl:,.0f} ms) the times come from the choice in "
                         "step 3, not from the velocity model."), kind="warn") if _deep else mo.md(""),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(Path, S, img, mo):
    # ---- 6. write ----------------------------------------------------------------------------------------------------
    _p = Path(img["path"])
    out_path = mo.ui.text(value=S.get("out_path", str(_p.with_name(f"{_p.stem}_time.sgy"))), label="Output SEG-Y file",
                          full_width=True)
    overwrite = mo.ui.checkbox(value=False, label="overwrite if it exists")
    write_xy = mo.ui.checkbox(value=S.get("write_xy", False),
                              label="write CDP X / Y (bytes 181 / 185, scalar -100) from the image corner points")
    write_btn = mo.ui.run_button(label="💾 Convert and write")
    mo.vstack([mo.md("## 6. Write the time image"), out_path, write_xy,
               mo.hstack([overwrite, write_btn], justify="start", gap=1)])
    return out_path, overwrite, write_btn, write_xy


@app.cell(hide_code=True)
def _(antialias, below, datetime, depth_to_time, encode_samples, grad_len, img, img_dz, img_il, img_len, img_xl,
      img_z0, match, mo, np, os, out_dt, out_fmt, out_path, out_tmax, overwrite, records, samples_of, save_settings,
      time, twt_of, vcap, vel, vel_box, vel_domain, vel_dz, vel_il, vel_len, vel_of, vel_ok, vel_type, vel_unit,
      vel_xl, vel_z0, vtable, write_btn, write_xy, img_corners, vel_corners, img_fit, ilxl_to_xy, img_ilv, img_xlv):
    mo.stop(not write_btn.value)
    mo.stop(not vel_ok, mo.callout(mo.md("Not written: the velocity file is not a velocity (step 2)."), kind="danger"))
    _dest = os.path.abspath(os.path.expanduser(out_path.value.strip()))
    mo.stop(_dest in (img["path"], vel["path"]), mo.callout(mo.md("The output must be a new file."), kind="danger"))
    mo.stop(os.path.exists(_dest) and not overwrite.value,
            mo.callout(mo.md(f"`{_dest}` exists - tick **overwrite** or change the name."), kind="warn"))
    _rows = vtable.value
    _rows = _rows.to_dict("records") if hasattr(_rows, "to_dict") else list(_rows or [])
    save_settings(img["path"], {
        "img_len": img_len.value, "img_dz": img_dz.value, "img_z0": img_z0.value, "img_il": img_il.value,
        "img_xl": img_xl.value, "vel_path": vel_box.value, "vel_type": vel_type.value, "vel_domain": vel_domain.value,
        "vel_dz": vel_dz.value, "vel_z0": vel_z0.value, "vel_len": vel_len.value, "vel_unit": vel_unit.value,
        "vel_il": vel_il.value, "vel_xl": vel_xl.value, "match": match.value, "below": below.value,
        "grad_len": grad_len.value, "vcap": vcap.value, "vtable": _rows, "out_dt": out_dt.value,
        "out_tmax": out_tmax.value, "antialias": antialias.value, "out_fmt": out_fmt.value, "out_path": out_path.value,
        "img_corners": img_corners, "vel_corners": vel_corners, "write_xy": write_xy.value})
    mo.stop(write_xy.value and img_fit is None,
            mo.callout(mo.md("Writing X / Y needs valid image corner points."), kind="danger"))
    _out_t = np.arange(0.0, float(out_tmax.value) + 1e-9, float(out_dt.value)) / 1000.0
    _ns = len(_out_t)
    _dtf = int(round(float(out_dt.value) * 1000))
    mo.stop(_ns > 65535 or _dtf > 65535, mo.callout(mo.md("Too many samples / too long a sample interval for SEG-Y."),
                                                    kind="danger"))
    _need = img["ntr"] * (240 + 4 * _ns) + 3600
    import shutil as _shutil
    _free = _shutil.disk_usage(os.path.dirname(_dest) or ".").free
    mo.stop(_free < _need * 1.02, mo.callout(mo.md(f"Not enough disk space: needs {_need / 1e9:.1f} GB, "
                                                   f"{_free / 1e9:.1f} GB free."), kind="danger"))
    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _new = [f"DEPTH TO TIME CONVERSION {_now}", f"IMAGE: {os.path.basename(img['path'])}"[:76],
            f"VELOCITY: {os.path.basename(vel['path'])}"[:76],
            f"VELOCITY {vel_type.value.upper()} {vel_domain.value.upper()} {vel_unit.value.upper()}",
            f"BELOW THE MODEL: {below.value.upper()}"[:76],
            f"TWT, DT {float(out_dt.value):g} MS, {_ns} SAMPLES, {out_fmt.value} FLOAT"
            + (", ANTI-ALIAS" if antialias.value else ""), "ORIGINAL TEXT HEADER:"]
    _old = [l[3:].rstrip() if l[:1] == "C" else l.rstrip() for l in img["lines"]]
    _cards = (_new + _old)[:39] + ["END TEXTUAL HEADER"]
    _text = "".join(f"C{_i + 1:2d} {_c}"[:80].ljust(80) for _i, _c in enumerate(_cards)).encode("cp037")
    _bin = bytearray(img["binary"])
    if img["order"] == "<":
        for _o in (12, 16, 18, 20, 22, 24, 304):
            _bin[_o:_o + 2] = _bin[_o:_o + 2][::-1]
    for _o, _val in ((16, _dtf), (18, _dtf), (20, _ns), (22, _ns), (24, 1 if out_fmt.value == "IBM" else 5), (304, 0)):
        _bin[_o:_o + 2] = int(_val).to_bytes(2, "big")
    _ri, _rv = records(img), records(vel)
    _chunk = 2000
    _t0 = time.perf_counter()
    _part = _dest + ".part"
    with open(_part, "wb") as _f:
        _f.write(_text)
        _f.write(bytes(_bin))
        with mo.status.progress_bar(total=img["ntr"], title="Depth to time", show_eta=True, show_rate=True) as _bar:
            for _a in range(0, img["ntr"], _chunk):
                _sel = np.arange(_a, min(_a + _chunk, img["ntr"]))
                _block = np.array(_ri[_sel])
                _amp = samples_of(img, _block)
                _live = np.abs(_amp).max(axis=1) > 0
                _out = np.zeros((len(_sel), _ns))
                if _live.any():
                    _vu, _inv = np.unique(vel_of[_sel[_live]], return_inverse=True)
                    _twt_u, _ = twt_of(samples_of(vel, _rv[_vu]))
                    _out[_live] = depth_to_time(_amp[_live], _twt_u[_inv], _out_t, antialias.value)
                _hdr = _block[:, :240].copy()
                if img["order"] == "<":
                    _hdr = _hdr.reshape(len(_sel), 60, 4)[:, :, ::-1].reshape(len(_sel), 240).copy()
                _hdr[:, 114:116] = np.frombuffer(_ns.to_bytes(2, "big"), np.uint8)
                _hdr[:, 116:118] = np.frombuffer(_dtf.to_bytes(2, "big"), np.uint8)
                _hdr[:, 108:110] = 0
                if write_xy.value:
                    _x, _y = ilxl_to_xy(img_fit, img_ilv[_sel].astype(float), img_xlv[_sel].astype(float))
                    _hdr[:, 70:72] = np.frombuffer((-100).to_bytes(2, "big", signed=True), np.uint8)
                    _hdr[:, 180:184] = np.round(_x * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                    _hdr[:, 184:188] = np.round(_y * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                np.hstack([_hdr, encode_samples(_out, out_fmt.value)]).tofile(_f)
                _bar.update(increment=len(_sel))
    os.replace(_part, _dest)
    mo.callout(mo.md(f"✅ Wrote **{img['ntr']:,} traces** × {_ns} samples (TWT every {float(out_dt.value):g} ms) to "
                     f"`{_dest}` ({os.path.getsize(_dest) / 1e9:.2f} GB) in {time.perf_counter() - _t0:.0f} s. "
                     "Answers saved for this image."), kind="success")
    return


if __name__ == "__main__":
    app.run()
