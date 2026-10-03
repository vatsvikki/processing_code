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

    **1.** Depth image &nbsp;→&nbsp; **2.** Velocity model &nbsp;→&nbsp; **3.** Velocity below the model &nbsp;→&nbsp;
    **4.** Output time axis &nbsp;→&nbsp; **5.** Preview &nbsp;→&nbsp; **6.** Write.
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
def _(DOMAINS, S, TYPES, VEL_UNITS, img_len, mo):
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
    match = mo.ui.dropdown(["Same inline / crossline", "Nearest X / Y (bytes 181 / 185, scalar 71)"],
                           value=S.get("match", "Same inline / crossline"), label="Match image and velocity traces by")
    mo.vstack([mo.md("## 2. The velocity model"), vel_box,
               mo.hstack([vel_type, vel_domain, vel_dz, vel_z0], justify="start", gap=1, wrap=True),
               mo.hstack([vel_len, vel_unit, vel_il, vel_xl], justify="start", gap=1, wrap=True), match])
    return match, vel_box, vel_domain, vel_dz, vel_il, vel_len, vel_type, vel_unit, vel_xl, vel_z0


@app.cell(hide_code=True)
def _(DOMAINS, header_word, img, img_il, img_xl, match, mo, np, records, samples_of, segy_open, vel_box, vel_dz,
      vel_domain, vel_il, vel_len, vel_xl, vel_z0):
    # both files' geometry, and which velocity trace each image trace uses
    mo.stop(not vel_box.value.strip(), mo.md("_Type the velocity model path._"))
    try:
        vel = segy_open(vel_box.value)
    except Exception as _e:
        mo.stop(True, mo.callout(mo.md(f"**Could not read the velocity model** - {type(_e).__name__}: {_e}"),
                                 kind="danger"))
    from scipy.spatial import cKDTree
    _ri, _rv = records(img), records(vel)
    img_ilv = header_word(_ri, int(img_il.value), "i4", img["order"])
    img_xlv = header_word(_ri, int(img_xl.value), "i4", img["order"])
    _vil = header_word(_rv, int(vel_il.value), "i4", vel["order"])
    _vxl = header_word(_rv, int(vel_xl.value), "i4", vel["order"])
    if match.value.startswith("Same"):
        _a, _b = np.column_stack([_vil, _vxl]).astype(float), np.column_stack([img_ilv, img_xlv]).astype(float)
    else:
        def _xy(rec, order):
            sc = header_word(rec, 71, "i2", order).astype(float)
            f = np.where(sc < 0, -1.0 / np.where(sc == 0, 1, sc), np.where(sc > 0, sc, 1.0))
            return np.column_stack([header_word(rec, 181, "i4", order) * f, header_word(rec, 185, "i4", order) * f])
        _a, _b = _xy(_rv, vel["order"]), _xy(_ri, img["order"])
    _dist, vel_of = cKDTree(_a).query(_b)
    _exact = float((_dist == 0).mean())
    vel_dz_val = float(vel_dz.value) if vel_dz.value else vel["dt"] / 1000.0
    vel_axis = float(vel_z0.value) + np.arange(vel["ns"]) * vel_dz_val
    _unit = "ms" if vel_domain.value == DOMAINS[0] else vel_len.value
    _pick = np.linspace(0, vel["ntr"] - 1, min(vel["ntr"], 200)).astype(np.int64)
    _v = samples_of(vel, _rv[_pick])
    _live = _v[_v != 0]
    _neg = float((_live < 0).mean()) if _live.size else 0.0
    _lo, _hi = (np.percentile(_live, [1, 99]) if _live.size else (0, 0))
    _msgs = [mo.callout(mo.md(
        f"Velocity model: **{vel['ntr']:,} traces**, {vel['ns']} samples, {vel_axis[0]:g} – {vel_axis[-1]:g} {_unit} · "
        f"values {_lo:,.0f} – {_hi:,.0f}  \nImage traces matched: **{_exact:.0%} exactly**"
        + ("" if _exact == 1 else f", the others to the nearest velocity trace (largest distance {_dist.max():.1f} "
           f"{'IL/XL' if match.value.startswith('Same') else 'coordinate units'})")),
        kind="info" if _exact > 0.5 else "warn")]
    if _neg > 0.01:
        _msgs.append(mo.callout(mo.md(f"**This does not look like a velocity** - {_neg:.0%} of its samples are negative. "
                                      "Load the velocity model, not an image."), kind="danger"))
    vel_ok = _neg <= 0.01
    mo.vstack(_msgs)
    return img_ilv, img_xlv, vel, vel_axis, vel_of, vel_ok


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
def _(img_ilv, mo, np):
    # ---- 5. preview -------------------------------------------------------------------------------------------------
    _u = np.unique(img_ilv)
    pv_il = mo.ui.number(value=int(_u[len(_u) // 2]), start=int(_u[0]), stop=int(_u[-1]), label="Preview inline")
    mo.vstack([mo.md("## 5. Preview"), pv_il])
    return (pv_il,)


@app.cell(hide_code=True)
def _(antialias, depth_to_time, img, img_axis_m, img_ilv, img_len, img_xlv, mo, np, out_dt, out_tmax, plt, pv_il,
      records, samples_of, time, twt_of, vel, vel_of, vel_ok):
    mo.stop(not vel_ok, mo.callout(mo.md("The velocity file is not a velocity (step 2)."), kind="danger"))
    _sel = np.flatnonzero(img_ilv == int(pv_il.value))
    mo.stop(len(_sel) == 0, mo.callout(mo.md(f"Inline {pv_il.value} is not in the image."), kind="warn"))
    _sel = _sel[np.argsort(img_xlv[_sel])]
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
    _fig, _axs = plt.subplots(1, 3, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1, 0.55]})
    _x0, _x1 = img_xlv[_sel][0], img_xlv[_sel][-1]
    _axs[0].imshow(_amp.T, aspect="auto", cmap="gray", vmin=-_clip, vmax=_clip, extent=[_x0, _x1, _zax[-1], _zax[0]])
    _axs[0].set_title(f"Depth image - inline {pv_il.value}", fontsize=10)
    _axs[0].set_ylabel(f"Depth ({img_len.value})", fontsize=9)
    _axs[1].imshow(_tim.T, aspect="auto", cmap="gray", vmin=-_clip, vmax=_clip,
                   extent=[_x0, _x1, _out_t[-1] * 1000, _out_t[0] * 1000])
    _axs[1].set_title(f"Converted to time - inline {pv_il.value}", fontsize=10)
    _axs[1].set_ylabel("TWT (ms)", fontsize=9)
    _zl = float(np.median(_zend)) / _du
    _axs[0].axhline(_zl, color="#e34948", lw=1, ls="--")
    _tl = float(np.median(np.interp(np.median(_zend), img_axis_m, _twt[_k]))) * 1000
    _axs[1].axhline(_tl, color="#e34948", lw=1, ls="--")
    for _a in _axs[:2]:
        _a.set_xlabel("Crossline", fontsize=9)
        _a.tick_params(labelsize=8)
    _axs[2].plot(_twt[_k] * 1000, _zax, color="#2a78d6")
    _axs[2].axhspan(_zl, _zax[-1], color="#e34948", alpha=0.08)
    _axs[2].axhline(_zl, color="#e34948", lw=1, ls="--")
    _axs[2].invert_yaxis()
    _axs[2].set_xlabel("TWT (ms)", fontsize=9)
    _axs[2].set_ylabel(f"Depth ({img_len.value})", fontsize=9)
    _axs[2].set_title(f"Time-depth, XL {img_xlv[_sel][_k]}", fontsize=10)
    _axs[2].tick_params(labelsize=8)
    _axs[2].grid(alpha=0.3)
    _fig.tight_layout()
    _deep = bool(np.median(_zend) < img_axis_m[-1])
    mo.vstack([
        mo.md(f"{len(_sel)} traces in {_secs:.1f} s · red dashed line = end of the velocity model "
              f"({_zl:,.0f} {img_len.value} ≈ {_tl:,.0f} ms)"),
        mo.callout(mo.md(f"⚠ Below **{_zl:,.0f} {img_len.value}** (≈ {_tl:,.0f} ms) the times come from the choice "
                         "in step 3, not from the velocity model - the deeper part is only as right as that "
                         "velocity."), kind="warn") if _deep else mo.md(""),
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
    write_btn = mo.ui.run_button(label="💾 Convert and write")
    mo.vstack([mo.md("## 6. Write the time image"), out_path, mo.hstack([overwrite, write_btn], justify="start", gap=1)])
    return out_path, overwrite, write_btn


@app.cell(hide_code=True)
def _(antialias, below, datetime, depth_to_time, encode_samples, grad_len, img, img_dz, img_il, img_len, img_xl,
      img_z0, match, mo, np, os, out_dt, out_fmt, out_path, out_tmax, overwrite, records, samples_of, save_settings,
      time, twt_of, vcap, vel, vel_box, vel_domain, vel_dz, vel_il, vel_len, vel_of, vel_ok, vel_type, vel_unit,
      vel_xl, vel_z0, vtable, write_btn):
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
        "out_tmax": out_tmax.value, "antialias": antialias.value, "out_fmt": out_fmt.value, "out_path": out_path.value})
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
                np.hstack([_hdr, encode_samples(_out, out_fmt.value)]).tofile(_f)
                _bar.update(increment=len(_sel))
    os.replace(_part, _dest)
    mo.callout(mo.md(f"✅ Wrote **{img['ntr']:,} traces** × {_ns} samples (TWT every {float(out_dt.value):g} ms) to "
                     f"`{_dest}` ({os.path.getsize(_dest) / 1e9:.2f} GB) in {time.perf_counter() - _t0:.0f} s. "
                     "Answers saved for this image."), kind="success")
    return


if __name__ == "__main__":
    app.run()
