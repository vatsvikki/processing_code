import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Migration velocity converter

    Load a migration velocity SEG-Y, tell it what the velocity is (**interval / RMS / average**, in **time or depth**,
    sample interval, units), check the **grid and corner points**, choose the **conversion** you want (type, domain,
    sampling, units, grid range), preview one inline, then write the converted SEG-Y.

    **1.** Load &nbsp;→&nbsp; **2.** Header bytes & grid &nbsp;→&nbsp; **3.** Input velocity &nbsp;→&nbsp;
    **4.** Corner points &nbsp;→&nbsp; **5.** Output &nbsp;→&nbsp; **6.** Preview &nbsp;→&nbsp; **7.** Write.
    Your answers are saved for this file (`migration_velocity_settings.json`) and come back next time.

    Time is **two-way time (TWT)** everywhere. Velocities are converted with Dix (RMS → interval), the time-depth
    relation of the interval velocity, and its time integrals (interval → RMS / average).
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

    return corners_from_text, fit_affine, ilxl_to_xy


@app.cell
def _(np):
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
        sample - the interval velocity of sample i holds between samples i-1 and i. Also the count of samples where
        Dix / the derivative gave an impossible (negative) value, clipped."""
        v = np.maximum(np.asarray(v, np.float64), vmin_clip)
        q = np.asarray(axis, np.float64)
        dq = np.diff(q)
        bad = 0
        if domain == DOMAINS[0]:                               # time domain: q = TWT
            t = np.broadcast_to(q, v.shape)
            if vtype == "Interval":
                vint = v.copy()
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
        if vtype == "Interval":
            vint = v.copy()
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

    def from_interval(vint, t, z, vtype, domain, out_axis):
        """Interval velocity with its TWT / depth at every sample -> the velocity of type vtype on out_axis (TWT s or
        depth m). Below the input the last interval velocity is held."""
        n, ns = vint.shape
        far = 1.0e3
        # (origin prepended so that every output from 0 down is covered; one far sample appended to extrapolate)
        V = np.column_stack([vint[:, :1], vint, vint[:, -1:]])
        T = np.column_stack([np.zeros(n), t, t[:, -1:] + far])
        Z = np.column_stack([np.zeros(n), z, z[:, -1:] + vint[:, -1:] * far / 2])
        S = np.zeros_like(T)                                   # integral of Vint^2 dt from 0
        S[:, 1:] = np.cumsum(V[:, 1:] ** 2 * np.diff(T, axis=1), axis=1)
        q = np.asarray(out_axis, np.float64)
        out = np.empty((n, len(q)))
        for r in range(n):
            if domain == DOMAINS[0]:
                tq = q
                zq = np.interp(q, T[r], Z[r])
                idx = np.searchsorted(T[r], q, side="left")
            else:
                zq = q
                tq = np.interp(q, Z[r], T[r])
                idx = np.searchsorted(Z[r], q, side="left")
            idx = np.clip(idx, 1, V.shape[1] - 1)
            if vtype == "Interval":
                out[r] = V[r, idx]
            elif vtype == "RMS":
                sq = np.interp(tq, T[r], S[r])
                out[r] = np.where(tq > 0, np.sqrt(np.maximum(sq, 0) / np.maximum(tq, 1e-12)), V[r, 1])
            else:
                out[r] = np.where(tq > 0, 2 * zq / np.maximum(tq, 1e-12), V[r, 1])
        return out

    def convert(values, spec):
        """The whole conversion of a block of traces [n, ns_in] (input units) -> [n, ns_out] (output units).
        spec: in_type, in_domain, in_axis (ms or depth in in_len), in_vel, in_len, out_type, out_domain, out_axis,
        out_vel, out_len, smooth, vmin, vmax. Returns (converted, impossible-value count, reach) - reach = how far
        down (output axis units) the shallowest-ending trace of the input goes; below it the last velocity is held."""
        v = np.asarray(values, np.float64) * VEL_UNITS[spec["in_vel"]]
        if spec["in_domain"] == DOMAINS[0]:
            ax_in = np.asarray(spec["in_axis"]) / 1000.0
        else:
            ax_in = np.asarray(spec["in_axis"]) * LEN_UNITS[spec["in_len"]]
        vint, t, z, bad = to_interval(v, ax_in, spec["in_type"], spec["in_domain"], spec.get("smooth", 0))
        if spec["out_domain"] == DOMAINS[0]:
            ax_out = np.asarray(spec["out_axis"]) / 1000.0
        else:
            ax_out = np.asarray(spec["out_axis"]) * LEN_UNITS[spec["out_len"]]
        out = from_interval(vint, t, z, spec["out_type"], spec["out_domain"], ax_out) / VEL_UNITS[spec["out_vel"]]
        reach = float(t[:, -1].min() * 1000.0 if spec["out_domain"] == DOMAINS[0]
                      else z[:, -1].min() / LEN_UNITS[spec["out_len"]])
        lo, hi = spec.get("vmin", 0) or 0, spec.get("vmax", 0) or 0
        if lo > 0 or hi > 0:
            out = np.clip(out, lo if lo > 0 else None, hi if hi > 0 else None)
        return out, bad, reach

    return DOMAINS, LEN_UNITS, TYPES, VEL_UNITS, convert


@app.cell
def _(Path, json):
    # ---- the answers given for each file, kept for next time --------------------------------------------------------
    SETTINGS_FILE = Path(__file__).resolve().with_name("migration_velocity_settings.json")

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

    return SETTINGS_FILE, load_settings, save_settings


@app.cell(hide_code=True)
def _(load_settings, mo):
    # ---- 1. load -----------------------------------------------------------------------------------------------------
    _saved = load_settings()
    file_box = mo.ui.text(value=_saved.get("last_file", ""), label="Migration velocity SEG-Y file", full_width=True)
    load_btn = mo.ui.run_button(label="📂 Load file")
    mo.vstack([mo.md("## 1. Load the velocity file"), file_box, load_btn])
    return file_box, load_btn


@app.cell(hide_code=True)
def _(file_box, load_btn, mo, segy_open):
    # the file's headers (re-read when Load is pressed)
    mo.stop(not load_btn.value and not file_box.value.strip(), mo.md("_Type the file path and press **Load file**._"))
    try:
        seg = segy_open(file_box.value)
        load_error = None
    except Exception as _e:
        seg, load_error = None, f"{type(_e).__name__}: {_e}"
    mo.stop(seg is None, mo.callout(mo.md(f"**Could not read the file** - {load_error}"), kind="danger"))
    _fmt = {1: "IBM float", 2: "int32", 3: "int16", 5: "IEEE float", 8: "int8"}[seg["fmt"]]
    mo.vstack([
        mo.md(f"**{seg['ntr']:,} traces** · {seg['ns']} samples · sample interval field **{seg['dt']}** · {_fmt} · "
              f"{'big' if seg['order'] == '>' else 'little'}-endian · {seg['size'] / 1e9:.2f} GB"
              + (f" · ⚠ {seg['leftover']} bytes left over at the end" if seg["leftover"] else "")),
        mo.accordion({"Text header": mo.md("```\n" + "\n".join(l.rstrip() for l in seg["lines"]) + "\n```")}),
    ])
    return (seg,)


@app.cell(hide_code=True)
def _(load_settings, mo, seg):
    # ---- 2. header bytes -----------------------------------------------------------------------------------------------
    _s = load_settings().get(seg["path"], {})
    il_byte = mo.ui.number(value=_s.get("il_byte", 189), start=1, stop=237, label="Inline byte")
    xl_byte = mo.ui.number(value=_s.get("xl_byte", 193), start=1, stop=237, label="Crossline byte")
    x_byte = mo.ui.number(value=_s.get("x_byte", 181), start=1, stop=237, label="X byte")
    y_byte = mo.ui.number(value=_s.get("y_byte", 185), start=1, stop=237, label="Y byte")
    use_scalar = mo.ui.checkbox(value=_s.get("use_scalar", True), label="apply the coordinate scalar (byte 71)")
    mo.vstack([mo.md("## 2. Trace-header bytes (4-byte integers)"),
               mo.hstack([il_byte, xl_byte, x_byte, y_byte], justify="start", gap=1, wrap=True), use_scalar])
    return il_byte, use_scalar, x_byte, xl_byte, y_byte


@app.cell(hide_code=True)
def _(header_word, il_byte, mo, np, records, seg, use_scalar, x_byte, xl_byte, y_byte):
    # the grid read from the trace headers
    _rec = records(seg)
    _o = seg["order"]
    il = header_word(_rec, int(il_byte.value), "i4", _o)
    xl = header_word(_rec, int(xl_byte.value), "i4", _o)
    _sc = header_word(_rec, 71, "i2", _o).astype(np.float64)
    _f = np.where(_sc < 0, -1.0 / np.where(_sc == 0, 1, _sc), np.where(_sc > 0, _sc, 1.0)) if use_scalar.value else 1.0
    hx = header_word(_rec, int(x_byte.value), "i4", _o) * _f
    hy = header_word(_rec, int(y_byte.value), "i4", _o) * _f

    def _steps(v):
        u = np.unique(v)
        d = np.diff(u)
        return u, (int(np.min(d[d > 0])) if (d > 0).any() else 1)

    _ilu, il_step = _steps(il)
    _xlu, xl_step = _steps(xl)
    _full = len(_ilu) * len(_xlu)
    grid = {"il_min": int(_ilu[0]), "il_max": int(_ilu[-1]), "xl_min": int(_xlu[0]), "xl_max": int(_xlu[-1]),
            "il_step": il_step, "xl_step": xl_step, "n_il": len(_ilu), "n_xl": len(_xlu)}
    _dup = seg["ntr"] - len(np.unique(il * 10_000_000 + xl))
    mo.callout(mo.md(
        f"Inline **{grid['il_min']} – {grid['il_max']}** (step {il_step}, {len(_ilu)} lines) · crossline "
        f"**{grid['xl_min']} – {grid['xl_max']}** (step {xl_step}, {len(_xlu)} lines) · {seg['ntr']:,} traces of a "
        f"{_full:,}-trace grid" + (" (complete)" if _full == seg["ntr"] else f" ({seg['ntr'] / _full:.0%} filled)")
        + (f" · ⚠ {_dup} traces repeat an IL / XL" if _dup else "")
        + f"  \nX {hx.min():,.1f} – {hx.max():,.1f} · Y {hy.min():,.1f} – {hy.max():,.1f} (from the headers)"),
        kind="info" if not _dup else "warn")
    return grid, hx, hy, il, xl


@app.cell(hide_code=True)
def _(DOMAINS, TYPES, VEL_UNITS, load_settings, mo, re, seg):
    # ---- 3. the input velocity ----  (number boxes: start 0 / step 0.001 - marimo rounds a value to start + n * step)------------------------------------------------------------------------------------
    _s = load_settings().get(seg["path"], {})
    _txt = " ".join(seg["lines"]).upper()
    _ft = bool(re.search(r"\b(FT|FEET|FOOT)\b", _txt))
    _guess_dom = DOMAINS[1] if re.search(r"\bDEPTH\b|\bFT\b|\bFEET\b|SAMPLE RATE:\s*\d+\s*(FT|M)\b", _txt) else DOMAINS[0]
    _guess_type = ("RMS" if "RMS" in _txt else "Average" if "AVERAGE" in _txt else "Interval")
    in_type = mo.ui.dropdown(TYPES, value=_s.get("in_type", _guess_type), label="Velocity type")
    in_domain = mo.ui.dropdown(DOMAINS, value=_s.get("in_domain", _guess_dom), label="Vertical domain")
    in_dz = mo.ui.number(value=_s.get("in_dz", seg["dt"] / 1000.0), start=0, step=0.001,
                         label="Sample interval (ms for time, depth unit for depth)")
    in_z0 = mo.ui.number(value=_s.get("in_z0", 0.0), step=0.001, label="First sample at (ms / depth unit)")
    in_len = mo.ui.dropdown(["ft", "m"], value=_s.get("in_len", "ft" if _ft else "m"), label="Depth unit")
    in_vel = mo.ui.dropdown(list(VEL_UNITS), value=_s.get("in_vel", "ft/s" if _ft else "m/s"), label="Velocity unit")
    force_convert = mo.ui.checkbox(value=False, label="convert anyway (the values do not look like velocities)")
    mo.vstack([
        mo.md("## 3. What is in the file?  \n<span style='opacity:.7'>Guessed from the text header - check every "
              "value. The header's sample-interval field is "
              f"{seg['dt']} (µs for time; for depth usually the step × 1000).</span>"),
        mo.hstack([in_type, in_domain], justify="start", gap=1, wrap=True),
        mo.hstack([in_dz, in_z0], justify="start", gap=1, wrap=True),
        mo.hstack([in_len, in_vel], justify="start", gap=1, wrap=True),
    ])
    return force_convert, in_domain, in_dz, in_len, in_type, in_vel, in_z0


@app.cell(hide_code=True)
def _(DOMAINS, force_convert, in_domain, in_dz, in_len, in_type, in_vel, in_z0, mo, np, re, records, samples_of,
      seg):
    # a quick look at the values, so a wrong unit stands out
    _rec = records(seg)
    _pick = np.linspace(0, seg["ntr"] - 1, min(seg["ntr"], 200)).astype(np.int64)
    _v = samples_of(seg, _rec[_pick])
    in_axis = float(in_z0.value) + np.arange(seg["ns"]) * float(in_dz.value)
    _unit = "ms" if in_domain.value == DOMAINS[0] else in_len.value
    _lo, _hi = np.percentile(_v, [1, 99])
    _warn = ""
    if in_vel.value in ("ft/s", "m/s") and _hi < 50:
        _warn = " ⚠ values this small look like km/s or kft/s"
    elif in_vel.value in ("km/s", "kft/s") and _lo > 100:
        _warn = " ⚠ values this large look like m/s or ft/s"
    # a velocity is positive everywhere it is defined (0 = no data); an image / a stack swings around 0
    _live = _v[_v != 0]
    _neg = float((_live < 0).mean()) if _live.size else 0.0
    _what = next((l[3:].strip() for l in seg["lines"] if re.search(r"DESCRIPTION|PRODUCT|DATA TYPE", l.upper())), "")
    _image = bool(re.search(r"\b(STACK|RTM|MIGRATED|MIGRATION STACK|AMPLITUDE|REFLECTIVITY|GATHER|IMAGE)\b",
                            _what.upper())) and not re.search(r"VEL|VINT|VRMS", _what.upper())
    not_velocity = _neg > 0.01 or (_live.size and float(np.median(np.abs(_live))) < 1e-6)
    _msg = mo.callout(mo.md(f"**{in_type.value}** velocity in **{in_domain.value}**, {seg['ns']} samples from "
                            f"{in_axis[0]:g} to {in_axis[-1]:g} {_unit} · values (1-99 %, 200 traces) "
                            f"**{_lo:,.0f} – {_hi:,.0f} {in_vel.value}**{_warn}"),
                      kind="warn" if _warn else "info")
    if not_velocity:
        _msg = mo.vstack([mo.callout(mo.md(
            f"**This file does not look like a velocity model.** {_neg:.0%} of its samples are negative (a velocity "
            f"is never negative) - values {_lo:,.4g} to {_hi:,.4g}."
            + (f"  \nIts text header says: *{_what}*" if _what else "")
            + ("  \nThat is a seismic image (amplitudes), not a velocity: converting it would give meaningless "
               "numbers. Load the **velocity** file the migration used (interval / RMS velocity model)."
               if _image or _neg > 0.2 else "")
            + "  \nPreview and writing are stopped - tick *convert anyway* in step 3 only if you are sure."),
            kind="danger"), force_convert])
    elif _image:
        _msg = mo.vstack([_msg, mo.callout(mo.md(f"The text header describes this file as *{_what}* - check that it "
                                                 "is a velocity."), kind="warn")])
    blocked = bool(not_velocity and not force_convert.value)
    _msg
    return blocked, in_axis


@app.cell(hide_code=True)
def _(corners_from_text, grid, hx, hy, il, load_settings, mo, np, seg, xl):
    # ---- 4. corner points -------------------------------------------------------------------------------------------
    def _from_headers():
        rows = []
        for a in (grid["il_min"], grid["il_max"]):
            for b in (grid["xl_min"], grid["xl_max"]):
                k = int(np.argmin((il - a) ** 2 + (xl - b) ** 2))
                rows.append({"IL": int(il[k]), "XL": int(xl[k]), "X": round(float(hx[k]), 2), "Y": round(float(hy[k]), 2)})
        return rows

    _s = load_settings().get(seg["path"], {})
    _text = corners_from_text(seg["lines"])
    corner_source = ("saved for this file" if _s.get("corners") else
                          "text header" if _text else "trace headers (grid corners)")
    _rows = (_s.get("corners") or [{"IL": r[0], "XL": r[1], "X": r[2], "Y": r[3]} for r in _text] or _from_headers())
    corner_table = mo.ui.data_editor(_rows, label="Corner points (IL, XL ↔ X, Y) - edit, add or delete rows")
    corners_hdr = _from_headers()
    mo.vstack([mo.md(f"## 4. Corner points  \n<span style='opacity:.7'>Filled from the **{corner_source}**. At least "
                     "3 corners (not on one line); more are fitted by least squares.</span>"), corner_table])
    return corner_table, corners_hdr


@app.cell(hide_code=True)
def _(corner_table, fit_affine, hx, hy, il, ilxl_to_xy, mo, np, plt, xl):
    # the fit, the bin sizes, and how well the trace-header X / Y agree with it - and a map
    _rows = corner_table.value
    _rows = _rows.to_dict("records") if hasattr(_rows, "to_dict") else list(_rows or [])
    corner_rows = _rows
    corner_fit = fit_affine(_rows)
    if corner_fit is None:
        _msg = mo.callout(mo.md("**The corner points do not define a grid** - give at least 3 corners that are not on "
                                "one line."), kind="danger")
        _fig = None
    else:
        _px, _py = ilxl_to_xy(corner_fit, il.astype(float), xl.astype(float))
        _mis = np.hypot(_px - hx, _py - hy)
        _has = np.ptp(hx) > 0 or np.ptp(hy) > 0
        _msg = mo.callout(mo.md(
            f"Bin size: **{corner_fit['il_bin']:.2f}** per inline step, **{corner_fit['xl_bin']:.2f}** per crossline "
            f"step · inline direction {corner_fit['il_az']:.1f}°, crossline direction {corner_fit['xl_az']:.1f}° (from "
            f"north) · corner misfit {corner_fit['rms']:.2f} (max {corner_fit['max']:.2f})"
            + (f"  \nTrace-header X / Y against the corners: median {np.median(_mis):.2f}, max {_mis.max():.2f}"
               if _has else "  \nThe trace headers have no X / Y - the corners are the only geometry.")),
            kind="success" if (not _has or np.median(_mis) < max(corner_fit["il_bin"], corner_fit["xl_bin"])) else "warn")
        _fig, _ax = plt.subplots(figsize=(7.5, 6))
        _k = np.linspace(0, len(il) - 1, min(len(il), 20000)).astype(np.int64)
        if _has:
            _ax.scatter(hx[_k], hy[_k], s=1, color="#9aa3b2", label="traces (header X / Y)")
        _ilc = [il.min(), il.min(), il.max(), il.max(), il.min()]
        _xlc = [xl.min(), xl.max(), xl.max(), xl.min(), xl.min()]
        _ox, _oy = ilxl_to_xy(corner_fit, np.array(_ilc, float), np.array(_xlc, float))
        _ax.plot(_ox, _oy, color="#e34948", lw=1.6, label="grid outline from the corners")
        for _r in _rows:
            try:
                _ax.plot(float(_r["X"]), float(_r["Y"]), "o", color="#2a78d6")
                _ax.annotate(f"IL {_r['IL']}\nXL {_r['XL']}", (float(_r["X"]), float(_r["Y"])), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
            except (KeyError, TypeError, ValueError):
                pass
        _ax.set_aspect("equal")
        _ax.ticklabel_format(useOffset=False, style="plain")
        _ax.tick_params(labelsize=8)
        _ax.legend(fontsize=8, loc="best")
        _ax.set_title("Grid and corner points", fontsize=10)
        _fig.tight_layout()
    mo.vstack([_msg] + ([_fig] if _fig is not None else []))
    return corner_fit, corner_rows


@app.cell(hide_code=True)
def _(DOMAINS, TYPES, VEL_UNITS, grid, in_domain, in_dz, in_len, in_type, in_vel, load_settings, mo, seg):
    # ---- 5. what you want -------------------------------------------------------------------------------------------
    _s = load_settings().get(seg["path"], {})
    _other = DOMAINS[0] if in_domain.value == DOMAINS[1] else DOMAINS[1]
    out_type = mo.ui.dropdown(TYPES, value=_s.get("out_type", "RMS" if in_type.value != "RMS" else "Interval"),
                              label="Output velocity type")
    out_domain = mo.ui.dropdown(DOMAINS, value=_s.get("out_domain", _other), label="Output domain")
    out_dz = mo.ui.number(value=_s.get("out_dz", 4.0 if _other == DOMAINS[0] else float(in_dz.value)), start=0,
                          step=0.001, label="Output sample interval (ms / depth unit)")
    out_max = mo.ui.number(value=_s.get("out_max", 6000.0), start=0, step=0.001,
                           label="Output record length (ms / depth unit)")
    out_len = mo.ui.dropdown(["ft", "m"], value=_s.get("out_len", in_len.value), label="Output depth unit")
    out_vel = mo.ui.dropdown(list(VEL_UNITS), value=_s.get("out_vel", in_vel.value), label="Output velocity unit")
    smooth = mo.ui.number(value=_s.get("smooth", 0), start=0, stop=101, step=1,
                          label="Smooth the interval velocity over N samples (0 = off; damps Dix noise)")
    vmin = mo.ui.number(value=_s.get("vmin", 0.0), start=0, step=0.001, label="Clip output below (0 = off)")
    vmax = mo.ui.number(value=_s.get("vmax", 0.0), start=0, step=0.001, label="Clip output above (0 = off)")
    il_from = mo.ui.number(value=_s.get("il_from", grid["il_min"]), label="Inline from")
    il_to = mo.ui.number(value=_s.get("il_to", grid["il_max"]), label="to")
    il_every = mo.ui.number(value=_s.get("il_every", 1), start=1, step=1, label="every Nth")
    xl_from = mo.ui.number(value=_s.get("xl_from", grid["xl_min"]), label="Crossline from")
    xl_to = mo.ui.number(value=_s.get("xl_to", grid["xl_max"]), label="to")
    xl_every = mo.ui.number(value=_s.get("xl_every", 1), start=1, step=1, label="every Nth")
    write_xy = mo.ui.checkbox(value=_s.get("write_xy", True),
                              label="write CDP X / Y (bytes 181 / 185, scalar -100 in byte 71) from the corner points")
    out_fmt = mo.ui.dropdown(["IBM", "IEEE"], value=_s.get("out_fmt", "IBM"), label="Sample format")
    mo.vstack([
        mo.md("## 5. What do you want?"),
        mo.hstack([out_type, out_domain], justify="start", gap=1, wrap=True),
        mo.hstack([out_dz, out_max], justify="start", gap=1, wrap=True),
        mo.hstack([out_len, out_vel], justify="start", gap=1, wrap=True),
        mo.hstack([smooth, vmin, vmax], justify="start", gap=1, wrap=True),
        mo.md("**Output grid** (the input grid by default)"),
        mo.hstack([il_from, il_to, il_every], justify="start", gap=1, wrap=True),
        mo.hstack([xl_from, xl_to, xl_every], justify="start", gap=1, wrap=True),
        write_xy, out_fmt,
    ])
    return (il_every, il_from, il_to, out_domain, out_dz, out_fmt, out_len, out_max, out_type, out_vel, smooth, vmax,
            vmin, write_xy, xl_every, xl_from, xl_to)


@app.cell(hide_code=True)
def _(DOMAINS, in_axis, in_domain, in_len, in_type, in_vel, np, out_domain, out_dz, out_len, out_max, out_type,
      out_vel, smooth, vmax, vmin):
    # the conversion as data (used by the preview and the writer)
    out_axis = np.arange(0.0, float(out_max.value) + 1e-9, float(out_dz.value))
    spec = {"in_type": in_type.value, "in_domain": in_domain.value, "in_axis": in_axis, "in_vel": in_vel.value,
            "in_len": in_len.value, "out_type": out_type.value, "out_domain": out_domain.value, "out_axis": out_axis,
            "out_vel": out_vel.value, "out_len": out_len.value, "smooth": int(smooth.value or 0),
            "vmin": float(vmin.value or 0), "vmax": float(vmax.value or 0)}
    out_unit = "ms" if out_domain.value == DOMAINS[0] else out_len.value
    in_unit = "ms" if in_domain.value == DOMAINS[0] else in_len.value
    return in_unit, out_axis, out_unit, spec


@app.cell(hide_code=True)
def _(grid, mo):
    # ---- 6. preview -----------------------------------------------------------------------------------------------
    pv_il = mo.ui.number(value=(grid["il_min"] + grid["il_max"]) // 2, start=grid["il_min"], stop=grid["il_max"],
                         label="Preview inline")
    pv_xl = mo.ui.number(value=(grid["xl_min"] + grid["xl_max"]) // 2, start=grid["xl_min"], stop=grid["xl_max"],
                         label="trace at crossline")
    mo.vstack([mo.md("## 6. Preview"), mo.hstack([pv_il, pv_xl], justify="start", gap=1)])
    return pv_il, pv_xl


@app.cell(hide_code=True)
def _(blocked, convert, il, in_axis, in_type, in_unit, in_vel, mo, np, out_axis, out_type, out_unit, out_vel, plt,
      pv_il, pv_xl, records, samples_of, seg, spec, time, xl):
    # one inline, input and output, and one trace's functions
    mo.stop(blocked, mo.callout(mo.md("Not a velocity file - see step 3."), kind="danger"))
    _sel = np.flatnonzero(il == int(pv_il.value))
    mo.stop(len(_sel) == 0, mo.callout(mo.md(f"Inline {pv_il.value} is not in the file."), kind="warn"))
    _sel = _sel[np.argsort(xl[_sel])]
    _vin = samples_of(seg, records(seg)[_sel])
    _t0 = time.perf_counter()
    _vout, _bad, _reach = convert(_vin, spec)
    _secs = time.perf_counter() - _t0
    _k = int(np.argmin(np.abs(xl[_sel] - int(pv_xl.value))))
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for _a, _data, _ax_v, _u, _ttl, _vu in ((_axs[0], _vin, in_axis, in_unit, f"Input: {in_type.value}", in_vel.value),
                                           (_axs[1], _vout, out_axis, out_unit, f"Output: {out_type.value}", out_vel.value)):
        _im = _a.imshow(_data.T, aspect="auto", cmap="jet",
                        extent=[xl[_sel][0], xl[_sel][-1], _ax_v[-1], _ax_v[0]])
        _a.axvline(xl[_sel][_k], color="white", lw=1, ls="--")
        _a.set_title(f"{_ttl} - inline {pv_il.value}", fontsize=10)
        _a.set_xlabel("Crossline", fontsize=9)
        _a.set_ylabel(_u, fontsize=9)
        _a.tick_params(labelsize=8)
        _fig.colorbar(_im, ax=_a, fraction=0.04).set_label(_vu, fontsize=8)
    _b = _axs[2]
    _b.plot(_vin[_k], in_axis, label=f"input {in_type.value} ({in_vel.value}, {in_unit})", color="#4a5263")
    _b2 = _b.twinx() if in_unit != out_unit else _b
    _b2.plot(_vout[_k], out_axis, label=f"output {out_type.value} ({out_vel.value}, {out_unit})", color="#e34948")
    _b.invert_yaxis()
    if _b2 is not _b:
        _b2.invert_yaxis()
        _b2.set_ylabel(out_unit, fontsize=9, color="#e34948")
    _b.set_ylabel(in_unit, fontsize=9)
    _b.set_xlabel("velocity", fontsize=9)
    _b.set_title(f"Trace IL {pv_il.value} / XL {xl[_sel][_k]}", fontsize=10)
    _b.tick_params(labelsize=8)
    _h1, _l1 = _b.get_legend_handles_labels()
    _h2, _l2 = (_b2.get_legend_handles_labels() if _b2 is not _b else ([], []))
    _b.legend(_h1 + _h2, _l1 + _l2, fontsize=7, loc="lower left")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"{len(_sel)} traces converted in {_secs:.2f} s · output {np.nanmin(_vout):,.0f} – {np.nanmax(_vout):,.0f} "
              f"{out_vel.value}"
              + (f"  \n⚠ The input reaches only **{_reach:,.0f} {out_unit}** on this inline; below that the last "
                 "interval velocity is held (extrapolated)." if _reach < out_axis[-1] else "")
              + (f" · ⚠ {_bad} impossible values (negative Dix / derivative) clipped - consider "
                                    "smoothing" if _bad else "")),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(Path, file_box, mo, out_domain, out_type, seg):
    # ---- 7. write --------------------------------------------------------------------------------------------------
    _p = Path(seg["path"])
    _tag = f"{out_type.value.lower()}_{'time' if out_domain.value.startswith('Time') else 'depth'}"
    out_path = mo.ui.text(value=str(_p.with_name(f"{_p.stem}_{_tag}.sgy")), label="Output SEG-Y file", full_width=True)
    overwrite = mo.ui.checkbox(value=False, label="overwrite if it exists")
    write_btn = mo.ui.run_button(label="💾 Convert and write")
    save_btn = mo.ui.run_button(label="Save my answers only")
    mo.vstack([mo.md("## 7. Write the converted file"), out_path, mo.hstack([overwrite, write_btn, save_btn],
                                                                             justify="start", gap=1)])
    return out_path, overwrite, save_btn, write_btn


@app.cell(hide_code=True)
def _(corner_rows, il_byte, il_every, il_from, il_to, in_domain, in_dz, in_len, in_type, in_vel, in_z0, out_domain,
      out_dz, out_fmt, out_len, out_max, out_type, out_vel, smooth, use_scalar, vmax, vmin, write_xy, x_byte, xl_byte,
      xl_every, xl_from, xl_to, y_byte):
    # every answer of the form, as saved for the file
    answers = {
        "il_byte": il_byte.value, "xl_byte": xl_byte.value, "x_byte": x_byte.value, "y_byte": y_byte.value,
        "use_scalar": use_scalar.value, "in_type": in_type.value, "in_domain": in_domain.value, "in_dz": in_dz.value,
        "in_z0": in_z0.value, "in_len": in_len.value, "in_vel": in_vel.value, "corners": corner_rows,
        "out_type": out_type.value, "out_domain": out_domain.value, "out_dz": out_dz.value, "out_max": out_max.value,
        "out_len": out_len.value, "out_vel": out_vel.value, "smooth": smooth.value, "vmin": vmin.value,
        "vmax": vmax.value, "il_from": il_from.value, "il_to": il_to.value, "il_every": il_every.value,
        "xl_from": xl_from.value, "xl_to": xl_to.value, "xl_every": xl_every.value, "write_xy": write_xy.value,
        "out_fmt": out_fmt.value,
    }
    return (answers,)


@app.cell(hide_code=True)
def _(answers, mo, save_btn, save_settings, seg):
    mo.stop(not save_btn.value)
    save_settings(seg["path"], answers)
    mo.callout(mo.md("Answers saved for this file - they come back when it is loaded again."), kind="success")
    return


@app.cell(hide_code=True)
def _(DOMAINS, answers, blocked, convert, corner_fit, datetime, encode_samples, il, il_every, il_from, il_to, ilxl_to_xy,
      in_unit, mo, np, os, out_axis, out_domain, out_dz, out_fmt, out_path, out_type, out_unit, out_vel, overwrite,
      records, samples_of, save_settings, seg, spec, time, write_btn, write_xy, xl, xl_every, xl_from, xl_to):
    mo.stop(not write_btn.value)
    mo.stop(blocked, mo.callout(mo.md("Not written: the file does not look like a velocity model (step 3)."),
                                kind="danger"))
    _dest = os.path.abspath(os.path.expanduser(out_path.value.strip()))
    mo.stop(_dest == seg["path"], mo.callout(mo.md("The output must not be the input file."), kind="danger"))
    mo.stop(os.path.exists(_dest) and not overwrite.value,
            mo.callout(mo.md(f"`{_dest}` exists - tick **overwrite** or change the name."), kind="warn"))
    mo.stop(write_xy.value and corner_fit is None,
            mo.callout(mo.md("Writing X / Y needs valid corner points (step 4)."), kind="danger"))
    save_settings(seg["path"], answers)

    # which traces: the output grid, in file order
    _keep = ((il >= int(il_from.value)) & (il <= int(il_to.value)) & (xl >= int(xl_from.value))
             & (xl <= int(xl_to.value)) & ((il - int(il_from.value)) % int(il_every.value) == 0)
             & ((xl - int(xl_from.value)) % int(xl_every.value) == 0))
    _idx = np.flatnonzero(_keep)
    mo.stop(len(_idx) == 0, mo.callout(mo.md("No trace inside the output grid."), kind="warn"))
    _ns_out = len(out_axis)
    mo.stop(_ns_out > 65535, mo.callout(mo.md("More than 65535 output samples - lower the record length."), kind="danger"))
    _is_time = out_domain.value == DOMAINS[0]
    _dt_field = int(round(float(out_dz.value) * 1000))         # µs for time; the depth step × 1000 for depth
    mo.stop(_dt_field > 65535, mo.callout(mo.md("The sample interval is too large for the SEG-Y header."), kind="danger"))

    # text header: what was done, then the original header
    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _new = [f"MIGRATION VELOCITY CONVERTED {_now}",
            f"INPUT: {os.path.basename(seg['path'])}"[:76],
            f"IN : {spec['in_type'].upper()} {('TIME' if spec['in_domain'] == DOMAINS[0] else 'DEPTH')} "
            f"{spec['in_vel'].upper()} DZ {spec['in_axis'][1] - spec['in_axis'][0]:g} {in_unit.upper()}",
            f"OUT: {out_type.value.upper()} {'TIME (TWT)' if _is_time else 'DEPTH'} {out_vel.value.upper()} "
            f"DZ {float(out_dz.value):g} {out_unit.upper()} NS {_ns_out}",
            f"DT FIELD = SAMPLE INTERVAL X 1000 ({'MICROSECONDS' if _is_time else out_unit.upper() + ' X 1000'})",
            f"SAMPLE FORMAT {out_fmt.value} FLOAT, GRID IL {int(il_from.value)}-{int(il_to.value)}/{int(il_every.value)} "
            f"XL {int(xl_from.value)}-{int(xl_to.value)}/{int(xl_every.value)}"]
    if corner_fit is not None:
        _new.append(f"BIN {corner_fit['il_bin']:.2f} (IL) X {corner_fit['xl_bin']:.2f} (XL), CORNERS IL XL X Y:")
        for _r in answers["corners"][:4]:
            try:
                _new.append(f"  {float(_r['IL']):8.0f} {float(_r['XL']):8.0f} {float(_r['X']):14.2f} {float(_r['Y']):14.2f}")
            except (KeyError, TypeError, ValueError):
                pass
    _new.append("ORIGINAL TEXT HEADER:")
    _old = [l[3:].rstrip() if l[:1] == "C" else l.rstrip() for l in seg["lines"]]
    _cards = (_new + _old)[:39] + ["END TEXTUAL HEADER"]
    _text = "".join(f"C{_i + 1:2d} {_c}"[:80].ljust(80) for _i, _c in enumerate(_cards)).encode("cp037")

    _bin = bytearray(seg["binary"])
    if seg["order"] == "<":                                   # (the output is always big-endian)
        for _o, _n in ((12, 2), (16, 2), (18, 2), (20, 2), (22, 2), (24, 2), (304, 2)):
            _bin[_o:_o + _n] = _bin[_o:_o + _n][::-1]
    _bin[16:18] = _dt_field.to_bytes(2, "big")
    _bin[18:20] = _dt_field.to_bytes(2, "big")
    _bin[20:22] = _ns_out.to_bytes(2, "big")
    _bin[22:24] = _ns_out.to_bytes(2, "big")
    _bin[24:26] = (1 if out_fmt.value == "IBM" else 5).to_bytes(2, "big")
    _bin[304:306] = (0).to_bytes(2, "big")                     # no extended text headers

    _rec = records(seg)
    _chunk = 2000
    _t0 = time.perf_counter()
    _bad = 0
    _reach = float("inf")
    _part = _dest + ".part"
    with open(_part, "wb") as _f:
        _f.write(_text)
        _f.write(bytes(_bin))
        with mo.status.progress_bar(total=len(_idx), title="Converting and writing", show_eta=True,
                                    show_rate=True) as _bar:
            for _a in range(0, len(_idx), _chunk):
                _sel = _idx[_a:_a + _chunk]
                _block = np.array(_rec[_sel])
                _vout, _b, _r = convert(samples_of(seg, _block), spec)
                _reach = min(_reach, _r)
                _bad += _b
                _hdr = _block[:, :240].copy()
                if seg["order"] == "<":                     # header words to big-endian (4-byte, then 2-byte fields)
                    _w = _hdr.reshape(len(_sel), 60, 4)[:, :, ::-1].reshape(len(_sel), 240).copy()
                    _hdr = _w
                _hdr[:, 114:116] = np.frombuffer(_ns_out.to_bytes(2, "big"), np.uint8)
                _hdr[:, 116:118] = np.frombuffer(_dt_field.to_bytes(2, "big"), np.uint8)
                _hdr[:, 108:110] = 0                            # delay: the output starts at 0
                if write_xy.value:
                    _x, _y = ilxl_to_xy(corner_fit, il[_sel].astype(float), xl[_sel].astype(float))
                    _hdr[:, 70:72] = np.frombuffer((-100).to_bytes(2, "big", signed=True), np.uint8)
                    _hdr[:, 180:184] = np.round(_x * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                    _hdr[:, 184:188] = np.round(_y * 100).astype(">i4").view(np.uint8).reshape(-1, 4)
                np.hstack([_hdr, encode_samples(_vout, out_fmt.value)]).tofile(_f)
                _bar.update(increment=len(_sel))
    os.replace(_part, _dest)
    _secs = time.perf_counter() - _t0
    mo.callout(mo.md(
        f"✅ Wrote **{len(_idx):,} traces** × {_ns_out} samples ({out_type.value} velocity in "
        f"{'time' if _is_time else 'depth'}, {out_vel.value}, every {float(out_dz.value):g} {out_unit}) to `{_dest}` "
        f"({os.path.getsize(_dest) / 1e9:.2f} GB) in {_secs:.0f} s."
        + (f"  \n⚠ {_bad:,} impossible values (negative Dix / derivative) were clipped." if _bad else "")
        + (f"  \nℹ The input reaches {_reach:,.0f} {out_unit} (shallowest trace); below that the last interval "
           "velocity is held." if _reach < out_axis[-1] else "")
        + "  \nYour answers were saved for this file."), kind="success")
    return


if __name__ == "__main__":
    app.run()
