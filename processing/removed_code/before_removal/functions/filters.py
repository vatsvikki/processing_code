"""Simple per-trace corrections used by the pipeline steps: gain, filter, mute (no UI).  Arrays are float32 [ntr, ns]."""
from __future__ import annotations

import numpy as np


def agc(data: np.ndarray, dt_ms: float, win_ms: float) -> np.ndarray:
    """Simple RMS automatic gain control: every sample divided by the RMS of a `win_ms` window centred on it."""
    n = max(int(round(win_ms / dt_ms)) | 1, 3)
    sq = np.pad(np.square(data, dtype=np.float64), ((0, 0), (n // 2, n // 2)), mode="edge")
    cs = np.cumsum(sq, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    mean = (cs[:, n:] - cs[:, :-n]) / n
    return (data / np.sqrt(mean + 1e-30 * (1 + mean.max()))).astype(np.float32)


def spreading_gain(data: np.ndarray, dt_ms: float, power: float) -> np.ndarray:
    """Geometrical-spreading (t^power) gain: every sample multiplied by (t / 1 s)^power, t at least one sample."""
    t = np.maximum(np.arange(data.shape[1]) * dt_ms, dt_ms) / 1000.0
    return (data * (t ** power)[None, :]).astype(np.float32)


def bandpass(data: np.ndarray, dt_ms: float, f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    """Zero-phase trapezoidal band-pass (Ormsby): 0 below f1, ramp to 1 at f2, flat to f3, ramp to 0 at f4 (Hz)."""
    nyq = 500.0 / dt_ms
    if not (0 <= f1 < f2 < f3 < f4 <= nyq):
        raise ValueError(f"band-pass corners must satisfy 0 <= low cut < low pass < high pass < high cut <= {nyq:g} Hz "
                         f"(Nyquist), got {f1:g} / {f2:g} / {f3:g} / {f4:g}")
    ns = data.shape[1]
    nfft = int(2 ** np.ceil(np.log2(ns)))
    freqs = np.fft.rfftfreq(nfft, dt_ms / 1000.0)
    resp = np.interp(freqs, [0, f1, f2, f3, f4, nyq], [0, 0, 1, 1, 0, 0])
    return np.fft.irfft(np.fft.rfft(data, nfft, axis=1) * resp[None, :], nfft, axis=1)[:, :ns].astype(np.float32)


def top_mute(data: np.ndarray, dt_ms: float, offsets: np.ndarray, velocity: float, t0_ms: float = 0.0,
             taper_ms: float = 100.0) -> np.ndarray:
    """Linear top mute: samples above t = t0 + |offset| / velocity are zeroed, with a linear taper of `taper_ms` below."""
    t = np.arange(data.shape[1]) * dt_ms
    line = t0_ms + np.abs(offsets).astype(np.float64) / velocity * 1000.0                 # ms
    w = np.clip((t[None, :] - line[:, None]) / max(taper_ms, dt_ms), 0.0, 1.0)
    return (data * w).astype(np.float32)
