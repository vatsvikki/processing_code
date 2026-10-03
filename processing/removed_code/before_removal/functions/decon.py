"""Spiking (Wiener-Levinson) deconvolution and its QC statistics (numpy only, no UI).

For every trace the autocorrelation inside a design window is computed, its zero lag is
raised by the pre-whitening percentage, and the Levinson-Durbin recursion gives the
prediction-error filter a = [1, a1 .. aL].  Convolving the trace with `a` is the
zero-lag-spiking deconvolution (unit prediction distance): it flattens the amplitude
spectrum and compresses the wavelet towards a spike.  All traces are solved at once.
"""
from __future__ import annotations

import numpy as np


def _window(ns: int, dt_ms: float, t_start_ms: float, t_end_ms: float) -> slice:
    i0 = int(round(max(t_start_ms, 0.0) / dt_ms))
    i1 = ns if t_end_ms <= 0 else min(int(round(t_end_ms / dt_ms)), ns)
    if i1 - i0 < 8:
        raise ValueError(f"design window {t_start_ms:g}-{t_end_ms:g} ms holds fewer than 8 samples")
    return slice(i0, i1)


def autocorrelation(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation of each row for lags 0..max_lag  ->  [ntr, max_lag + 1] (not normalised)."""
    n = data.shape[1]
    nfft = 1 << int(np.ceil(np.log2(n + max_lag + 1)))
    spec = np.fft.rfft(data, nfft, axis=1)
    return np.fft.irfft(spec.real ** 2 + spec.imag ** 2, nfft, axis=1)[:, :max_lag + 1]


def prediction_error_filter(r: np.ndarray) -> np.ndarray:
    """Levinson-Durbin for all rows at once.  r: [ntr, L+1] autocorrelations -> a: [ntr, L+1], a[:, 0] = 1.

    A row with r[0] == 0 (a dead trace) gets the identity filter [1, 0, ... 0].
    """
    ntr, n = r.shape
    a = np.zeros((ntr, n))
    a[:, 0] = 1.0
    err = np.where(r[:, 0] > 0, r[:, 0], 1.0)
    for m in range(1, n):
        acc = r[:, m] + (a[:, 1:m] * r[:, m - 1:0:-1]).sum(axis=1)
        k = np.where(r[:, 0] > 0, -acc / err, 0.0)
        prev = a[:, 1:m].copy()
        a[:, 1:m] = prev + k[:, None] * prev[:, ::-1]
        a[:, m] = k
        err = err * (1.0 - k * k)
        err = np.where(err > 1e-30, err, 1e-30)
    return a


def spiking_decon(
    data: np.ndarray,
    dt_ms: float,
    *,
    operator_ms: float = 160.0,
    prewhite_pct: float = 0.1,
    t_start_ms: float = 0.0,
    t_end_ms: float = 0.0,
    balance: bool = True,
) -> np.ndarray:
    """Spiking deconvolution of every trace of `data` [ntr, ns] -> float32 [ntr, ns].

    operator_ms   filter length
    prewhite_pct  white-noise added to the autocorrelation zero lag, in percent (stabilises the solution)
    t_start_ms / t_end_ms  design window (end <= 0 = end of trace); the filter is applied to the whole trace
    balance       rescale every output trace to the RMS of its input trace
    """
    data = np.asarray(data, np.float64)
    ntr, ns = data.shape
    lag = int(round(operator_ms / dt_ms))
    if lag < 1:
        raise ValueError(f"operator length {operator_ms:g} ms is shorter than one sample ({dt_ms:g} ms)")
    if lag >= ns // 2:
        raise ValueError(f"operator length {operator_ms:g} ms is too long for a {ns * dt_ms:g} ms record")
    win = _window(ns, dt_ms, t_start_ms, t_end_ms)

    r = autocorrelation(data[:, win], lag)
    r[:, 0] *= 1.0 + prewhite_pct / 100.0
    a = prediction_error_filter(r)

    nfft = 1 << int(np.ceil(np.log2(ns + lag)))
    out = np.fft.irfft(np.fft.rfft(data, nfft, axis=1) * np.fft.rfft(a, nfft, axis=1), nfft, axis=1)[:, :ns]

    if balance:
        rms_in = np.sqrt((data ** 2).mean(axis=1))
        rms_out = np.sqrt((out ** 2).mean(axis=1))
        out *= np.divide(rms_in, rms_out, out=np.ones(ntr), where=rms_out > 0)[:, None]
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# QC statistics for before / after comparison
# --------------------------------------------------------------------------
def _live(data: np.ndarray) -> np.ndarray:
    rms = np.sqrt((np.asarray(data, np.float64) ** 2).mean(axis=1))
    return rms > 0


def mean_amplitude_spectrum(data: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """(frequency in Hz, mean amplitude spectrum of the live traces, each trace normalised to its own peak)."""
    live = _live(data)
    x = np.asarray(data, np.float64)[live] if live.any() else np.asarray(data, np.float64)
    amp = np.abs(np.fft.rfft(x, axis=1))
    amp /= np.maximum(amp.max(axis=1, keepdims=True), 1e-30)
    return np.fft.rfftfreq(x.shape[1], dt_ms / 1000.0), amp.mean(axis=0)


def mean_autocorrelation(data: np.ndarray, dt_ms: float, max_lag_ms: float = 400.0) -> tuple[np.ndarray, np.ndarray]:
    """(lag in ms, mean normalised autocorrelation of the live traces)."""
    live = _live(data)
    x = np.asarray(data, np.float64)[live] if live.any() else np.asarray(data, np.float64)
    lag = min(int(round(max_lag_ms / dt_ms)), x.shape[1] // 2)
    r = autocorrelation(x, lag)
    r /= np.maximum(r[:, :1], 1e-30)
    return np.arange(lag + 1) * dt_ms, r.mean(axis=0)
