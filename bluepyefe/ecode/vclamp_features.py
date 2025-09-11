"""Simple VClamp feature extraction helpers"""

import numpy as np
from scipy.optimize import curve_fit


def extract_peak_current(t, current, ton, toff):
    """Return min (inward) and max (outward) peak between ton and toff."""
    if any(x is None for x in (t, current, ton, toff)):
        return {"min": None, "max": None}
    mask = (t >= ton) & (t <= toff)
    seg = current[mask]
    if seg.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.min(seg)), "max": float(np.max(seg))}


def extract_steady_state_current(t, current, toff, tail_ms=5.0):
    """Estimate steady-state current using the last tail_ms ms before toff."""
    if any(x is None for x in (t, current, toff)):
        return None
    dt = t[1] - t[0]
    n = max(1, int((tail_ms / 1000.0) / dt))  # tail_ms in ms
    end_idx = np.searchsorted(t, toff)
    start_idx = max(0, end_idx - n)
    seg = current[start_idx:end_idx]
    if seg.size == 0:
        return None
    return float(np.mean(seg))


def extract_charge(t, current, ton, toff):
    """Integrate current from ton to toff (area = charge). Result in A*s (Coulombs)."""
    if any(x is None for x in (t, current, ton, toff)):
        return None
    mask = (t >= ton) & (t <= toff)
    seg_t = t[mask]
    seg_i = current[mask]
    if seg_t.size < 2:
        return None
    return float(np.trapz(seg_i, seg_t))


def _exp(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_exponential_decay(t, current, start_t, end_t):
    """Fit single-exponential to [start_t, end_t] portion of current. Returns tau (s) or None."""
    if any(x is None for x in (t, current, start_t, end_t)):
        return None
    mask = (t >= start_t) & (t <= end_t)
    seg_t = t[mask]
    seg_i = current[mask]
    if seg_t.size < 5:
        return None
    # make time relative
    x = seg_t - seg_t[0]
    y = seg_i
    try:
        p0 = (y[0] - y[-1], 0.002, y[-1])  # initial guess: tau ~2 ms
        popt, _ = curve_fit(_exp, x, y, p0=p0, maxfev=2000)
        tau = float(abs(popt[1]))
        return tau
    except Exception:
        return None