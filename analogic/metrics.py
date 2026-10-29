"""Metric extraction from simulated waveforms (pure numpy, unit tested)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _interp_crossing(x: np.ndarray, y: np.ndarray, level: float, log_x: bool = False,
                     falling: bool | None = None) -> float:
    """First x at which y crosses ``level`` (linear or log-x interpolation)."""
    s = y - level
    idx = np.where(np.sign(s[:-1]) * np.sign(s[1:]) <= 0)[0]
    for i in idx:
        if s[i] == s[i + 1]:
            continue
        if falling is True and not s[i] > s[i + 1]:
            continue
        if falling is False and not s[i] < s[i + 1]:
            continue
        frac = s[i] / (s[i] - s[i + 1])
        if log_x:
            return float(10 ** (np.log10(x[i]) + frac * (np.log10(x[i + 1]) - np.log10(x[i]))))
        return float(x[i] + frac * (x[i + 1] - x[i]))
    return float("nan")


@dataclass
class AcMetrics:
    dc_gain_db: float
    f3db_hz: float
    ugb_hz: float
    phase_margin_deg: float
    gain_margin_db: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def ac_metrics(freq: np.ndarray, h: np.ndarray) -> AcMetrics:
    """Open-loop metrics from a complex transfer function H(f).

    Phase margin is 180 deg + phase(H) at the unity-gain frequency, with the
    phase unwrapped and referenced to the low-frequency phase (so an
    inverting or non-inverting measurement both work).
    """
    mag_db = 20 * np.log10(np.abs(h))
    phase = np.degrees(np.unwrap(np.angle(h)))
    phase = phase - phase[0]  # 0 deg at DC for the loop
    a0 = mag_db[0]
    f3 = _interp_crossing(freq, mag_db, a0 - 3.0, log_x=True, falling=True)
    ugb = _interp_crossing(freq, mag_db, 0.0, log_x=True, falling=True)
    if np.isnan(ugb):
        pm = float("nan")
    else:
        ph_u = np.interp(np.log10(ugb), np.log10(freq), phase)
        pm = 180.0 + ph_u
    f180 = _interp_crossing(freq, phase, -180.0, log_x=True, falling=True)
    gm = float("nan") if np.isnan(f180) else -float(np.interp(np.log10(f180), np.log10(freq), mag_db))
    return AcMetrics(float(a0), f3, ugb, float(pm), gm)


def slew_rate(t: np.ndarray, v: np.ndarray, lo: float = 0.2, hi: float = 0.8) -> float:
    """Slew rate (V/s) measured between ``lo`` and ``hi`` fractions of the step."""
    v0, v1 = v[0], v[-1]
    rising = v1 > v0
    a = v0 + lo * (v1 - v0)
    b = v0 + hi * (v1 - v0)
    ta = _interp_crossing(t, v, a, falling=not rising)
    tb = _interp_crossing(t, v, b, falling=not rising)
    return float(abs(b - a) / (tb - ta))


def settling_time(t: np.ndarray, v: np.ndarray, t_step: float, tol: float = 0.01) -> float:
    """Time after ``t_step`` until v stays within ``tol`` (fraction of step) of final."""
    v_final = v[-1]
    v_init = v[np.searchsorted(t, t_step) - 1] if t_step > t[0] else v[0]
    band = abs(v_final - v_init) * tol
    outside = np.where(np.abs(v - v_final) > band)[0]
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    if last + 1 >= len(t):
        return float("nan")
    return float(t[last + 1] - t_step)


def tempco_ppm(temp: np.ndarray, v: np.ndarray) -> float:
    """Box-method temperature coefficient in ppm/degC."""
    return float((v.max() - v.min()) / (np.mean(v) * (temp.max() - temp.min())) * 1e6)


@dataclass
class Linearity:
    lsb: float
    dnl: np.ndarray
    inl: np.ndarray
    dnl_max: float
    inl_max: float
    monotonic: bool


def dac_linearity(levels: np.ndarray, method: str = "endpoint") -> Linearity:
    """INL/DNL in LSB from measured output per code (index = code).

    ``endpoint``: straight line through first and last code.
    ``bestfit``: least-squares line.
    """
    y = np.asarray(levels, dtype=float)
    codes = np.arange(len(y))
    if method == "endpoint":
        lsb = (y[-1] - y[0]) / (len(y) - 1)
        ideal = y[0] + lsb * codes
    elif method == "bestfit":
        slope, icpt = np.polyfit(codes, y, 1)
        lsb = slope
        ideal = icpt + slope * codes
    else:
        raise ValueError(method)
    inl = (y - ideal) / lsb
    dnl = np.diff(y) / lsb - 1.0
    steps = np.diff(y) * np.sign(lsb)
    return Linearity(float(lsb), dnl, inl, float(np.max(np.abs(dnl))),
                     float(np.max(np.abs(inl))), bool(np.all(steps > 0)))


def lock_time(t: np.ndarray, err: np.ndarray, tol: float) -> float:
    """First time after which |err| stays below ``tol`` for the rest of the record."""
    bad = np.where(np.abs(err) > tol)[0]
    if len(bad) == 0:
        return float(t[0])
    if bad[-1] + 1 >= len(t):
        return float("nan")
    return float(t[bad[-1] + 1])


def sfdr_db(x: np.ndarray, signal_bin: int) -> float:
    """SFDR (dBc) from a coherently sampled record."""
    spec = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    sig = spec[signal_bin]
    spur = np.delete(spec, [0, signal_bin]).max()
    return float(10 * np.log10(sig / spur))
