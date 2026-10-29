"""Behavioural charge-pump PLL: loop-filter design, linear analysis and a
time-domain event-driven simulator.

The time-domain model uses the *measured* transistor-level VCO tuning curve
and charge-pump UP/DN current curves (from :mod:`analogic.pll`) as lookup
tables, so non-linear Kvco and current mismatch are included.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class FilterValues:
    r: float
    c1: float
    c2: float


def design_loop_filter(icp: float, kvco_hz: float, n: int, fc: float, pm_deg: float) -> FilterValues:
    """Second-order passive filter (R + C1 in series, C2 in parallel).

    Places the phase-margin peak at ``fc``: wz = wc/sqrt(1+b), wp = wc*sqrt(1+b)
    with b = C1/C2, and scales total capacitance for |T(j wc)| = 1.
    """
    pm = np.radians(pm_deg)
    b = brentq(lambda b: np.arctan(np.sqrt(1 + b)) - np.arctan(1 / np.sqrt(1 + b)) - pm, 1e-6, 1e6)
    wc = 2 * np.pi * fc
    wz = wc / np.sqrt(1 + b)
    wp = wc * np.sqrt(1 + b)
    mag = abs(1 + 1j * wc / wz) / abs(1 + 1j * wc / wp)
    ct = icp * kvco_hz * mag / (n * wc ** 2)
    c2 = ct / (1 + b)
    c1 = b * c2
    r = 1 / (wz * c1)
    return FilterValues(float(r), float(c1), float(c2))


def loop_gain(f: np.ndarray, icp: float, kvco_hz: float, n: int, fv: FilterValues) -> np.ndarray:
    """Open-loop gain T(jw) = Icp * Kvco * Z(jw) / (j w N)  (continuous-time approximation)."""
    s = 2j * np.pi * np.asarray(f)
    z = (1 + s * fv.r * fv.c1) / (s * (fv.c1 + fv.c2) * (1 + s * fv.r * fv.c1 * fv.c2 / (fv.c1 + fv.c2)))
    # Kpd = Icp/(2 pi) [A/rad], Kv = 2 pi Kvco [rad/s/V] -> the 2 pi factors cancel
    return icp * kvco_hz * z / (s * n)


def loop_margins(icp, kvco_hz, n, fv: FilterValues, f=None) -> tuple[float, float]:
    """(crossover frequency, phase margin in degrees)."""
    f = np.logspace(3, 9, 4000) if f is None else f
    t = loop_gain(f, icp, kvco_hz, n, fv)
    mag = np.abs(t)
    i = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0][0]
    lm0, lm1 = np.log(mag[i]), np.log(mag[i + 1])
    frac = lm0 / (lm0 - lm1)
    fc = float(np.exp(np.log(f[i]) + frac * (np.log(f[i + 1]) - np.log(f[i]))))
    tc = loop_gain(np.array([fc]), icp, kvco_hz, n, fv)[0]
    return fc, float(180 + np.degrees(np.angle(tc)))


@dataclass
class BehavioralConfig:
    fref: float = 25e6
    n_div: int = 16
    t_reset: float = 150e-12
    dt: float = 10e-12
    t_stop: float = 6e-6
    v_init: float = 0.0
    hop_n: int | None = None     # optional divider change (channel hop)
    t_hop: float = 0.0


def simulate(cfg: BehavioralConfig, fv: FilterValues, vco_v, vco_f, cp_v, cp_iup, cp_idn,
             ref_phase_noise: float = 0.0, seed: int = 0) -> dict:
    """Fixed-step event-driven PLL simulation.

    Returns time, vctrl, per-reference-cycle VCO frequency and phase error.
    """
    rng = np.random.default_rng(seed)
    vco_v = np.asarray(vco_v, float)
    vco_f = np.asarray(vco_f, float)
    cp_v = np.asarray(cp_v, float)
    iup_t = np.abs(np.asarray(cp_iup, float))
    idn_t = np.abs(np.asarray(cp_idn, float))
    dt = cfg.dt
    nsteps = int(cfg.t_stop / dt)
    tref = 1.0 / cfg.fref
    vc, v1 = cfg.v_init, cfg.v_init
    theta = 0.0  # VCO phase in cycles
    next_div = float(cfg.n_div)
    next_ref = tref + (rng.normal(0, ref_phase_noise) if ref_phase_noise else 0.0)
    up = dn = False
    reset_at = None
    t_up_edge = t_dn_edge = None
    rec_stride = max(1, int(1e-9 / dt))
    t_rec, vc_rec = [], []
    edges_ref, edges_div = [], []
    r, c1, c2 = fv.r, fv.c1, fv.c2
    for k in range(nsteps):
        t = k * dt
        f = np.interp(vc, vco_v, vco_f)
        theta_new = theta + f * dt
        if theta_new >= next_div:
            frac = (next_div - theta) / (theta_new - theta)
            te = t + frac * dt
            edges_div.append(te)
            n_cur = cfg.hop_n if (cfg.hop_n is not None and t >= cfg.t_hop) else cfg.n_div
            next_div += n_cur
            if not dn:
                dn = True
                t_dn_edge = te
        theta = theta_new
        if t + dt >= next_ref:
            edges_ref.append(next_ref)
            if not up:
                up = True
                t_up_edge = next_ref
            next_ref += tref + (rng.normal(0, ref_phase_noise) if ref_phase_noise else 0.0)
        if up and dn and reset_at is None:
            reset_at = max(t_up_edge, t_dn_edge) + cfg.t_reset
        if reset_at is not None and t >= reset_at:
            up = dn = False
            reset_at = None
        i = 0.0
        if up:
            i += np.interp(vc, cp_v, iup_t)
        if dn:
            i -= np.interp(vc, cp_v, idn_t)
        # loop filter: C2 at vctrl, series R-C1 to ground
        ir = (vc - v1) / r
        vc += (i - ir) * dt / c2
        v1 += ir * dt / c1
        if k % rec_stride == 0:
            t_rec.append(t)
            vc_rec.append(vc)
    return {
        "time": np.array(t_rec), "vctrl": np.array(vc_rec),
        "ref_edges": np.array(edges_ref), "div_edges": np.array(edges_div),
    }


def cycle_metrics(ref_edges: np.ndarray, div_edges: np.ndarray, n_div: int, fref: float):
    """Per-reference-cycle averaged VCO frequency and ref-div timing error.

    The divider edge nearest to each reference edge is paired with it. The
    averaged VCO frequency over a cycle is N / (div period).
    """
    if len(div_edges) < 3:
        return np.array([]), np.array([]), np.array([])
    fv = n_div / np.diff(div_edges)
    tf = div_edges[1:]
    ref_edges = ref_edges[(ref_edges > div_edges[0]) & (ref_edges < div_edges[-1])]
    idx = np.searchsorted(div_edges, ref_edges)
    idx = np.clip(idx, 1, len(div_edges) - 1)
    prev = div_edges[idx - 1]
    nxt = div_edges[idx]
    nearest = np.where(np.abs(ref_edges - prev) < np.abs(nxt - ref_edges), prev, nxt)
    perr = (nearest - ref_edges) * fref  # fraction of a reference period
    return tf, fv, np.column_stack([ref_edges, perr])


def lock_time_from_edges(ref_edges, div_edges, n_div, fref, ftol=1e-3, ptol=0.01) -> dict:
    """Lock = averaged frequency within ``ftol`` and |phase error| < ``ptol`` cycles thereafter."""
    from .metrics import lock_time
    tf, fv, pe = cycle_metrics(ref_edges, div_edges, n_div, fref)
    target = n_div * fref
    t_f = lock_time(tf, (fv - target) / target, ftol) if len(tf) else float("nan")
    t_p = lock_time(pe[:, 0], pe[:, 1], ptol) if len(pe) else float("nan")
    return {"freq_lock_s": t_f, "phase_lock_s": t_p, "tf": tf, "fv": fv, "perr": pe}
