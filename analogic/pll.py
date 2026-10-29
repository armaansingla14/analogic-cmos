"""Charge-pump PLL building blocks in ngspice.

* ``vco``  - 5-stage current-starved ring oscillator (transistor level).
             Frequency vs control voltage is extracted from transients.
* ``cp``   - drain-switched charge pump (transistor level). UP/DN currents
             vs output voltage are extracted from DC sweeps.
* ``loop`` - mixed-level closed loop: XSPICE digital PFD (two DFFs + AND
             reset with delay) and divider, transistor-level charge pump,
             passive 2nd-order loop filter, and a behavioural VCO
             (XSPICE ``d_osc``) whose tuning table is the transistor VCO
             curve measured by :func:`vco_curve`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .models import Process
from .spice import NgSpice, wrap


# --------------------------------------------------------------------------
# VCO
# --------------------------------------------------------------------------

@dataclass
class VcoSizing:
    stages: int = 5
    wn: float = 1.0e-6
    wp: float = 2.5e-6
    l: float = 0.18e-6
    wns: float = 1e-6     # starving NMOS
    wps: float = 2.5e-6   # starving PMOS
    ls: float = 0.18e-6
    cload: float = 15e-15

    def as_dict(self):
        return asdict(self)


def vco_netlist(p: Process, s: VcoSizing) -> str:
    lines = [
        "* control-voltage to PMOS bias replica",
        p.mos("Mbn", "vbp", "vctrl", "0", "0", "n", s.wns, s.ls),
        p.mos("Mbp", "vbp", "vbp", "vdd", "vdd", "p", s.wps, s.ls),
    ]
    n = s.stages
    for k in range(n):
        a, y = f"n{k}", f"n{(k + 1) % n}"
        lines += [
            p.mos(f"Mps{k}", f"sp{k}", "vbp", "vdd", "vdd", "p", s.wps, s.ls),
            p.mos(f"Mp{k}", y, a, f"sp{k}", "vdd", "p", s.wp, s.l),
            p.mos(f"Mn{k}", y, a, f"sn{k}", "0", "n", s.wn, s.l),
            p.mos(f"Mns{k}", f"sn{k}", "vctrl", "0", "0", "n", s.wns, s.ls),
            f"Cl{k} {y} 0 {s.cload}",
        ]
    return "\n".join(lines) + "\n"


def bench_vco(name: str, p: Process, s: VcoSizing, vctrl: float, vdd=1.8, temp=27.0,
              t_stop=60e-9) -> str:
    body = vco_netlist(p, s) + f".temp {temp}\nVDD vdd 0 dc {vdd}\nVc vctrl 0 dc {vctrl}\n.ic v(n0)=0 v(n1)={vdd}\n"
    ctrl = f"""tran 5p {t_stop} 0 5p uic
let iavg = -i(VDD)
linearize v(n0) iavg
wrdata {name}_osc.dat v(n0) iavg
"""
    return wrap(name, body, ctrl)


def osc_freq(t: np.ndarray, v: np.ndarray, vth: float, skip: float = 0.3) -> float:
    """Average frequency from rising threshold crossings after ``skip`` fraction."""
    m = t >= t[-1] * skip
    t, v = t[m], v[m]
    s = v - vth
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(idx) < 3:
        return 0.0
    tc = t[idx] + (t[idx + 1] - t[idx]) * (-s[idx]) / (s[idx + 1] - s[idx])
    return float((len(tc) - 1) / (tc[-1] - tc[0]))


def vco_curve(sim: NgSpice, p: Process, s: VcoSizing, vctrls, vdd=1.8, temp=27.0,
              tag="vco") -> dict:
    jobs = {f"{tag}_{i}": bench_vco(f"{tag}_{i}", p, s, v, vdd, temp) for i, v in enumerate(vctrls)}
    res = sim.run(jobs)
    f, pw = [], []
    for i, _ in enumerate(vctrls):
        tb = res[f"{tag}_{i}"].table(f"{tag}_{i}_osc.dat")
        f.append(osc_freq(tb["time"], tb["v(n0)"], vdd / 2))
        tm = tb["time"] >= tb["time"][-1] * 0.3
        pw.append(float(np.mean(tb["iavg"][tm]) * vdd))
    return {"vctrl": np.asarray(vctrls, float), "freq": np.asarray(f), "power": np.asarray(pw)}


# --------------------------------------------------------------------------
# Charge pump
# --------------------------------------------------------------------------

@dataclass
class CpSizing:
    icp: float = 50e-6
    wn: float = 4e-6
    ln: float = 1e-6
    wp: float = 10e-6
    lp: float = 1e-6
    wsn: float = 1e-6
    wsp: float = 2.5e-6
    lsw: float = 0.18e-6

    def as_dict(self):
        return asdict(self)


def cp_netlist(p: Process, s: CpSizing, up: str = "up", upb: str = "upb",
               dn: str = "dn", out: str = "vctrl") -> str:
    """Charge pump; ``upb`` drives the PMOS switch (active low)."""
    lines = [
        f"Icp vdd nbias dc {s.icp:.6g}",
        p.mos("Mb1", "nbias", "nbias", "0", "0", "n", s.wn, s.ln),
        p.mos("Mb2", "pbias", "nbias", "0", "0", "n", s.wn, s.ln),
        p.mos("Mb3", "pbias", "pbias", "vdd", "vdd", "p", s.wp, s.lp),
        p.mos("Mup", "sp", "pbias", "vdd", "vdd", "p", s.wp, s.lp),
        p.mos("Mswp", out, upb, "sp", "vdd", "p", s.wsp, s.lsw),
        p.mos("Mswn", out, dn, "sn", "0", "n", s.wsn, s.lsw),
        p.mos("Mdn", "sn", "nbias", "0", "0", "n", s.wn, s.ln),
    ]
    _ = up
    return "\n".join(lines) + "\n"


def bench_cp(name: str, p: Process, s: CpSizing, vdd=1.8, temp=27.0) -> str:
    """Sweep the output voltage with UP only, then DN only."""
    body = cp_netlist(p, s) + f""".temp {temp}
VDD vdd 0 dc {vdd}
Vupb upb 0 dc 0
Vdn dn 0 dc 0
Vout vctrl 0 dc 0.9
"""
    ctrl = f"""dc Vout 0 {vdd} 0.02
let iup = i(Vout)
wrdata {name}_up.dat iup
alter Vupb dc = {vdd}
alter Vdn dc = {vdd}
dc Vout 0 {vdd} 0.02
let idn = -i(Vout)
wrdata {name}_dn.dat idn
"""
    return wrap(name, body, ctrl)


def cp_curves(sim: NgSpice, items: dict[str, tuple[Process, CpSizing, float, float]]) -> dict:
    jobs = {tag: bench_cp(tag, p, s, vdd, t) for tag, (p, s, vdd, t) in items.items()}
    res = sim.run(jobs)
    out = {}
    for tag in items:
        up = res[tag].table(f"{tag}_up.dat")
        dn = res[tag].table(f"{tag}_dn.dat")
        out[tag] = {"vout": up["v-sweep"], "iup": up["iup"], "idn": dn["idn"]}
    return out


def cp_mismatch_window(curve: dict, tol: float = 0.05) -> tuple[float, float]:
    """Output-voltage range where |Iup-Idn|/mean < tol and both are within tol of nominal."""
    v, iu, idn = curve["vout"], curve["iup"], curve["idn"]
    mid = np.argmin(np.abs(v - v.mean()))
    nom = 0.5 * (iu[mid] + idn[mid])
    ok = (np.abs(iu - idn) / nom < tol) & (np.abs(iu - nom) / nom < tol) & (np.abs(idn - nom) / nom < tol)
    if not ok.any():
        return float("nan"), float("nan")
    # contiguous range around the midpoint
    lo = hi = mid
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(v) - 1 and ok[hi + 1]:
        hi += 1
    return float(v[lo]), float(v[hi])


# --------------------------------------------------------------------------
# Closed loop (mixed-level)
# --------------------------------------------------------------------------

@dataclass
class LoopParams:
    fref: float = 25e6
    n_div: int = 16
    r: float = 10e3
    c1: float = 100e-12
    c2: float = 10e-12
    t_reset: float = 150e-12
    v_init: float = 0.0
    t_stop: float = 6e-6

    def as_dict(self):
        return asdict(self)


def loop_netlist(p: Process, cp: CpSizing, lp: LoopParams, vco_table: dict, vdd=1.8) -> str:
    vc = np.asarray(vco_table["vctrl"])
    fr = np.maximum(np.asarray(vco_table["freq"]), 1e6)
    # d_osc requires a monotonically increasing control array
    order = np.argsort(vc)
    vc, fr = vc[order], fr[order]
    carr = " ".join(f"{x:.4g}" for x in vc)
    farr = " ".join(f"{x:.6g}" for x in fr)
    half = 0.5 / lp.fref
    body = f"""VDD vdd 0 dc {vdd}
.model a2d adc_bridge(in_low={vdd/2} in_high={vdd/2})
.model d2a dac_bridge(out_low=0 out_high={vdd} t_rise=50p t_fall=50p)
.model dff d_dff(clk_delay=50p set_delay=50p reset_delay=50p)
.model and2 d_and(rise_delay={lp.t_reset:.4g} fall_delay=50p)
.model inv d_inverter(rise_delay=20p fall_delay=20p)
.model div d_fdiv(div_factor={lp.n_div} high_cycles={lp.n_div // 2})
.model vcom d_osc(cntl_array=[{carr}] freq_array=[{farr}] duty_cycle=0.5)
* reference clock
Vref refa 0 pulse(0 {vdd} 1n 50p 50p {half - 50e-12:.6g} {2 * half:.6g})
aref [refa] [refd] a2d
* behavioural VCO driven by the loop-filter voltage
avco vctrl vcod vcom
adiv vcod divd div
* PFD
Vone onea 0 dc {vdd}
aone [onea] [oned] a2d
aff1 oned refd nullset rst upd upbd dff
aff2 oned divd nullset rst dnd dnbd dff
aand [upd dnd] rst and2
* tie-off for unused set pins
Vzero zeroa 0 dc 0
azero [zeroa] [nullset] a2d
adac [upd upbd dnd] [up upb dn] d2a
* transistor-level charge pump
{cp_netlist(p, cp)}
* 2nd-order passive loop filter
Rlf vctrl v1 {lp.r:.6g}
C1 v1 0 {lp.c1:.6g}
C2 vctrl 0 {lp.c2:.6g}
.ic v(vctrl)={lp.v_init} v(v1)={lp.v_init}
"""
    return body


def bench_loop(name: str, p: Process, cp: CpSizing, lp: LoopParams, vco_table: dict,
               vdd=1.8, temp=27.0) -> str:
    body = f".temp {temp}\n" + loop_netlist(p, cp, lp, vco_table, vdd)
    ctrl = f"""tran 0.1n {lp.t_stop:.6g} 0 0.1n uic
linearize v(vctrl) v(up) v(dn)
wrdata {name}_loop.dat v(vctrl) v(up) v(dn)
"""
    return wrap(name, body, ctrl)


def run_loop(sim: NgSpice, tag: str, p: Process, cp: CpSizing, lp: LoopParams,
             vco_table: dict, vdd=1.8) -> dict:
    res = sim.run({tag: bench_loop(tag, p, cp, lp, vco_table, vdd)})[tag]
    tb = res.table(f"{tag}_loop.dat")
    vc = tb["v(vctrl)"]
    f = np.interp(vc, vco_table["vctrl"], vco_table["freq"])
    return {"time": tb["time"], "vctrl": vc, "fvco": f, "up": tb["v(up)"], "dn": tb["v(dn)"]}


def loop_lock(res: dict, fref: float, f_target: float, vdd: float = 1.8,
              ftol: float = 1e-3, ptol: float = 0.01) -> dict:
    """Lock time from a closed-loop transient.

    Per reference cycle: averaged VCO frequency (from the tuning table) and
    net PFD pulse width (UP minus DN, as a fraction of the reference period).
    Lock = both within tolerance for the rest of the record.
    """
    from .metrics import lock_time
    t = res["time"]
    tref = 1.0 / fref
    ncyc = int(t[-1] / tref)
    tc, fav, pe = [], [], []
    dt = np.gradient(t)
    for k in range(1, ncyc):
        m = (t >= k * tref) & (t < (k + 1) * tref)
        if m.sum() < 3:
            continue
        tc.append((k + 1) * tref)
        fav.append(np.sum(res["fvco"][m] * dt[m]) / np.sum(dt[m]))
        pe.append(np.sum((res["up"][m] - res["dn"][m]) / vdd * dt[m]) / tref)
    tc, fav, pe = map(np.asarray, (tc, fav, pe))
    return {
        "freq_lock_s": lock_time(tc, (fav - f_target) / f_target, ftol),
        "phase_lock_s": lock_time(tc, pe, ptol),
        "tc": tc, "fav": fav, "perr": pe,
    }
