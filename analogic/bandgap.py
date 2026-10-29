"""Op-amp based bandgap reference with substrate PNPs.

::

    pg ----+-----------+-----------+------------+
           |           |           |            |
          Ma          Mb          Mc        Mtail (op-amp tail, 2x)
           |A          |B          |vref
          Q1 (1x)     R1          R2
                       |           |
                      Q2 (Nx)     Q3 (1x)

The PMOS-input op-amp forces V(A) = V(B), so the branch current is
I = V_T ln(N) / R1 (PTAT) and

    VREF = V_EB3 + (R2 / R1) V_T ln(N).

The op-amp tail is mirrored from the same gate line, making the core fully
self-biased; a three-transistor start-up circuit removes the zero-current
state. ``R2`` is chosen either from the first-order equation or by a
two-point slope trim (see :func:`trim_r2`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .metrics import tempco_ppm
from .models import Process
from .spice import NgSpice, wrap

VT27 = 1.380649e-23 * 300.15 / 1.602176634e-19


@dataclass
class BandgapSizing:
    n_ratio: int = 8
    r1: float = 10.8e3
    r2: float = 110e3
    wm: float = 4e-6    # mirror PMOS
    lm: float = 2e-6
    win: float = 8e-6   # op-amp input pair (PMOS)
    lin: float = 1e-6
    wl: float = 2e-6    # op-amp NMOS load
    ll: float = 2e-6
    cc: float = 2e-12   # compensation on the gate line

    def as_dict(self):
        return asdict(self)


def first_order_r2(s: BandgapSizing, vbe_guess: float = 0.66, vref_target: float = 1.2) -> float:
    """R2 from VREF = VBE + (R2/R1) VT ln N for a target output (hand design)."""
    return s.r1 * (vref_target - vbe_guess) / (VT27 * np.log(s.n_ratio))


def bandgap_core(p: Process, s: BandgapSizing) -> str:
    mm = p.mos
    lines = [
        "* --- PTAT/CTAT core ---",
        mm("Ma", "a", "pg", "vdd", "vdd", "p", s.wm, s.lm),
        mm("Mb", "b", "pg", "vdd", "vdd", "p", s.wm, s.lm),
        mm("Mc", "vref", "pg", "vdd", "vdd", "p", s.wm, s.lm),
        _pnp(p, "Q1", "a", 1),
        p.res("R1", "b", "e2", s.r1, w_um=2.0, l_um=s.r1 / 1000 * 2.0),
        _pnp(p, "Q2", "e2", s.n_ratio),
        p.res("R2", "vref", "e3", s.r2, w_um=2.0, l_um=s.r2 / 1000 * 2.0),
        _pnp(p, "Q3", "e3", 1),
        "* --- self-biased PMOS-input op-amp (+ = b, - = a, out = pg) ---",
        mm("Mt", "tail", "pg", "vdd", "vdd", "p", 2 * s.wm, s.lm),
        mm("Mi1", "d1", "b", "tail", "vdd", "p", s.win, s.lin),
        mm("Mi2", "pg", "a", "tail", "vdd", "p", s.win, s.lin),
        mm("Ml1", "d1", "d1", "0", "0", "n", s.wl, s.ll),
        mm("Ml2", "pg", "d1", "0", "0", "n", s.wl, s.ll),
        f"Cc pg vdd {s.cc}",
        "* --- start-up ---",
        mm("Msw", "su", "0", "vdd", "vdd", "p", 0.5e-6, 20e-6),
        mm("Msd", "su", "vref", "0", "0", "n", 2e-6, 0.5e-6),
        mm("Mst", "pg", "su", "0", "0", "n", 0.5e-6, 2e-6),
    ]
    return "\n".join(lines) + "\n"


def _pnp(p: Process, name: str, emitter: str, area: float) -> str:
    """Diode-connected substrate PNP (collector and base grounded).

    Under mismatch the emitter area is perturbed, which scales IS; the
    assumed sigma(dIs/Is) is 2 % / sqrt(units) (approximate).
    """
    a = float(area)
    if p.rng is not None:
        a *= 1 + p.rng.normal(0, 0.02 / np.sqrt(area)) * p.mismatch_scale
    return f"{name} 0 0 {emitter} vpnp area={a:.6g}"


def bench_temp(name: str, p: Process, s: BandgapSizing, vdd=1.8,
               t0=-40.0, t1=125.0, dt=5.0) -> str:
    body = bandgap_core(p, s) + f"VDD vdd 0 dc {vdd}\n.nodeset v(pg)=1.0\n"
    ctrl = f"""dc temp {t0} {t1} {dt}
let iq = -i(VDD)
wrdata {name}_temp.dat v(vref) v(a) v(b) iq
"""
    return wrap(name, body, ctrl)


def bench_line(name: str, p: Process, s: BandgapSizing, temp=27.0) -> str:
    body = bandgap_core(p, s) + f".temp {temp}\nVDD vdd 0 dc 1.8 ac 1\n.nodeset v(pg)=1.0\n"
    ctrl = f"""dc VDD 1.4 2.2 0.02
wrdata {name}_line.dat v(vref)
op
ac dec 10 1 100Meg
let ps = v(vref)
wrdata {name}_psrr.dat ps
"""
    return wrap(name, body, ctrl)


def bench_startup(name: str, p: Process, s: BandgapSizing, temp=27.0, vdd=1.8) -> str:
    body = bandgap_core(p, s) + f".temp {temp}\nVDD vdd 0 pwl(0 0 2u {vdd})\n"
    ctrl = f"""tran 5n 20u 0 20n uic
wrdata {name}_start.dat v(vref) v(vdd)
"""
    return wrap(name, body, ctrl)


def run_temp(sim: NgSpice, items: dict[str, tuple[Process, BandgapSizing, float]]) -> dict:
    """Batch temperature sweeps. items: tag -> (process, sizing, vdd)."""
    jobs = {tag: bench_temp(tag, p, s, vdd) for tag, (p, s, vdd) in items.items()}
    res = sim.run(jobs)
    out = {}
    for tag in items:
        t = res[tag].table(f"{tag}_temp.dat")
        temp, v = t["temp-sweep"], t["v(vref)"]
        i27 = int(np.argmin(np.abs(temp - 27.5)))
        out[tag] = {
            "temp": temp, "vref": v, "iq": t["iq"],
            "vref27": float(np.interp(27.0, temp, v)),
            "tc_ppm": tempco_ppm(temp, v),
            "iq27_ua": float(t["iq"][i27] * 1e6),
            "slope27_uv_per_c": float(np.gradient(v, temp)[i27] * 1e6),
        }
    return out


def trim_r2(sim: NgSpice, s: BandgapSizing, vdd: float = 1.8, iters: int = 2) -> BandgapSizing:
    """Secant trim of R2 so that dVREF/dT = 0 at the centre of the range.

    This mimics a production trim step: the slope at ~40 degC is measured
    at two R2 values and R2 is linearly interpolated (VREF slope is affine
    in R2 to first order). Done at TT, no mismatch.
    """
    centre = 42.5
    def slope(r2):
        res = run_temp(sim, {"trim": (Process.at("tt"), replace(s, r2=r2), vdd)})["trim"]
        g = np.gradient(res["vref"], res["temp"])
        return float(np.interp(centre, res["temp"], g))
    ra, rb = s.r2, s.r2 * 1.05
    sa, sb = slope(ra), slope(rb)
    for _ in range(iters):
        rc = rb - sb * (rb - ra) / (sb - sa)
        ra, sa = rb, sb
        rb, sb = rc, slope(rc)
    return replace(s, r2=float(rb))
