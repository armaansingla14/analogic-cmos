"""Op-amp based bandgap reference with substrate PNPs.

::

    pg ----+-----------+-----------+------------+
           |           |           |            |
          Ma          Mb          Mc        Mtail (op-amp tail, 2x)
           |A          |B          |vref
          Q1 (1x)     R1          R2
                       |           |
                      Q2 (Nx)     Q3 (1x)

The op-amp forces V(A) = V(B), so the branch current is
I = V_T ln(N) / R1 (PTAT) and

    VREF = V_EB3 + (R2 / R1) V_T ln(N).

Two op-amp variants are provided:

* ``pmos``   - 5T PMOS-input op-amp sensing A and B directly. Its output is
               referenced to ground, so the gate line (which must track
               VDD - Vsg) needs a differential input change of dVDD / A_ol:
               poor line regulation / PSRR. Kept as the baseline.
* ``nmos_ls``- 5T NMOS-input op-amp sensing level-shifted nodes A' = A + I R3
               and B' = B + I R3 (equal R3 in both branches keeps A = B).
               The output is referenced to VDD through the PMOS mirror load,
               so the gate line tracks the supply naturally.

The op-amp tail is derived from the same gate line, making the core fully
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
    wm: float = 16e-6   # mirror PMOS
    lm: float = 4e-6
    win: float = 16e-6  # op-amp input pair
    lin: float = 2e-6
    wl: float = 2e-6    # op-amp NMOS load (pmos) / NMOS tail mirror (nmos_ls)
    ll: float = 2e-6
    cc: float = 2e-12   # compensation on the gate line
    opamp: str = "nmos_ls"
    r3: float = 50e3    # level-shift resistors (nmos_ls only)

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
    ]
    if s.opamp == "pmos":
        lines += [
            "* --- self-biased PMOS-input op-amp (+ = b, - = a, out = pg) ---",
            mm("Mt", "tail", "pg", "vdd", "vdd", "p", 2 * s.wm, s.lm),
            mm("Mi1", "d1", "b", "tail", "vdd", "p", s.win, s.lin),
            mm("Mi2", "pg", "a", "tail", "vdd", "p", s.win, s.lin),
            mm("Ml1", "d1", "d1", "0", "0", "n", s.wl, s.ll),
            mm("Ml2", "pg", "d1", "0", "0", "n", s.wl, s.ll),
        ]
    elif s.opamp == "nmos_ls":
        # Ma/Mb now drive a3/b3; R3 level-shifts down to a/b.
        lines[1] = mm("Ma", "a3", "pg", "vdd", "vdd", "p", s.wm, s.lm)
        lines[2] = mm("Mb", "b3", "pg", "vdd", "vdd", "p", s.wm, s.lm)
        lines += [
            p.res("R3a", "a3", "a", s.r3, w_um=2.0, l_um=s.r3 / 1000 * 2.0),
            p.res("R3b", "b3", "b", s.r3, w_um=2.0, l_um=s.r3 / 1000 * 2.0),
            "* --- self-biased NMOS-input op-amp (+ = b3, - = a3, out = pg) ---",
            mm("Mbp", "nb", "pg", "vdd", "vdd", "p", s.wm, s.lm),
            mm("Mbn", "nb", "nb", "0", "0", "n", s.wl, s.ll),
            mm("Mt", "tail", "nb", "0", "0", "n", 2 * s.wl, s.ll),
            mm("Mi1", "d1", "b3", "tail", "0", "n", s.win, s.lin),
            mm("Mi2", "pg", "a3", "tail", "0", "n", s.win, s.lin),
            mm("Ml1", "d1", "d1", "vdd", "vdd", "p", s.wm, s.lm),
            mm("Ml2", "pg", "d1", "vdd", "vdd", "p", s.wm, s.lm),
        ]
    else:
        raise ValueError(s.opamp)
    lines += [
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


NODESET = ".nodeset v(pg)=0.9 v(vref)=1.2\n"
STATE_NODES = ("pg", "a", "b", "vref", "e2", "e3", "su", "tail", "d1")
STATE_NODES_LS = ("a3", "b3", "nb")


def _state_nodes(core: str) -> tuple[str, ...]:
    return STATE_NODES + (STATE_NODES_LS if " a3 " in core else ())


def bench_startup(name: str, core: str, temp=27.0, vdd=1.8, t_ramp=2e-6, t_stop=20e-6) -> str:
    """Supply ramp from 0 V; also dumps the settled node voltages."""
    nodes = " ".join(f"v({n})" for n in _state_nodes(core))
    body = core + f".temp {temp}\nVDD vdd 0 pwl(0 0 {t_ramp} {vdd})\n"
    ctrl = f"""tran 5n {t_stop} 0 20n uic
wrdata {name}_start.dat v(vref) v(vdd)
wrdata {name}_state.dat {nodes}
"""
    return wrap(name, body, ctrl)


def _nodeset(state: dict | None) -> str:
    if not state:
        return NODESET
    return ".nodeset " + " ".join(f"v({k})={v:.6g}" for k, v in state.items()) + "\n"


def bench_temp(name: str, core: str, vdd=1.8, state: dict | None = None,
               t0=-40.0, t1=125.0, dt=5.0, t_mid=25.0) -> str:
    """Temperature sweep run outward from ``t_mid`` in two halves.

    The first DC point is seeded (``.nodeset``) with the node voltages of a
    start-up transient at ``t_mid`` and each later point continues from the
    previous solution. Without this, the DC solver can land on a
    non-physical high-current solution that a transient start-up never
    reaches.
    """
    body = core + f"VDD vdd 0 dc {vdd}\n" + _nodeset(state)
    ctrl = f"""dc temp {t_mid} {t1} {dt}
let iq = -i(VDD)
wrdata {name}_hot.dat v(vref) v(a) v(b) iq
dc temp {t_mid} {t0} {-dt}
let iq = -i(VDD)
wrdata {name}_cold.dat v(vref) v(a) v(b) iq
"""
    return wrap(name, body, ctrl)


def _merge_halves(hot: dict, cold: dict) -> dict:
    return {k: np.concatenate([cold[k][::-1], hot[k][1:]]) for k in hot}


def bench_line(name: str, core: str, temp=27.0, state: dict | None = None) -> str:
    body = core + f".temp {temp}\nVDD vdd 0 dc 1.8 ac 1\n" + _nodeset(state)
    ctrl = f"""op
ac dec 10 1 100Meg
let ps = v(vref)
wrdata {name}_psrr.dat ps
dc VDD 1.8 2.2 0.02
wrdata {name}_lineup.dat v(vref)
dc VDD 1.8 1.4 -0.02
wrdata {name}_linedn.dat v(vref)
"""
    return wrap(name, body, ctrl)


def settled_states(sim: NgSpice, cores: dict[str, tuple[str, float]], temp=25.0) -> dict:
    """Run start-up transients and return the final node voltages per tag."""
    jobs = {f"{tag}_su": bench_startup(f"{tag}_su", core, temp, vdd) for tag, (core, vdd) in cores.items()}
    res = sim.run(jobs)
    out = {}
    for tag in cores:
        tb = res[f"{tag}_su"].table(f"{tag}_su_state.dat")
        out[tag] = {k[2:-1]: float(v[-1]) for k, v in tb.items() if k.startswith("v(")}
    return out


def run_temp(sim: NgSpice, items: dict[str, tuple[Process, BandgapSizing, float]]) -> dict:
    """Batch temperature sweeps. items: tag -> (process, sizing, vdd).

    Each netlist is generated exactly once so the start-up transient and the
    sweep see the same mismatch draw.
    """
    cores = {tag: (bandgap_core(p, s), vdd) for tag, (p, s, vdd) in items.items()}
    states = settled_states(sim, cores)
    jobs = {tag: bench_temp(tag, core, vdd, states[tag]) for tag, (core, vdd) in cores.items()}
    res = sim.run(jobs)
    out = {}
    for tag in items:
        t = _merge_halves(res[tag].table(f"{tag}_hot.dat"), res[tag].table(f"{tag}_cold.dat"))
        temp, v = t["temp-sweep"], t["v(vref)"]
        i27 = int(np.argmin(np.abs(temp - 27.5)))
        out[tag] = {
            "temp": temp, "vref": v, "iq": t["iq"],
            "vref27": float(np.interp(27.0, temp, v)),
            "tc_ppm": tempco_ppm(temp, v),
            "iq27_ua": float(t["iq"][i27] * 1e6),
            "slope27_uv_per_c": float(np.gradient(v, temp)[i27] * 1e6),
            "startup_vref": states[tag].get("vref", float("nan")),
        }
    return out


def trim_r2(sim: NgSpice, s: BandgapSizing, vdd: float = 1.8, span: float = 0.04,
            points: int = 11, rounds: int = 2) -> BandgapSizing:
    """Pick R2 that minimizes the measured box-method tempco (TT, no mismatch).

    Mimics a production trim: a grid of R2 values around the hand-design
    value is simulated, the best one is kept, and the grid is refined.
    """
    centre = s.r2
    for _ in range(rounds):
        grid = centre * (1 + np.linspace(-span, span, points))
        items = {f"trim{i}": (Process.at("tt"), replace(s, r2=float(r)), vdd) for i, r in enumerate(grid)}
        res = run_temp(sim, items)
        tcs = [res[f"trim{i}"]["tc_ppm"] for i in range(points)]
        centre = float(grid[int(np.argmin(tcs))])
        span = span * 2.5 / (points - 1)
    return replace(s, r2=centre)
