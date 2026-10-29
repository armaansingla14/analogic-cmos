"""8-bit current-steering DAC (transistor-level cells, behavioural decoder).

Each cell is a cascoded NMOS current source (``m`` unit devices in parallel)
feeding a differential NMOS switch pair that steers the current to ``outp``
or ``outn``. Outputs are terminated by resistors to VDD, and the measured
quantity is the differential voltage ``v(outn) - v(outp)``, which increases
with code.

Two architectures are generated from the same unit cell:

* ``binary``      - eight binary-weighted cells (1,2,...,128 units)
* ``segmented``   - 4 MSBs thermometer coded (15 cells of 16 units) plus
                    4 binary LSB cells (1,2,4,8 units)

The digital decoder is ideal: Python computes each cell's switch state per
code and emits PWL gate drives. The code ramp is applied in a transient run
and every code is sampled at the end of its hold interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import dac_linearity, sfdr_db
from .models import Process
from .spice import NgSpice, wrap

NBITS = 8


@dataclass
class DacSizing:
    arch: str = "segmented"     # or "binary"
    n_therm_bits: int = 4
    iunit: float = 4e-6
    wcs: float = 1.5e-6         # unit current source
    lcs: float = 6e-6
    wcas: float = 1e-6          # unit cascode
    lcas: float = 0.5e-6
    wsw: float = 0.5e-6         # unit switch
    lsw: float = 0.18e-6
    rl: float = 400.0
    vcas: float = 1.2
    vdd: float = 1.8
    t_hold: float = 40e-9
    t_edge: float = 0.2e-9

    def as_dict(self):
        return asdict(self)


def cell_weights(s: DacSizing) -> list[int]:
    if s.arch == "binary":
        return [1 << i for i in range(NBITS)]
    if s.arch == "segmented":
        nl = NBITS - s.n_therm_bits
        return [1 << i for i in range(nl)] + [1 << nl] * ((1 << s.n_therm_bits) - 1)
    raise ValueError(s.arch)


def cell_states(s: DacSizing, code: int) -> list[int]:
    """Ideal decoder: which cells steer to the positive (increasing) side."""
    if not 0 <= code < (1 << NBITS):
        raise ValueError(code)
    if s.arch == "binary":
        return [(code >> i) & 1 for i in range(NBITS)]
    nl = NBITS - s.n_therm_bits
    lsb = [(code >> i) & 1 for i in range(nl)]
    msb_val = code >> nl
    therm = [1 if k < msb_val else 0 for k in range((1 << s.n_therm_bits) - 1)]
    return lsb + therm


def dac_netlist(p: Process, s: DacSizing, codes: list[int]) -> tuple[str, float]:
    """Return (netlist body, sample period). Codes are applied sequentially."""
    w = cell_weights(s)
    T, te = s.t_hold, s.t_edge
    lines = [
        f"VDD vdd 0 dc {s.vdd}",
        f"Vcas vcas 0 dc {s.vcas}",
        f"Iref vdd vbn dc {s.iunit:.6g}",
        p.mos("Mref", "vbn", "vbn", "0", "0", "n", s.wcs, s.lcs),
        f"RLp vdd outp {s.rl}",
        f"RLn vdd outn {s.rl}",
    ]
    states = np.array([cell_states(s, c) for c in codes])  # [n_codes, n_cells]
    for k, m in enumerate(w):
        lines += [
            p.mos(f"Mcs{k}", f"c{k}", "vbn", "0", "0", "n", s.wcs, s.lcs, m),
            p.mos(f"Mca{k}", f"d{k}", "vcas", f"c{k}", "0", "n", s.wcas, s.lcas, m),
            # state 1 -> current pulled from outn (v(outn) drops) ... see below
            p.mos(f"Msp{k}", "outn", f"g{k}", f"d{k}", "0", "n", s.wsw, s.lsw, m),
            p.mos(f"Msn{k}", "outp", f"gb{k}", f"d{k}", "0", "n", s.wsw, s.lsw, m),
        ]
        lines.append(f"Vg{k} g{k} 0 pwl({_pwl(states[:, k], T, te, s.vdd)})")
        lines.append(f"Vgb{k} gb{k} 0 pwl({_pwl(1 - states[:, k], T, te, s.vdd)})")
    return "\n".join(lines) + "\n", T


def _pwl(bits: np.ndarray, T: float, te: float, vdd: float) -> str:
    pts = [f"0 {bits[0] * vdd:g}"]
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            t0 = i * T
            pts.append(f"{t0:.6g} {bits[i - 1] * vdd:g}")
            pts.append(f"{t0 + te:.6g} {bits[i] * vdd:g}")
    return " ".join(pts)


def bench(name: str, p: Process, s: DacSizing, codes: list[int], temp=27.0) -> str:
    body, T = dac_netlist(p, s, codes)
    t_stop = len(codes) * T
    ctrl = f"""tran 0.5n {t_stop:.6g} 0 0.2n
let vd = v(outp) - v(outn)
linearize vd
wrdata {name}_tran.dat vd
"""
    # outp collects the current of cells in state 0 and outn of state 1, so
    # vd = v(outp) - v(outn) = RL*(I_on - I_off) increases with code.
    return wrap(name, f".temp {temp}\n" + body, ctrl)


def sample_codes(tr: dict, n: int, T: float, frac: float = 0.95) -> np.ndarray:
    t, v = tr["time"], tr["vd"]
    ts = (np.arange(n) + frac) * T
    return np.interp(ts, t, v)


def ramp_codes() -> list[int]:
    return list(range(1 << NBITS))


def sine_codes(n: int = 256, cycles: int = 7, amp: float = 0.98) -> list[int]:
    k = np.arange(n)
    x = 127.5 + 127.5 * amp * np.sin(2 * np.pi * cycles * k / n)
    return [int(v) for v in np.clip(np.round(x), 0, 255)]


def run_linearity(sim: NgSpice, items: dict[str, tuple[Process, DacSizing]]) -> dict:
    codes = ramp_codes()
    jobs = {tag: bench(tag, p, s, codes) for tag, (p, s) in items.items()}
    res = sim.run(jobs)
    out = {}
    for tag, (_, s) in items.items():
        tr = res[tag].table(f"{tag}_tran.dat")
        lv = sample_codes(tr, len(codes), s.t_hold)
        ep = dac_linearity(lv, "endpoint")
        bf = dac_linearity(lv, "bestfit")
        out[tag] = {
            "levels": lv, "endpoint": ep, "bestfit": bf,
            "glitch": midscale_glitch(tr, s),
        }
    return out


def midscale_glitch(tr: dict, s: DacSizing) -> float:
    """Glitch area (V*s) of the 127->128 transition relative to the settled step."""
    T = s.t_hold
    t, v = tr["time"], tr["vd"]
    t0 = 128 * T
    mask = (t >= t0) & (t < t0 + 0.8 * T)
    v_final = np.interp(t0 + 0.95 * T, t, v)
    v_init = np.interp(t0 - 0.05 * T, t, v)
    seg_t, seg_v = t[mask], v[mask]
    # Error relative to an ideal instantaneous step to the final value.
    err = seg_v - v_final
    del v_init
    return float(np.trapezoid(np.abs(err), seg_t))


def run_sfdr(sim: NgSpice, tag: str, p: Process, s: DacSizing, n=256, cycles=7) -> float:
    codes = sine_codes(n, cycles)
    res = sim.run({tag: bench(tag, p, s, codes)})[tag]
    lv = sample_codes(res.table(f"{tag}_tran.dat"), n, s.t_hold)
    return sfdr_db(lv, cycles)
