"""Two-stage Miller-compensated OTA with NMOS input pair and PMOS current-mirror load.

Topology::

    M8  : diode-connected NMOS bias reference (fed by Ib)
    M5  : tail current source (mirror of M8)
    M1,2: NMOS differential pair (M1 gate = inn, M2 gate = inp)
    M3,4: PMOS current-mirror active load
    M6  : PMOS common-source second stage
    M7  : NMOS current-sink load of the second stage (mirror of M8)
    Rz,Cc: Miller compensation with nulling resistor

The benches below generate complete ngspice decks for AC, noise, transient,
DC offset and supply/common-mode rejection measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import ac_metrics, settling_time, slew_rate
from .models import Process
from .spice import NgSpice, wrap


@dataclass
class OtaSizing:
    w1: float = 8e-6
    l1: float = 0.5e-6
    w3: float = 4e-6
    l3: float = 0.5e-6
    w5: float = 8e-6
    l5: float = 1e-6
    w6: float = 32e-6
    l6: float = 0.5e-6
    w7: float = 32e-6
    l7: float = 1e-6
    w8: float = 4e-6
    l8: float = 1e-6
    ib: float = 20e-6
    cc: float = 1.0e-12
    rz: float = 1.5e3
    cl: float = 2e-12

    def as_dict(self):
        return asdict(self)


def ota_subckt(p: Process, s: OtaSizing, name: str = "ota") -> str:
    lines = [
        f".subckt {name} inp inn out vdd vss",
        f"Ib vdd nb dc {s.ib:.6g}",
        p.mos("M8", "nb", "nb", "vss", "vss", "n", s.w8, s.l8),
        p.mos("M5", "tail", "nb", "vss", "vss", "n", s.w5, s.l5),
        p.mos("M1", "x", "inn", "tail", "vss", "n", s.w1, s.l1),
        p.mos("M2", "y", "inp", "tail", "vss", "n", s.w1, s.l1),
        p.mos("M3", "x", "x", "vdd", "vdd", "p", s.w3, s.l3),
        p.mos("M4", "y", "x", "vdd", "vdd", "p", s.w3, s.l3),
        p.mos("M6", "out", "y", "vdd", "vdd", "p", s.w6, s.l6),
        p.mos("M7", "out", "nb", "vss", "vss", "n", s.w7, s.l7),
        f"Rz y z {s.rz:.6g}",
        f"Cc z out {s.cc:.6g}",
        f".ends {name}",
    ]
    return "\n".join(lines) + "\n"


def _supply(vdd: float, temp: float, vdd_ac: bool = False) -> str:
    ac = " ac 1" if vdd_ac else ""
    return f".temp {temp}\nVDD vdd 0 dc {vdd}{ac}\nVSS vss 0 dc 0\n"


def bench_ac(name: str, p: Process, s: OtaSizing, vdd=1.8, temp=27.0, vcm=0.9) -> str:
    """Open-loop AC with a large-L DC feedback path that sets the operating point."""
    body = ota_subckt(p, s) + _supply(vdd, temp) + f"""
Vin inp 0 dc {vcm} ac 1
Lfb out inn 1e9
Cfb inn 0 1
CL out 0 {s.cl}
X1 inp inn out vdd vss ota
"""
    ctrl = f"""op
let vout_dc = v(out)
print vout_dc
ac dec 40 1 10G
let h = v(out)
wrdata {name}_ac.dat h
"""
    return wrap(name, body, ctrl)


def bench_psrr(name: str, p: Process, s: OtaSizing, vdd=1.8, temp=27.0, vcm=0.9) -> str:
    """Supply-to-output gain in the same open-loop configuration."""
    body = ota_subckt(p, s) + _supply(vdd, temp, vdd_ac=True) + f"""
Vin inp 0 dc {vcm}
Lfb out inn 1e9
Cfb inn 0 1
CL out 0 {s.cl}
X1 inp inn out vdd vss ota
"""
    ctrl = f"""ac dec 20 1 1G
let hs = v(out)
wrdata {name}_ps.dat hs
"""
    return wrap(name, body, ctrl)


def bench_cm(name: str, p: Process, s: OtaSizing, vdd=1.8, temp=27.0, vcm=0.9) -> str:
    """Common-mode gain: both inputs driven, DC loop closed through a large L."""
    body = ota_subckt(p, s) + _supply(vdd, temp) + f"""
Vcm cm 0 dc {vcm} ac 1
* inp sees the CM source; inn sees it through a 1F cap while DC is set by Lfb
Eip inp 0 cm 0 1
Lfb out inn 1e9
Cfb inn cm 1
CL out 0 {s.cl}
X1 inp inn out vdd vss ota
"""
    ctrl = f"""ac dec 20 1 1G
let hc = v(out)
wrdata {name}_cm.dat hc
"""
    return wrap(name, body, ctrl)


def bench_noise(name: str, p: Process, s: OtaSizing, vdd=1.8, temp=27.0, vcm=0.9) -> str:
    body = ota_subckt(p, s) + _supply(vdd, temp) + f"""
Vin inp 0 dc {vcm} ac 1
Lfb out inn 1e9
Cfb inn 0 1
CL out 0 {s.cl}
X1 inp inn out vdd vss ota
"""
    ctrl = f"""noise v(out) Vin dec 20 1 100Meg 1
setplot noise1
let vn_in = inoise_spectrum
wrdata {name}_noise.dat vn_in
setplot noise2
print inoise_total
"""
    return wrap(name, body, ctrl)


def bench_tran(name: str, p: Process, s: OtaSizing, vdd=1.8, temp=27.0,
               v_lo=0.6, v_hi=1.2, t_step=50e-9, t_stop=1.0e-6) -> str:
    """Unity-gain buffer large-signal step response."""
    body = ota_subckt(p, s) + _supply(vdd, temp) + f"""
Vin inp 0 pwl(0 {v_lo} {t_step} {v_lo} {t_step + 0.1e-9} {v_hi})
CL out 0 {s.cl}
X1 inp out out vdd vss ota
"""
    ctrl = f"""tran 0.1n {t_stop} 0 0.2n
wrdata {name}_tran.dat v(out) v(inp)
"""
    return wrap(name, body, ctrl)


def bench_offset(name: str, p: Process, s: OtaSizing, vdd=1.8, temp=27.0, vcm=0.9) -> str:
    """Unity-gain buffer DC operating point: offset = v(out) - v(inp)."""
    body = ota_subckt(p, s) + _supply(vdd, temp) + f"""
Vin inp 0 dc {vcm}
CL out 0 {s.cl}
X1 inp out out vdd vss ota
"""
    ctrl = """op
let vos = v(out) - v(inp)
let idd = -i(VDD)
print vos idd
"""
    return wrap(name, body, ctrl)


# ---------------------------------------------------------------------------
# High-level characterization helpers
# ---------------------------------------------------------------------------

def characterize(sim: NgSpice, p: Process, s: OtaSizing, vdd=1.8, temp=27.0,
                 tag: str = "ota", full: bool = True) -> dict:
    """Run AC (+ optional noise/tran/PSRR/CMRR) and return a metric dict."""
    jobs = {
        f"{tag}_a": bench_ac(f"{tag}_a", p, s, vdd, temp),
        f"{tag}_o": bench_offset(f"{tag}_o", p, s, vdd, temp),
    }
    if full:
        jobs[f"{tag}_n"] = bench_noise(f"{tag}_n", p, s, vdd, temp)
        jobs[f"{tag}_t"] = bench_tran(f"{tag}_t", p, s, vdd, temp)
        jobs[f"{tag}_p"] = bench_psrr(f"{tag}_p", p, s, vdd, temp)
        jobs[f"{tag}_c"] = bench_cm(f"{tag}_c", p, s, vdd, temp)
    res = sim.run(jobs)
    return extract(res, tag, full)


def extract(res, tag: str, full: bool = True) -> dict:
    ac = res[f"{tag}_a"].table(f"{tag}_a_ac.dat")
    m = ac_metrics(ac["frequency"], ac["h"]).as_dict()
    out = {
        "dc_gain_db": m["dc_gain_db"],
        "f3db_hz": m["f3db_hz"],
        "ugb_hz": m["ugb_hz"],
        "pm_deg": m["phase_margin_deg"],
        "gm_db": m["gain_margin_db"],
        "offset_mv": res[f"{tag}_o"].scalars["vos"] * 1e3,
        "idd_ua": res[f"{tag}_o"].scalars["idd"] * 1e6,
    }
    if full:
        nz = res[f"{tag}_n"].table(f"{tag}_n_noise.dat")
        f, vn = nz["frequency"], np.real(nz["vn_in"])
        out["vn_1k_nv"] = float(np.interp(np.log10(1e3), np.log10(f), vn) * 1e9)
        out["vn_1m_nv"] = float(np.interp(np.log10(1e6), np.log10(f), vn) * 1e9)
        out["vn_int_1_100m_uv"] = res[f"{tag}_n"].scalars["inoise_total"] * 1e6
        tr = res[f"{tag}_t"].table(f"{tag}_t_tran.dat")
        t, v = tr["time"], tr["v(out)"]
        mask = t >= 50e-9
        out["slew_v_per_us"] = slew_rate(t[mask], v[mask]) / 1e6
        out["settle_1pct_ns"] = settling_time(t, v, 50e-9, 0.01) * 1e9
        ps = res[f"{tag}_p"].table(f"{tag}_p_ps.dat")
        cm = res[f"{tag}_c"].table(f"{tag}_c_cm.dat")
        a_dm = np.interp(np.log10(ps["frequency"]), np.log10(ac["frequency"]), np.abs(ac["h"]))
        out["psrr_db_dc"] = float(20 * np.log10(a_dm[0] / abs(ps["hs"][0])))
        out["cmrr_db_dc"] = float(20 * np.log10(a_dm[0] / abs(cm["hc"][0])))
    return out
