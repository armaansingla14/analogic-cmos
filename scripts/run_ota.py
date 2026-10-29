"""Two-stage OTA: nominal characterization, compensation sweep, PVT corners and Monte Carlo.

Writes results/ota_*.csv, results/ota_summary.md and plots.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analogic import ota  # noqa: E402
from analogic.metrics import ac_metrics  # noqa: E402
from analogic.models import Process, pvt_grid  # noqa: E402
from analogic.report import (SERIES, md_table, new_fig, save, stats, style_axes,  # noqa: E402
                             write_csv, write_md)
from analogic.spice import NgSpice  # noqa: E402


def nominal(sim, s):
    p = Process.at("tt")
    jobs = {
        "nom_a": ota.bench_ac("nom_a", p, s), "nom_o": ota.bench_offset("nom_o", p, s),
        "nom_n": ota.bench_noise("nom_n", p, s), "nom_t": ota.bench_tran("nom_t", p, s),
        "nom_p": ota.bench_psrr("nom_p", p, s), "nom_c": ota.bench_cm("nom_c", p, s),
    }
    res = sim.run(jobs)
    m = ota.extract(res, "nom", full=True)
    ac = res["nom_a"].table("nom_a_ac.dat")
    nz = res["nom_n"].table("nom_n_noise.dat")
    tr = res["nom_t"].table("nom_t_tran.dat")

    fig, (a1, a2) = new_fig(2, 1, 5, 4.6)
    f, h = ac["frequency"], ac["h"]
    a1.semilogx(f, 20 * np.log10(np.abs(h)), color=SERIES[0], lw=1.6)
    style_axes(a1, "", "gain (dB)", "OTA open-loop response (TT, 27 C, CL = 2 pF)")
    ph = np.degrees(np.unwrap(np.angle(h)))
    a2.semilogx(f, ph - ph[0], color=SERIES[1], lw=1.6)
    style_axes(a2, "frequency (Hz)", "phase (deg)")
    save(fig, "ota_bode.png")

    fig, ax = new_fig()
    ax.loglog(nz["frequency"], np.real(nz["vn_in"]) * 1e9, color=SERIES[0], lw=1.6)
    style_axes(ax, "frequency (Hz)", "nV/sqrt(Hz)", "OTA input-referred noise")
    save(fig, "ota_noise.png")

    fig, ax = new_fig()
    ax.plot(tr["time"] * 1e9, tr["v(inp)"], color=SERIES[1], lw=1.2, label="input")
    ax.plot(tr["time"] * 1e9, tr["v(out)"], color=SERIES[0], lw=1.6, label="output")
    ax.set_xlim(0, 300)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "time (ns)", "V", "Unity-gain buffer, 0.6 V step")
    save(fig, "ota_step.png")
    return m


def comp_sweep(sim, s):
    rows, jobs, keys = [], {}, []
    for cc in (0.5e-12, 1e-12, 2e-12):
        for rz in (0.0, 500.0, 1000.0, 1500.0, 2500.0, 4000.0):
            tag = f"cs{len(keys)}_a"
            jobs[tag] = ota.bench_ac(tag, Process.at("tt"), replace(s, cc=cc, rz=max(rz, 1e-3)))
            keys.append((tag, cc, rz))
    res = sim.run(jobs)
    for tag, cc, rz in keys:
        ac = res[tag].table(f"{tag}_ac.dat")
        m = ac_metrics(ac["frequency"], ac["h"])
        rows.append({"cc_pf": cc * 1e12, "rz_ohm": rz, "dc_gain_db": m.dc_gain_db,
                     "ugb_mhz": m.ugb_hz / 1e6, "pm_deg": m.phase_margin_deg})
    return rows


def pvt(sim, s):
    jobs, keys = {}, []
    for i, (c, t, v) in enumerate(pvt_grid()):
        p = Process.at(c)
        jobs[f"pvt{i}_a"] = ota.bench_ac(f"pvt{i}_a", p, s, v, t)
        jobs[f"pvt{i}_o"] = ota.bench_offset(f"pvt{i}_o", p, s, v, t)
        keys.append((i, c, t, v))
    res = sim.run(jobs)
    rows = []
    for i, c, t, v in keys:
        m = ota.extract(res, f"pvt{i}", full=False)
        rows.append({"corner": c, "temp_c": t, "vdd": v, **m})
    return rows


def monte_carlo(sim, s, n, seed0=1000):
    jobs = {}
    for k in range(n):
        p = Process.at("tt", seed=seed0 + k, mismatch=True)
        jobs[f"mc{k}_a"] = ota.bench_ac(f"mc{k}_a", p, s)
        # same seed -> identical mismatch draw for the offset bench
        p = Process.at("tt", seed=seed0 + k, mismatch=True)
        jobs[f"mc{k}_o"] = ota.bench_offset(f"mc{k}_o", p, s)
    res = sim.run(jobs)
    return [{"seed": seed0 + k, **ota.extract(res, f"mc{k}", full=False)} for k in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=200)
    args = ap.parse_args(argv)
    sim = NgSpice(parallel=2)
    s = ota.OtaSizing()

    m = nominal(sim, s)
    write_csv("ota_nominal.csv", [m])
    cs = comp_sweep(sim, s)
    write_csv("ota_comp_sweep.csv", cs)
    pv = pvt(sim, s)
    write_csv("ota_pvt.csv", pv)
    mc = monte_carlo(sim, s, args.mc)
    write_csv("ota_mc.csv", mc)

    fig, ax = new_fig()
    ax.hist([r["offset_mv"] for r in mc], bins=25, color=SERIES[0], edgecolor="white", linewidth=1)
    style_axes(ax, "input offset (mV)", "count", f"OTA offset, Monte Carlo mismatch (n={args.mc})")
    save(fig, "ota_mc_offset.png")

    def span(rows, k):
        vals = [r[k] for r in rows]
        return min(vals), max(vals)

    lines = ["# OTA results (ngspice, approximate 180 nm-class models)", "",
             "## Nominal (TT, 27 C, VDD = 1.8 V, CL = 2 pF, Ibias = 20 uA)", "",
             md_table([m]), "", "## Compensation sweep (TT)", "", md_table(cs), "",
             "## PVT (5 corners x {-40, 27, 125} C x VDD +/-10 %)", ""]
    summ = []
    for k in ("dc_gain_db", "ugb_hz", "pm_deg", "idd_ua", "offset_mv"):
        lo, hi = span(pv, k)
        summ.append({"metric": k, "min": lo, "max": hi})
    worst = min(pv, key=lambda r: r["pm_deg"])
    lines += [md_table(summ), "", f"Worst-case phase margin: {worst['pm_deg']:.1f} deg at "
              f"{worst['corner']}, {worst['temp_c']} C, {worst['vdd']} V.", "",
              f"## Monte Carlo mismatch (n = {args.mc}, TT)", ""]
    mcs = [{"metric": k, **stats([r[k] for r in mc])} for k in ("offset_mv", "dc_gain_db", "ugb_hz", "pm_deg")]
    lines += [md_table(mcs), ""]
    write_md("ota_summary.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
