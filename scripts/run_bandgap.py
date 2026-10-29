"""Bandgap reference: baseline (PMOS-input op-amp) vs level-shifted NMOS-input
op-amp. Temperature sweeps (untrimmed / trimmed), corners, line regulation,
PSRR, start-up and Monte Carlo mismatch."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analogic import bandgap as bg  # noqa: E402
from analogic.models import CORNERS, Process  # noqa: E402
from analogic.report import (SERIES, md_table, new_fig, save, stats, style_axes,  # noqa: E402
                             write_csv, write_md)
from analogic.spice import NgSpice  # noqa: E402

# Each variant uses the mirror sizing it works with: the PMOS-input op-amp
# runs out of output swing with the larger mirror devices.
DESIGNS = {
    "baseline_pmos_opamp": dict(opamp="pmos", wm=4e-6, lm=2e-6),
    "improved_nmos_ls_opamp": dict(opamp="nmos_ls", wm=16e-6, lm=4e-6),
}


def line_psrr(sim, tag, s):
    core = bg.bandgap_core(Process.at("tt"), s)
    state = bg.settled_states(sim, {tag: (core, 1.8)}, temp=27.0)[tag]
    res = sim.run({tag: bg.bench_line(tag, core, state=state)})[tag]
    up, dn = res.table(f"{tag}_lineup.dat"), res.table(f"{tag}_linedn.dat")
    v = np.concatenate([dn["v-sweep"][::-1], up["v-sweep"][1:]])
    vr = np.concatenate([dn["v(vref)"][::-1], up["v(vref)"][1:]])
    m = (v >= 1.6) & (v <= 2.0)
    ps = res.table(f"{tag}_psrr.dat")
    mag = np.abs(ps["ps"])
    return {
        "line_reg_mv_per_v": float(np.polyfit(v[m], vr[m], 1)[0] * 1e3),
        "psrr_dc_db": float(-20 * np.log10(mag[0])),
        "psrr_1mhz_db": float(-20 * np.log10(np.interp(1e6, ps["frequency"], mag))),
    }


def startup(sim, s):
    core = bg.bandgap_core(Process.at("tt"), s)
    st = sim.run({"start": bg.bench_startup("start", core)})["start"].table("start_start.dat")
    vfinal = st["v(vref)"][-1]
    bad = np.where(np.abs(st["v(vref)"] - vfinal) >= 0.01 * vfinal)[0]
    t_settle = float(st["time"][bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(st["time"]) else float("nan")
    return st, {"startup_settle_1pct_us": t_settle * 1e6, "startup_final_v": float(vfinal)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=200)
    args = ap.parse_args(argv)
    sim = NgSpice(parallel=2)

    nominal_rows, curves, trimmed, mc_rows_all, corner_rows_all = [], {}, {}, {}, {}
    for dname, kw in DESIGNS.items():
        s0 = bg.BandgapSizing(**kw)
        s0 = replace(s0, r2=bg.first_order_r2(s0))
        s1 = bg.trim_r2(sim, s0)
        trimmed[dname] = s1
        nom = bg.run_temp(sim, {"untrim": (Process.at("tt"), s0, 1.8), "trim": (Process.at("tt"), s1, 1.8)})
        curves[dname] = nom
        lp = line_psrr(sim, "line", s1)
        for tag, s in (("untrim", s0), ("trim", s1)):
            r = nom[tag]
            nominal_rows.append({"design": dname, "r2": tag, "r2_kohm": s.r2 / 1e3,
                                 "vref_27c_v": r["vref27"], "tempco_ppm_per_c": r["tc_ppm"],
                                 "iq_27c_ua": r["iq27_ua"],
                                 **(lp if tag == "trim" else {k: float("nan") for k in lp})})

        items = {f"c_{c}_{int(round(v * 100))}": (Process.at(c), s1, v)
                 for c in CORNERS for v in (1.62, 1.8, 1.98)}
        cr = bg.run_temp(sim, items)
        corner_rows_all[dname] = [{"design": dname, "corner": k.split("_")[1], "vdd": items[k][2],
                                   "vref_27c_v": r["vref27"], "tempco_ppm_per_c": r["tc_ppm"],
                                   "iq_27c_ua": r["iq27_ua"]} for k, r in cr.items()]

        mc_items = {f"mc{k}": (Process.at("tt", seed=5000 + k, mismatch=True), s1, 1.8) for k in range(args.mc)}
        mc = bg.run_temp(sim, mc_items)
        mc_rows_all[dname] = [{"design": dname, "seed": 5000 + k, "vref_27c_v": mc[f"mc{k}"]["vref27"],
                               "tempco_ppm_per_c": mc[f"mc{k}"]["tc_ppm"]} for k in range(args.mc)]

    write_csv("bandgap_nominal.csv", nominal_rows)
    write_csv("bandgap_corners.csv", sum(corner_rows_all.values(), []))
    write_csv("bandgap_mc.csv", sum(mc_rows_all.values(), []))
    imp = "improved_nmos_ls_opamp"
    write_csv("bandgap_temp_curve.csv", [
        {"temp_c": t, **{f"vref_{d}_trim_v": curves[d]["trim"]["vref"][i] for d in DESIGNS}}
        for i, t in enumerate(curves[imp]["trim"]["temp"])])

    st, su = startup(sim, trimmed[imp])
    write_csv("bandgap_startup.csv", [su])

    fig, ax = new_fig()
    for i, d in enumerate(DESIGNS):
        c = curves[d]["trim"]
        ax.plot(c["temp"], c["vref"] * 1e3, color=SERIES[i], lw=1.6,
                label=f"{d} ({c['tc_ppm']:.1f} ppm/C)")
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, "temperature (C)", "VREF (mV)", "Bandgap VREF vs temperature (TT, 1.8 V, trimmed R2)")
    save(fig, "bandgap_temp.png")

    fig, ax = new_fig()
    ax.plot(st["time"] * 1e6, st["v(vdd)"], color=SERIES[1], lw=1.2, label="VDD")
    ax.plot(st["time"] * 1e6, st["v(vref)"], color=SERIES[0], lw=1.6, label="VREF")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "time (us)", "V", "Bandgap start-up (2 us supply ramp)")
    save(fig, "bandgap_startup.png")

    fig, ax = new_fig()
    for i, d in enumerate(DESIGNS):
        ax.hist([r["vref_27c_v"] * 1e3 for r in mc_rows_all[d]], bins=25, color=SERIES[i], alpha=0.75,
                edgecolor="white", linewidth=1, label=d)
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, "VREF at 27 C (mV)", "count", f"Bandgap Monte Carlo mismatch (n={args.mc} each)")
    save(fig, "bandgap_mc.png")

    corner_summary, mc_summary = [], []
    for d in DESIGNS:
        rows = corner_rows_all[d]
        for k in ("vref_27c_v", "tempco_ppm_per_c", "iq_27c_ua"):
            vals = [r[k] for r in rows]
            corner_summary.append({"design": d, "metric": k, "min": min(vals), "max": max(vals)})
        for k in ("vref_27c_v", "tempco_ppm_per_c"):
            mc_summary.append({"design": d, "metric": k, **stats([r[k] for r in mc_rows_all[d]])})

    cols = ["design", "r2", "r2_kohm", "vref_27c_v", "tempco_ppm_per_c", "iq_27c_ua",
            "line_reg_mv_per_v", "psrr_dc_db", "psrr_1mhz_db"]
    text = "\n".join([
        "# Bandgap results (ngspice, approximate models)", "",
        "## Nominal (TT, 1.8 V, -40 to 125 C); line/PSRR measured on the trimmed design", "",
        md_table(nominal_rows, cols), "",
        "## Corners (tt/ff/ss/fs/sf) x VDD (1.62/1.8/1.98 V), trimmed R2", "", md_table(corner_summary), "",
        f"## Monte Carlo mismatch (n = {args.mc} per design, TT, trimmed nominal R2, no per-sample trim)", "",
        md_table(mc_summary), "",
        "## Start-up (improved design, 2 us supply ramp)", "", md_table([su]), "",
    ])
    write_md("bandgap_summary.md", text)
    print(text)


if __name__ == "__main__":
    main()
