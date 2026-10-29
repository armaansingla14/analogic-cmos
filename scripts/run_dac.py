"""8-bit current-steering DAC: binary vs 4+4 segmented, static linearity under
Monte Carlo mismatch, mid-scale glitch, SFDR, and unit-cell area sweep."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analogic import dac  # noqa: E402
from analogic.metrics import sfdr_db  # noqa: E402
from analogic.models import Process  # noqa: E402
from analogic.report import (SERIES, md_table, new_fig, save, stats, style_axes,  # noqa: E402
                             write_csv, write_md)
from analogic.spice import NgSpice  # noqa: E402

ARCHS = ("binary", "segmented")


def lin_row(tag, arch, r, **extra):
    return {"tag": tag, "arch": arch, **extra,
            "lsb_mv": r["endpoint"].lsb * 1e3,
            "dnl_max_lsb": r["endpoint"].dnl_max, "inl_ep_max_lsb": r["endpoint"].inl_max,
            "inl_bf_max_lsb": r["bestfit"].inl_max, "monotonic": r["endpoint"].monotonic,
            "glitch_fvs": r["glitch"] * 1e15}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=40)
    ap.add_argument("--area-mc", type=int, default=15)
    ap.add_argument("--hold", type=float, default=20e-9)
    args = ap.parse_args(argv)
    sim = NgSpice(parallel=2)
    base = {a: dac.DacSizing(arch=a, t_hold=args.hold) for a in ARCHS}

    # nominal (no mismatch) and Monte Carlo
    items = {f"nom_{a}": (Process.at("tt"), base[a]) for a in ARCHS}
    for a in ARCHS:
        for k in range(args.mc):
            items[f"mc_{a}_{k}"] = (Process.at("tt", seed=9000 + k, mismatch=True), base[a])
    res = dac.run_linearity(sim, items)
    rows = []
    for tag, r in res.items():
        arch = items[tag][1].arch
        rows.append(lin_row(tag, arch, r, mismatch=tag.startswith("mc")))
    write_csv("dac_linearity.csv", rows)

    # worst-INL sample plot for the segmented design + DNL of both archs (same seed)
    k_plot = 0
    fig, (a1, a2) = new_fig(2, 1, 5.2, 4.8)
    for i, a in enumerate(ARCHS):
        r = res[f"mc_{a}_{k_plot}"]
        a1.plot(np.arange(255), r["endpoint"].dnl, color=SERIES[i], lw=1.0, label=a)
        a2.plot(np.arange(256), r["endpoint"].inl, color=SERIES[i], lw=1.2, label=a)
    a1.legend(frameon=False, fontsize=8)
    style_axes(a1, "", "DNL (LSB)", f"DAC linearity, Monte Carlo sample seed {9000 + k_plot}")
    style_axes(a2, "code", "INL (LSB, endpoint)")
    save(fig, "dac_inl_dnl.png")

    # SFDR: nominal + first few MC samples
    sf_rows = []
    sf_jobs = {}
    cyc, n = 7, 256
    codes = dac.sine_codes(n, cyc)
    for a in ARCHS:
        sf_jobs[f"sf_{a}_nom"] = (Process.at("tt"), base[a])
        for k in range(min(5, args.mc)):
            sf_jobs[f"sf_{a}_{k}"] = (Process.at("tt", seed=9000 + k, mismatch=True), base[a])
    sres = sim.run({t: dac.bench(t, p, s, codes) for t, (p, s) in sf_jobs.items()})
    for t, (p, s) in sf_jobs.items():
        lv = dac.sample_codes(sres[t].table(f"{t}_tran.dat"), n, s.t_hold)
        sf_rows.append({"tag": t, "arch": s.arch, "sfdr_dbc": sfdr_db(lv, cyc)})
    write_csv("dac_sfdr.csv", sf_rows)

    # unit current-source area sweep (segmented)
    area_rows = []
    sweep = [(0.5e-6, 2e-6), (1.0e-6, 4e-6), (1.5e-6, 6e-6), (3e-6, 12e-6)]
    a_items = {}
    for j, (w, l) in enumerate(sweep):
        s = replace(base["segmented"], wcs=w, lcs=l)
        for k in range(args.area_mc):
            a_items[f"ar{j}_{k}"] = (Process.at("tt", seed=9500 + k, mismatch=True), s)
    ares = dac.run_linearity(sim, a_items)
    for j, (w, l) in enumerate(sweep):
        inl = [ares[f"ar{j}_{k}"]["endpoint"].inl_max for k in range(args.area_mc)]
        dnl = [ares[f"ar{j}_{k}"]["endpoint"].dnl_max for k in range(args.area_mc)]
        area_rows.append({"wcs_um": w * 1e6, "lcs_um": l * 1e6, "unit_area_um2": w * l * 1e12,
                          "n": args.area_mc, "inl_mean_lsb": float(np.mean(inl)), "inl_max_lsb": float(np.max(inl)),
                          "dnl_mean_lsb": float(np.mean(dnl)), "dnl_max_lsb": float(np.max(dnl))})
    write_csv("dac_area_sweep.csv", area_rows)

    summary = []
    for a in ARCHS:
        mc = [r for r in rows if r["arch"] == a and r["mismatch"]]
        nom = next(r for r in rows if r["tag"] == f"nom_{a}")
        sf_nom = next(r["sfdr_dbc"] for r in sf_rows if r["tag"] == f"sf_{a}_nom")
        sf_mc = [r["sfdr_dbc"] for r in sf_rows if r["arch"] == a and not r["tag"].endswith("nom")]
        summary.append({
            "arch": a, "nom_inl_lsb": nom["inl_ep_max_lsb"], "nom_dnl_lsb": nom["dnl_max_lsb"],
            "mc_n": len(mc),
            "mc_inl_mean": stats([r["inl_ep_max_lsb"] for r in mc])["mean"],
            "mc_inl_worst": max(r["inl_ep_max_lsb"] for r in mc),
            "mc_dnl_mean": stats([r["dnl_max_lsb"] for r in mc])["mean"],
            "mc_dnl_worst": max(r["dnl_max_lsb"] for r in mc),
            "mc_monotonic_frac": float(np.mean([r["monotonic"] for r in mc])),
            "glitch_nom_fVs": nom["glitch_fvs"],
            "sfdr_nom_dbc": sf_nom, "sfdr_mc_mean_dbc": float(np.mean(sf_mc)) if sf_mc else float("nan"),
        })
    write_csv("dac_summary.csv", summary)
    text = "\n".join([
        "# DAC results (ngspice transient, approximate models)", "",
        f"8-bit, unit current {base['segmented'].iunit * 1e6:.1f} uA, RL = {base['segmented'].rl:.0f} ohm per side, "
        f"code hold {args.hold * 1e9:.0f} ns. INL is endpoint-fit; DNL/INL are |max| over codes.", "",
        md_table(summary), "", "## Unit current-source area sweep (segmented, Monte Carlo)", "",
        md_table(area_rows), "",
    ])
    write_md("dac_summary.md", text)
    print(text)


if __name__ == "__main__":
    main()
