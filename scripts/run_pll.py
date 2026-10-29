"""Charge-pump PLL: transistor-level VCO and charge-pump characterization,
loop-filter design, behavioural and mixed-level ngspice lock-time runs."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analogic import pll  # noqa: E402
from analogic import pll_model as pm  # noqa: E402
from analogic.models import Process  # noqa: E402
from analogic.report import SERIES, md_table, new_fig, save, style_axes, write_csv, write_md  # noqa: E402
from analogic.spice import NgSpice  # noqa: E402

FREF = 25e6
NDIV = 16
F_TARGET = FREF * NDIV
VGRID = np.round(np.arange(0.4, 1.801, 0.05), 3)


def kvco_at(tbl, f):
    v0 = float(np.interp(f, tbl["freq"], tbl["vctrl"]))
    k = np.gradient(tbl["freq"], tbl["vctrl"])
    return v0, float(np.interp(v0, tbl["vctrl"], k))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", type=float, default=1.25e6, help="loop crossover target (Hz)")
    ap.add_argument("--pm", type=float, default=60.0)
    ap.add_argument("--icp", type=float, default=50e-6)
    args = ap.parse_args(argv)
    sim = NgSpice(parallel=2)
    vs, cs = pll.VcoSizing(), replace(pll.CpSizing(), icp=args.icp)

    # 1. VCO tuning curves
    conds = {"tt_27": ("tt", 27.0), "ff_27": ("ff", 27.0), "ss_27": ("ss", 27.0),
             "tt_m40": ("tt", -40.0), "tt_125": ("tt", 125.0)}
    vco = {k: pll.vco_curve(sim, Process.at(c), vs, VGRID, temp=t, tag=f"v{k}") for k, (c, t) in conds.items()}
    vrows = []
    for k, tb in vco.items():
        v0, k0 = kvco_at(tb, F_TARGET)
        vrows.append({"cond": k, "fmin_mhz": tb["freq"][tb["freq"] > 0].min() / 1e6, "fmax_mhz": tb["freq"].max() / 1e6,
                      "vctrl_at_400mhz": v0, "kvco_mhz_per_v": k0 / 1e6,
                      "power_at_400mhz_uw": float(np.interp(v0, tb["vctrl"], tb["power"]) * 1e6)})
    write_csv("pll_vco.csv", vrows)
    write_csv("pll_vco_curves.csv", [{"vctrl": v, **{k: vco[k]["freq"][i] for k in vco}} for i, v in enumerate(VGRID)])
    fig, ax = new_fig()
    for i, k in enumerate(vco):
        ax.plot(vco[k]["vctrl"], vco[k]["freq"] / 1e6, color=SERIES[i], lw=1.4, label=k)
    ax.axhline(F_TARGET / 1e6, color="#9a9994", lw=0.8, ls="--")
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, "Vctrl (V)", "frequency (MHz)", "Current-starved ring VCO tuning (transistor level)")
    save(fig, "pll_vco.png")

    # 2. Charge pump
    cpc = pll.cp_curves(sim, {k: (Process.at(c), cs, 1.8, t) for k, (c, t) in conds.items()})
    crows = []
    for k, c in cpc.items():
        lo, hi = pll.cp_mismatch_window(c, 0.05)
        i09 = int(np.argmin(np.abs(c["vout"] - 0.9)))
        crows.append({"cond": k, "iup_0v9_ua": c["iup"][i09] * 1e6, "idn_0v9_ua": c["idn"][i09] * 1e6,
                      "mismatch_0v9_pct": (c["iup"][i09] - c["idn"][i09]) / c["idn"][i09] * 100,
                      "window5pct_lo_v": lo, "window5pct_hi_v": hi})
    write_csv("pll_cp.csv", crows)
    c = cpc["tt_27"]
    fig, ax = new_fig()
    ax.plot(c["vout"], c["iup"] * 1e6, color=SERIES[0], lw=1.6, label="UP")
    ax.plot(c["vout"], c["idn"] * 1e6, color=SERIES[1], lw=1.6, label="DN")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "output voltage (V)", "current (uA)", "Charge-pump currents (TT, 27 C)")
    save(fig, "pll_cp.png")

    # 3. Loop design at TT
    tt = vco["tt_27"]
    v0, k0 = kvco_at(tt, F_TARGET)
    fv = pm.design_loop_filter(args.icp, k0, NDIV, args.fc, args.pm)
    fc_lin, pm_lin = pm.loop_margins(args.icp, k0, NDIV, fv)

    # 4. Behavioural model: cold start and channel hop, plus fc sweep
    def behave(fvals, t_stop=10e-6, **kw):
        cfg = pm.BehavioralConfig(fref=FREF, n_div=NDIV, t_stop=t_stop, **kw)
        return pm.simulate(cfg, fvals, tt["vctrl"], tt["freq"], c["vout"], c["iup"], c["idn"])

    cold = behave(fv)
    lk_cold = pm.lock_time_from_edges(cold["ref_edges"], cold["div_edges"], NDIV, FREF)
    # hop: start locked at N=16 (pre-charged filter) and switch to N=18 at 1 us
    hop = behave(fv, t_stop=8e-6, v_init=v0, hop_n=18, t_hop=1e-6)
    re, de = hop["ref_edges"], hop["div_edges"]
    lk_hop = pm.lock_time_from_edges(re[re > 1.5e-6], de[de > 1.5e-6], 18, FREF)
    hop_rel = lk_hop["freq_lock_s"] - 1e-6, lk_hop["phase_lock_s"] - 1e-6

    sweep_rows = []
    for fc in (0.5e6, 0.75e6, 1.0e6, 1.25e6, 1.5e6, 2.0e6):
        fvs = pm.design_loop_filter(args.icp, k0, NDIV, fc, args.pm)
        r = behave(fvs, t_stop=8e-6, v_init=v0, hop_n=18, t_hop=1e-6)
        re, de = r["ref_edges"], r["div_edges"]
        lk = pm.lock_time_from_edges(re[re > 1.5e-6], de[de > 1.5e-6], 18, FREF)
        fcl, pml = pm.loop_margins(args.icp, k0, NDIV, fvs)
        sweep_rows.append({"fc_target_mhz": fc / 1e6, "fc_lin_mhz": fcl / 1e6, "pm_lin_deg": pml,
                           "r_kohm": fvs.r / 1e3, "c1_pf": fvs.c1 * 1e12, "c2_pf": fvs.c2 * 1e12,
                           "hop_freq_lock_us": (lk["freq_lock_s"] - 1e-6) * 1e6,
                           "hop_phase_lock_us": (lk["phase_lock_s"] - 1e-6) * 1e6})
    write_csv("pll_fc_sweep.csv", sweep_rows)

    # 5. Mixed-level ngspice closed loop (TT + process corners, cold start)
    ng_rows, ng_traces = [], {}
    for k in ("tt_27", "ff_27", "ss_27"):
        corner = conds[k][0]
        lp = pll.LoopParams(fref=FREF, n_div=NDIV, r=fv.r, c1=fv.c1, c2=fv.c2, t_stop=10e-6)
        res = pll.run_loop(sim, f"loop_{k}", Process.at(corner), cs, lp, vco[k])
        lk = pll.loop_lock(res, FREF, F_TARGET)
        ng_traces[k] = res
        ng_rows.append({"cond": k, "freq_lock_us": lk["freq_lock_s"] * 1e6, "phase_lock_us": lk["phase_lock_s"] * 1e6,
                        "vctrl_final_v": float(np.mean(res["vctrl"][-2000:])),
                        "fvco_final_mhz": float(np.mean(lk["fav"][-20:]) / 1e6),
                        "residual_perr_pct": float(np.mean(lk["perr"][-20:]) * 100)})
    write_csv("pll_ngspice_loop.csv", ng_rows)

    fig, ax = new_fig()
    ax.plot(cold["time"] * 1e6, cold["vctrl"], color=SERIES[0], lw=1.6, label="behavioural model")
    ax.plot(ng_traces["tt_27"]["time"] * 1e6, ng_traces["tt_27"]["vctrl"], color=SERIES[1], lw=1.0,
            label="ngspice mixed-level")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "time (us)", "Vctrl (V)", "PLL cold-start acquisition (TT, 400 MHz)")
    save(fig, "pll_lock.png")

    design = {"fref_mhz": FREF / 1e6, "n_div": NDIV, "fout_mhz": F_TARGET / 1e6, "icp_ua": args.icp * 1e6,
              "kvco_mhz_per_v": k0 / 1e6, "vctrl_lock_v": v0, "r_kohm": fv.r / 1e3, "c1_pf": fv.c1 * 1e12,
              "c2_pf": fv.c2 * 1e12, "fc_lin_mhz": fc_lin / 1e6, "pm_lin_deg": pm_lin}
    beh = {"cold_freq_lock_us": lk_cold["freq_lock_s"] * 1e6, "cold_phase_lock_us": lk_cold["phase_lock_s"] * 1e6,
           "hop_freq_lock_us": hop_rel[0] * 1e6, "hop_phase_lock_us": hop_rel[1] * 1e6}
    write_csv("pll_design.csv", [design])
    write_csv("pll_behavioral.csv", [beh])
    text = "\n".join([
        "# PLL results", "",
        "Lock criteria: reference-cycle-averaged VCO frequency within 0.1 % of target AND "
        "|phase error| < 1 % of the reference period, for the rest of the record.", "",
        "## Loop design (TT)", "", md_table([design]), "",
        "## VCO (transistor level, 5-stage current-starved ring)", "", md_table(vrows), "",
        "## Charge pump (transistor level)", "", md_table(crows), "",
        "## Behavioural model (measured VCO / CP tables, TT)", "",
        "Cold start from Vctrl = 0 V; hop = divider change 16 -> 18 (400 -> 450 MHz) from lock.", "",
        md_table([beh]), "",
        "## Loop bandwidth sweep (behavioural, hop 16 -> 18)", "", md_table(sweep_rows), "",
        "## Mixed-level ngspice loop (XSPICE PFD/divider, transistor CP, d_osc VCO), cold start", "",
        md_table(ng_rows), "",
    ])
    write_md("pll_summary.md", text)
    print(text)


if __name__ == "__main__":
    main()
