# PLL results

Lock criteria: reference-cycle-averaged VCO frequency within 0.1 % of target AND |phase error| < 1 % of the reference period, for the rest of the record.

## Loop design (TT)

| fref_mhz | n_div | fout_mhz | icp_ua | kvco_mhz_per_v | vctrl_lock_v | r_kohm | c1_pf | c2_pf | fc_lin_mhz | pm_lin_deg |
|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 16 | 400 | 50 | 947.9 | 0.9729 | 2.857 | 166.3 | 12.87 | 1.25 | 60 |

## VCO (transistor level, 5-stage current-starved ring)

| cond | fmin_mhz | fmax_mhz | vctrl_at_400mhz | kvco_mhz_per_v | power_at_400mhz_uw |
|---|---|---|---|---|---|
| tt_27 | 58.33 | 680.9 | 0.9729 | 947.9 | 381.6 |
| ff_27 | 58.81 | 777.8 | 0.8902 | 1139 | 378.2 |
| ss_27 | 57.06 | 589 | 1.073 | 722.5 | 393.3 |
| tt_m40 | 102 | 826.7 | 0.911 | 1350 | 370.4 |
| tt_125 | 63.58 | 534 | 1.122 | 452.8 | 414.9 |

## Charge pump (transistor level)

| cond | iup_0v9_ua | idn_0v9_ua | mismatch_0v9_pct | window5pct_lo_v | window5pct_hi_v |
|---|---|---|---|---|---|
| tt_27 | 49.79 | 49.96 | -0.3364 | 0.32 | 1.34 |
| ff_27 | 50.12 | 50.1 | 0.04908 | 0.32 | 1.38 |
| ss_27 | 49.45 | 49.81 | -0.7317 | 0.34 | 1.32 |
| tt_m40 | 49.95 | 50.03 | -0.1725 | 0.28 | 1.42 |
| tt_125 | 49.56 | 49.83 | -0.5405 | 0.4 | 1.22 |

## Behavioural model (measured VCO / CP tables, TT)

Cold start from Vctrl = 0 V; hop = divider change 16 -> 18 (400 -> 450 MHz) from lock.

| cold_freq_lock_us | cold_phase_lock_us | hop_freq_lock_us | hop_phase_lock_us |
|---|---|---|---|
| 5.12 | 5 | 1.52 | 1.32 |

## Loop bandwidth sweep (behavioural, hop 16 -> 18)

| fc_target_mhz | fc_lin_mhz | pm_lin_deg | r_kohm | c1_pf | c2_pf | hop_freq_lock_us | hop_phase_lock_us |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 60 | 1.143 | 1040 | 80.42 | 3.72 | 3.64 |
| 0.75 | 0.75 | 60 | 1.714 | 462.1 | 35.74 | 2.52 | 2.36 |
| 1 | 1 | 60 | 2.285 | 259.9 | 20.1 | 1.88 | 1.72 |
| 1.25 | 1.25 | 60 | 2.857 | 166.3 | 12.87 | 1.52 | 1.32 |
| 1.5 | 1.5 | 60 | 3.428 | 115.5 | 8.935 | 1.28 | 1.08 |
| 2 | 2 | 60 | 4.57 | 64.98 | 5.026 | 0.9601 | 0.76 |

## Mixed-level ngspice loop (XSPICE PFD/divider, transistor CP, d_osc VCO), cold start

| cond | freq_lock_us | phase_lock_us | vctrl_final_v | fvco_final_mhz | residual_perr_pct |
|---|---|---|---|---|---|
| tt_27 | 5.2 | 5 | 0.9729 | 400 | -0.5865 |
| ff_27 | 4.92 | 4.76 | 0.8903 | 400 | -0.6497 |
| ss_27 | 5.48 | 5.24 | 1.073 | 400 | -0.4821 |
