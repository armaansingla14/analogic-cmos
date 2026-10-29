# Bandgap results (ngspice, approximate models)

## Nominal (TT, 1.8 V, -40 to 125 C); line/PSRR measured on the trimmed design

| design | r2 | r2_kohm | vref_27c_v | tempco_ppm_per_c | iq_27c_ua | line_reg_mv_per_v | psrr_dc_db | psrr_1mhz_db |
|---|---|---|---|---|---|---|---|---|
| baseline_pmos_opamp | untrim | 108.4 | 1.247 | 7.682 | 26.27 | nan | nan | nan |
| baseline_pmos_opamp | trim | 108.4 | 1.247 | 7.682 | 26.27 | 69.57 | 24.94 | 27.16 |
| improved_nmos_ls_opamp | untrim | 108.4 | 1.234 | 21.6 | 30.6 | nan | nan | nan |
| improved_nmos_ls_opamp | trim | 106.9 | 1.226 | 8.549 | 30.6 | 3.277 | 51.84 | 36.67 |

## Corners (tt/ff/ss/fs/sf) x VDD (1.62/1.8/1.98 V), trimmed R2

| design | metric | min | max |
|---|---|---|---|
| baseline_pmos_opamp | vref_27c_v | 1.231 | 1.312 |
| baseline_pmos_opamp | tempco_ppm_per_c | 7.682 | 867.1 |
| baseline_pmos_opamp | iq_27c_ua | 22.62 | 31.91 |
| improved_nmos_ls_opamp | vref_27c_v | 1.223 | 1.23 |
| improved_nmos_ls_opamp | tempco_ppm_per_c | 8.549 | 24.63 |
| improved_nmos_ls_opamp | iq_27c_ua | 27.52 | 34.38 |

## Monte Carlo mismatch (n = 200 per design, TT, trimmed nominal R2, no per-sample trim)

| design | metric | n | mean | std | min | max |
|---|---|---|---|---|---|---|
| baseline_pmos_opamp | vref_27c_v | 200 | 1.249 | 0.03362 | 1.167 | 1.329 |
| baseline_pmos_opamp | tempco_ppm_per_c | 200 | 39.29 | 26.29 | 4.538 | 112.6 |
| improved_nmos_ls_opamp | vref_27c_v | 200 | 1.229 | 0.02861 | 1.162 | 1.342 |
| improved_nmos_ls_opamp | tempco_ppm_per_c | 200 | 28.51 | 17.23 | 6.682 | 101.3 |

## Start-up (improved design, 2 us supply ramp)

| startup_settle_1pct_us | startup_final_v |
|---|---|
| 1.67 | 1.227 |
