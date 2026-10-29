# DAC results (ngspice transient, approximate models)

8-bit, unit current 4.0 uA, RL = 400 ohm per side, code hold 20 ns. INL is endpoint-fit; DNL/INL are |max| over codes.

| arch | nom_inl_lsb | nom_dnl_lsb | mc_n | mc_inl_mean | mc_inl_worst | mc_dnl_mean | mc_dnl_worst | mc_monotonic_frac | glitch_nom_fVs | sfdr_nom_dbc | sfdr_mc_mean_dbc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | 0.0001153 | 6.591e-06 | 30 | 0.09711 | 0.1587 | 0.1225 | 0.2663 | 1 | 835.2 | 61.08 | 60.84 |
| segmented | 0.000115 | 6.581e-06 | 30 | 0.1136 | 0.2229 | 0.07775 | 0.1465 | 1 | 846.3 | 61.08 | 60.98 |

## Unit current-source area sweep (segmented, Monte Carlo)

| wcs_um | lcs_um | unit_area_um2 | n | inl_mean_lsb | inl_max_lsb | dnl_mean_lsb | dnl_max_lsb |
|---|---|---|---|---|---|---|---|
| 0.5 | 2 | 1 | 8 | 0.392 | 0.5567 | 0.2456 | 0.3354 |
| 1 | 4 | 4 | 8 | 0.1994 | 0.2826 | 0.1252 | 0.1716 |
| 1.5 | 6 | 9 | 8 | 0.1337 | 0.1891 | 0.084 | 0.1152 |
| 3 | 12 | 36 | 8 | 0.0671 | 0.0946 | 0.04231 | 0.05797 |
