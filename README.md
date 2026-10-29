# analogic-cmos

CMOS analog building blocks designed and characterized in simulation: a
two-stage Miller OTA, an op-amp based bandgap reference, an 8-bit
current-steering DAC, and a charge-pump PLL (ring VCO, charge pump, PFD,
loop filter). Netlists are generated from Python and simulated with
**ngspice** (AC, noise, transient, DC sweeps, PVT corners and Monte Carlo
mismatch); Python parses the results, extracts the metrics and draws the
plots.

> **Models are approximate.** No foundry PDK is used. `models/cmos180_approx.lib`
> is a hand-written BSIM3v3 card loosely shaped after public 180 nm-class
> values, plus a Gummel-Poon substrate PNP. Corners are emulated with
> threshold/mobility shifts and mismatch with a Pelgrom model
> (`analogic/models.py`). All numbers below are simulation results with
> these models and say nothing about any real process.

## Features

| Block | Topology | Analyses |
|---|---|---|
| OTA | NMOS diff pair, PMOS mirror load, PMOS CS second stage, Miller Cc + nulling Rz | open-loop AC (gain, UGB, PM, GM), input noise, unity-gain step (slew, settling), PSRR, CMRR, offset, Cc/Rz sweep, 45-point PVT grid, 200-sample Monte Carlo |
| Bandgap | PNP core (N = 8), PMOS mirrors, self-biased op-amp, 3-transistor start-up | -40..125 C sweep, R2 trim, line regulation, PSRR, start-up transient, corners x VDD, 200-sample Monte Carlo; two op-amp variants compared |
| DAC | 8-bit, cascoded NMOS unit cells + differential switches; binary vs 4+4 segmented | full 256-code transient ramp, endpoint/best-fit INL and DNL, mid-scale glitch, SFDR from a coherent 256-sample sine, Monte Carlo, unit-area sweep |
| PLL | 5-stage current-starved ring VCO, drain-switched charge pump, 2nd-order passive filter, XSPICE PFD/divider | VCO tuning curves (corners/temperature), CP current vs Vout, loop-filter design, behavioural and mixed-level ngspice lock transients, loop-bandwidth sweep |

## Layout

```
models/cmos180_approx.lib   approximate BSIM3v3 + PNP model card
analogic/spice.py           ngspice batch runner (native or docker), wrdata parser
analogic/models.py          corners, Pelgrom mismatch, MOSFET/resistor line generation
analogic/metrics.py         gain/UGB/PM, slew, settling, tempco, INL/DNL, lock time, SFDR
analogic/ota.py             OTA netlist + benches
analogic/bandgap.py         bandgap netlist, benches, R2 trim
analogic/dac.py             DAC cells, ideal decoder, ramp/sine benches
analogic/pll.py             VCO, charge pump, mixed-level closed loop (ngspice)
analogic/pll_model.py       loop-filter design, linear loop analysis, event-driven PLL model
scripts/run_*.py            experiments -> results/*.csv, results/*_summary.md, results/plots/
scripts/export_netlists.py  writes nominal decks to netlists/*.cir
netlists/                   exported ngspice decks (TT, no mismatch)
spectre/                    Spectre-syntax versions of the decks (not run here)
tests/                      pytest: metric extraction, models, parsers, decoder, PLL model, ngspice integration
```

## How to run

ngspice is used natively if it is on `PATH`; otherwise every batch runs inside
the `devtools:latest` docker image (one container per batch).

```bash
pip install numpy scipy matplotlib pandas pytest
python -m pytest -q                      # 30 tests (3 invoke ngspice)
python scripts/run_ota.py --mc 200       # ~20 s
python scripts/run_bandgap.py --mc 200   # ~2 min
python scripts/run_dac.py --mc 30 --area-mc 8   # ~30 min (transistor-level 256-code transients)
python scripts/run_pll.py                # ~3 min
python scripts/run_all.py                # everything
```

## Results

All values were produced by the scripts above (simulated, approximate models).
Full tables are in `results/*_summary.md` and the CSV files next to them.

### OTA (`scripts/run_ota.py --mc 200`)

TT, 27 C, VDD = 1.8 V, CL = 2 pF, Ibias = 20 uA, Cc = 1 pF, Rz = 1.5 kOhm:

| DC gain | UGB | Phase margin | Gain margin | Slew | 1 % settling (0.6 V step) | PSRR (DC) | CMRR (DC) | Input noise @1 MHz | IDD |
|---|---|---|---|---|---|---|---|---|---|
| 80.1 dB | 43.1 MHz | 59.0 deg | 29.5 dB | 38.6 V/us | 26.3 ns | 83.3 dB | 73.3 dB | 68.9 nV/rtHz | 221 uA |

- PVT (5 corners x -40/27/125 C x 1.62/1.8/1.98 V): gain 75.7-83.1 dB, UGB 32.6-54.6 MHz,
  PM 53.3-65.7 deg (worst: fs, 125 C).
- Monte Carlo mismatch (200 samples): input offset sigma 4.27 mV (mean -0.39 mV),
  PM sigma 0.32 deg.
- Compensation sweep: at Cc = 1 pF, Rz = 0 gives PM 36.9 deg; Rz = 1.5 k gives 59.0 deg.
- Input noise at 1 kHz is 2.15 uV/rtHz, dominated by the flicker coefficients chosen for
  the approximate model.

### Bandgap (`scripts/run_bandgap.py --mc 200`)

Two op-amp variants of the same PNP core were built. The baseline senses the
PNP nodes with a PMOS-input op-amp; the improved version senses level-shifted
nodes with an NMOS-input op-amp whose output is referenced to VDD.

| Design (trimmed R2) | VREF @27 C | Tempco (-40..125 C, box) | Line reg. (1.6-2.0 V) | PSRR DC / 1 MHz | IQ |
|---|---|---|---|---|---|
| baseline, PMOS-input op-amp | 1.247 V | 7.7 ppm/C | 69.6 mV/V | 24.9 / 27.2 dB | 26.3 uA |
| improved, NMOS-input + level shift | 1.226 V | 8.5 ppm/C | 3.3 mV/V | 51.8 / 36.7 dB | 30.6 uA |

| Design | Corners x VDD tempco | MC VREF sigma | MC tempco mean (max) |
|---|---|---|---|
| baseline | 7.7-867 ppm/C (fails at 1.98 V: op-amp out of swing) | 33.6 mV | 39.3 (113) ppm/C |
| improved | 8.5-24.6 ppm/C | 28.6 mV | 28.5 (101) ppm/C |

The improved variant trades a slightly higher untrimmed tempco (21.6 ppm/C before
trim) for ~21x better line regulation and corner robustness. Start-up from a 2 us
supply ramp settles to 1 % in 1.67 us. DC temperature sweeps are seeded from a
start-up transient because a cold DC sweep can converge to a non-physical
high-current solution.

### DAC (`scripts/run_dac.py --mc 30 --area-mc 8`)

8-bit, 4 uA unit current, 400 Ohm per output, 20 ns per code (transistor-level transient).

| Architecture | MC n | INL mean / worst (LSB) | DNL mean / worst (LSB) | Monotonic | SFDR (nominal / MC mean) |
|---|---|---|---|---|---|
| binary-weighted | 30 | 0.097 / 0.159 | 0.123 / 0.266 | 30/30 | 61.1 / 60.8 dBc |
| 4+4 segmented | 30 | 0.114 / 0.223 | 0.078 / 0.147 | 30/30 | 61.1 / 61.0 dBc |

Segmentation reduces DNL (mid-scale carry no longer switches all binary cells)
while INL, set by random unit mismatch, is statistically similar. The unit-area
sweep (segmented, 8 samples each) shows mean INL 0.39 / 0.20 / 0.13 / 0.067 LSB for
1 / 4 / 9 / 36 um^2 current sources, following the expected 1/sqrt(area).
SFDR is limited by 8-bit quantization for this record length.

### PLL (`scripts/run_pll.py`)

25 MHz reference, N = 16, 400 MHz output, Icp = 50 uA, loop designed for
fc = 1.25 MHz / PM = 60 deg (R = 2.86 kOhm, C1 = 166 pF, C2 = 12.9 pF).
Lock = cycle-averaged frequency within 0.1 % and phase error < 1 % of Tref.

| Item | Result |
|---|---|
| VCO (TT) | 58-681 MHz, Kvco = 948 MHz/V at 400 MHz, 382 uW |
| Charge pump (TT) | UP/DN 49.8/50.0 uA at 0.9 V, within 5 % for 0.32-1.34 V |
| Lock time, behavioural model, cold start (Vctrl = 0) | 5.12 us (freq) / 5.00 us (phase) |
| Lock time, mixed-level ngspice, cold start | TT 5.20 us, FF 4.92 us, SS 5.48 us |
| Relock, behavioural, 400 -> 450 MHz hop | 1.52 us (freq) / 1.32 us (phase) |

Behavioural and ngspice loops settle to the same control voltage (0.973 V at TT).
The loop-bandwidth sweep shows hop lock time falling from 3.7 us at 0.5 MHz to
0.96 us at 2 MHz, at the cost of capacitance only in this model (no noise
trade-off is simulated). The ngspice loop shows a residual static phase offset
of about -0.6 % of Tref from charge-pump non-idealities.

## Limitations

- Device models are approximate and uncalibrated; absolute values (noise, leakage,
  tempco, speed) are indicative only. FinFET effects are not modelled.
- Bias currents for the OTA, DAC and charge pump are ideal sources; the DAC
  decoder and the PLL PFD/divider/VCO-in-loop are behavioural (XSPICE).
- The full PLL loop uses the transistor-level VCO only through its measured
  tuning table; phase noise and jitter are not simulated.
- The Spectre decks in `spectre/` were not simulated (no licence or PDK here).
- No layout, parasitic extraction or post-layout verification was done.

## License

MIT, see `LICENSE`.
