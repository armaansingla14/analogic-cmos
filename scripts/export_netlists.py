"""Write the nominal (TT, no mismatch) generated ngspice decks to netlists/.

The exported files are self-contained except for the model library include;
run e.g. ``cd netlists && ngspice -b ota_ac.cir`` with
``models/cmos180_approx.lib`` copied or linked next to them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analogic import bandgap as bg  # noqa: E402
from analogic import dac, ota, pll  # noqa: E402
from analogic.models import Process  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "netlists"


def main():
    OUT.mkdir(exist_ok=True)
    tt = Process.at
    s = ota.OtaSizing()
    decks = {
        "ota_ac": ota.bench_ac("ota_ac", tt("tt"), s),
        "ota_noise": ota.bench_noise("ota_noise", tt("tt"), s),
        "ota_tran": ota.bench_tran("ota_tran", tt("tt"), s),
        "ota_psrr": ota.bench_psrr("ota_psrr", tt("tt"), s),
    }
    b = bg.BandgapSizing(r2=106.9e3)
    core = bg.bandgap_core(tt("tt"), b)
    decks["bandgap_temp"] = bg.bench_temp("bandgap_temp", core)
    decks["bandgap_startup"] = bg.bench_startup("bandgap_startup", core)
    decks["bandgap_line_psrr"] = bg.bench_line("bandgap_line_psrr", core)
    ds = dac.DacSizing(t_hold=20e-9)
    decks["dac_seg_ramp"] = dac.bench("dac_seg_ramp", tt("tt"), ds, dac.ramp_codes())
    decks["vco_0v9"] = pll.bench_vco("vco_0v9", tt("tt"), pll.VcoSizing(), 0.9)
    decks["charge_pump"] = pll.bench_cp("charge_pump", tt("tt"), pll.CpSizing())
    for name, text in decks.items():
        (OUT / f"{name}.cir").write_text(text)
        print("wrote", OUT / f"{name}.cir")


if __name__ == "__main__":
    main()
