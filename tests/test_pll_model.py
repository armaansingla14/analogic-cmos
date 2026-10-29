import numpy as np
import pytest

from analogic.bandgap import VT27, BandgapSizing, first_order_r2
from analogic.pll import cp_mismatch_window, osc_freq
from analogic.pll_model import (BehavioralConfig, design_loop_filter, lock_time_from_edges,
                                loop_margins, simulate)


@pytest.mark.parametrize("pm", [45.0, 60.0, 70.0])
def test_filter_design_hits_targets(pm):
    fv = design_loop_filter(50e-6, 1e9, 16, 1e6, pm)
    fc, pm_meas = loop_margins(50e-6, 1e9, 16, fv)
    assert fc == pytest.approx(1e6, rel=0.01)
    assert pm_meas == pytest.approx(pm, abs=0.2)


def test_behavioral_pll_locks_with_linear_vco():
    v = np.linspace(0, 1.8, 19)
    f = 100e6 + 400e6 * v           # 400 MHz/V
    cpv = np.linspace(0, 1.8, 10)
    i = np.full_like(cpv, 50e-6)
    fv = design_loop_filter(50e-6, 400e6, 16, 1e6, 60)
    cfg = BehavioralConfig(t_stop=8e-6, v_init=0.5)
    r = simulate(cfg, fv, v, f, cpv, i, i)
    lk = lock_time_from_edges(r["ref_edges"], r["div_edges"], 16, 25e6)
    assert np.isfinite(lk["freq_lock_s"]) and lk["freq_lock_s"] < 6e-6
    assert r["vctrl"][-1] == pytest.approx(0.75, abs=0.01)  # 400 MHz -> 0.75 V


def test_osc_freq_on_sine():
    t = np.linspace(0, 1e-6, 200001)
    v = 0.9 + 0.9 * np.sin(2 * np.pi * 37e6 * t)
    assert osc_freq(t, v, 0.9) == pytest.approx(37e6, rel=1e-4)


def test_cp_window():
    v = np.linspace(0, 1.8, 91)
    iup = 50e-6 * np.clip((1.6 - v) / 0.2, 0, 1)
    idn = 50e-6 * np.clip(v / 0.2, 0, 1)
    lo, hi = cp_mismatch_window({"vout": v, "iup": iup, "idn": idn}, 0.05)
    assert 0.18 <= lo <= 0.22 and 1.38 <= hi <= 1.42


def test_bandgap_first_order_r2():
    s = BandgapSizing()
    r2 = first_order_r2(s, vbe_guess=0.65, vref_target=1.2)
    assert 0.65 + r2 / s.r1 * VT27 * np.log(8) == pytest.approx(1.2)
