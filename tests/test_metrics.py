import numpy as np
import pytest

from analogic.metrics import (ac_metrics, dac_linearity, lock_time, settling_time, sfdr_db,
                              slew_rate, tempco_ppm)


def two_pole(f, a0, p1, p2):
    s = 2j * np.pi * f
    return a0 / ((1 + s / (2 * np.pi * p1)) * (1 + s / (2 * np.pi * p2)))


def test_ac_metrics_two_pole_matches_analytic():
    f = np.logspace(0, 10, 4001)
    a0, p1, p2 = 1e4, 1e3, 20e6
    m = ac_metrics(f, two_pole(f, a0, p1, p2))
    assert m.dc_gain_db == pytest.approx(80.0, abs=0.01)
    assert m.f3db_hz == pytest.approx(p1, rel=0.01)
    # analytic unity-gain frequency of the two-pole system
    w = np.logspace(5, 9, 200001)
    mag = a0 / np.sqrt((1 + (w / p1) ** 2) * (1 + (w / p2) ** 2))
    fu = w[np.argmin(np.abs(mag - 1))]
    assert m.ugb_hz == pytest.approx(fu, rel=0.01)
    pm = 180 - np.degrees(np.arctan(fu / p1) + np.arctan(fu / p2))
    assert m.phase_margin_deg == pytest.approx(pm, abs=0.3)


def test_ac_metrics_inverting_reference_is_removed():
    f = np.logspace(0, 10, 2001)
    h = -two_pole(f, 1e3, 1e4, 1e8)
    m = ac_metrics(f, h)
    assert 80 < m.phase_margin_deg < 100


def test_slew_and_settling_on_synthetic_step():
    t = np.linspace(0, 1e-6, 10001)
    sr = 50e6  # V/s
    v = np.clip(0.6 + sr * np.clip(t - 1e-7, 0, None), 0.6, 1.2)
    assert slew_rate(t, v) == pytest.approx(sr, rel=1e-3)
    ts = settling_time(t, v, 1e-7, 0.01)
    assert ts == pytest.approx(0.6 * 0.99 / sr, rel=0.01)


def test_tempco_box_method():
    temp = np.linspace(-40, 125, 34)
    v = 1.2 * (1 + 10e-6 * (temp - 27))
    assert tempco_ppm(temp, v) == pytest.approx(10.0, rel=0.01)


def test_dac_linearity_ideal_and_known_error():
    ideal = np.arange(256) * 1e-3
    lin = dac_linearity(ideal)
    assert lin.inl_max < 1e-9 and lin.dnl_max < 1e-9 and lin.monotonic
    bad = ideal.copy()
    bad[128:] += 0.5e-3  # half-LSB step at mid-scale
    lin = dac_linearity(bad, "endpoint")
    assert lin.dnl_max == pytest.approx(0.498, abs=0.01)
    nonmono = ideal.copy()
    nonmono[100] = nonmono[99] - 1e-4
    assert not dac_linearity(nonmono).monotonic


def test_bestfit_inl_not_larger_than_endpoint_for_bow():
    codes = np.arange(256)
    y = codes + 2.0 * np.sin(np.pi * codes / 255)
    assert dac_linearity(y, "bestfit").inl_max < dac_linearity(y, "endpoint").inl_max


def test_lock_time():
    t = np.linspace(0, 10, 101)
    err = np.exp(-t)
    assert lock_time(t, err, 0.01) == pytest.approx(4.7, abs=0.11)


def test_sfdr_single_tone_with_spur():
    n = 1024
    k = np.arange(n)
    x = np.sin(2 * np.pi * 31 * k / n) + 1e-3 * np.sin(2 * np.pi * 93 * k / n)
    assert sfdr_db(x, 31) == pytest.approx(60.0, abs=0.1)
