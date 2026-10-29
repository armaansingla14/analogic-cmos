"""End-to-end checks that run ngspice (skipped when it is unavailable)."""
import numpy as np
import pytest

from analogic import bandgap as bg
from analogic import ota
from analogic.models import Process
from analogic.spice import NgSpice, ngspice_available

pytestmark = [pytest.mark.spice,
              pytest.mark.skipif(not ngspice_available(), reason="ngspice/docker not available")]


@pytest.fixture(scope="module")
def sim():
    return NgSpice(parallel=2)


def test_ota_open_loop(sim):
    m = ota.characterize(sim, Process.at("tt"), ota.OtaSizing(), full=False)
    assert m["dc_gain_db"] > 60
    assert 1e6 < m["ugb_hz"] < 1e9
    assert 30 < m["pm_deg"] < 90
    assert abs(m["offset_mv"]) < 2


def test_bandgap_tt(sim):
    s = bg.BandgapSizing()
    r = bg.run_temp(sim, {"bg": (Process.at("tt"), s, 1.8)})["bg"]
    assert 1.1 < r["vref27"] < 1.35
    assert r["tc_ppm"] < 100
    assert abs(r["startup_vref"] - r["vref27"]) < 0.02


def test_dac_ramp_is_monotonic(sim):
    from analogic import dac
    s = dac.DacSizing(t_hold=10e-9)
    r = dac.run_linearity(sim, {"d": (Process.at("tt"), s)})["d"]
    assert r["endpoint"].monotonic
    assert r["endpoint"].lsb == pytest.approx(2 * s.rl * s.iunit, rel=0.05)
    assert np.isfinite(r["endpoint"].inl_max)
