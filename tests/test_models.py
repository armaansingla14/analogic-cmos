import re

import numpy as np

from analogic.models import CORNERS, Process, pvt_grid


def _param(line, name):
    return float(re.search(rf"{name}=([-\d.e+]+)", line).group(1))


def test_tt_has_no_shift():
    ln = Process.at("tt").mos("M1", "d", "g", "s", "b", "n", 1e-6, 1e-6)
    assert _param(ln, "delvto") == 0 and _param(ln, "mulu0") == 1


def test_slow_corner_sign_convention():
    p = Process.at("ss")
    n = p.mos("M1", "d", "g", "s", "b", "n", 1e-6, 1e-6)
    pm = p.mos("M2", "d", "g", "s", "b", "p", 1e-6, 1e-6)
    # slow NMOS: higher Vt -> positive delvto; slow PMOS: more negative Vt
    assert _param(n, "delvto") > 0
    assert _param(pm, "delvto") < 0
    assert _param(n, "mulu0") < 1


def test_mismatch_reproducible_and_area_scaled():
    a = Process.at("tt", seed=3, mismatch=True)
    b = Process.at("tt", seed=3, mismatch=True)
    la = a.mos("M1", "d", "g", "s", "b", "n", 1e-6, 1e-6)
    lb = b.mos("M1", "d", "g", "s", "b", "n", 1e-6, 1e-6)
    assert la == lb
    small, big = [], []
    p = Process.at("tt", seed=11, mismatch=True)
    for i in range(2000):
        small.append(_param(p.mos("Ma", "d", "g", "s", "b", "n", 1e-6, 1e-6), "delvto"))
        big.append(_param(p.mos("Mb", "d", "g", "s", "b", "n", 4e-6, 4e-6), "delvto"))
    ratio = np.std(small) / np.std(big)
    assert 3.6 < ratio < 4.4  # sigma ~ 1/sqrt(area)


def test_pvt_grid_size():
    assert len(list(pvt_grid())) == len(CORNERS) * 3 * 3
