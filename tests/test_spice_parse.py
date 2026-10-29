import numpy as np

from analogic.spice import parse_scalars, read_wrdata


def test_read_wrdata_complex_columns(tmp_path):
    p = tmp_path / "x_ac.dat"
    p.write_text(" frequency h h \n 1.0 2.0 -1.0\n 10.0 0.5 0.25\n")
    t = read_wrdata(p)
    assert np.allclose(t["frequency"], [1, 10])
    assert np.allclose(t["h"], [2 - 1j, 0.5 + 0.25j])


def test_read_wrdata_real_columns(tmp_path):
    p = tmp_path / "x_tran.dat"
    p.write_text(" time v(out) v(inp) \n 0 0.1 0.2\n 1e-9 0.3 0.4\n")
    t = read_wrdata(p)
    assert set(t) == {"time", "v(out)", "v(inp)"}
    assert t["v(inp)"][1] == 0.4


def test_parse_scalars():
    log = "vos = -4.4690e-05\nidd = 2.2e-04\ninoise_total = 3.7e-04\nrandom text = here\n"
    s = parse_scalars(log)
    assert s["vos"] == -4.469e-05 and s["idd"] == 2.2e-4 and "random" not in s
