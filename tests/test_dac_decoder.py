import pytest

from analogic.dac import DacSizing, _pwl, cell_states, cell_weights, sine_codes


@pytest.mark.parametrize("arch", ["binary", "segmented"])
def test_decoder_reproduces_code(arch):
    s = DacSizing(arch=arch)
    w = cell_weights(s)
    assert sum(w) == 255
    for code in range(256):
        st = cell_states(s, code)
        assert sum(a * b for a, b in zip(w, st)) == code


def test_thermometer_cells_switch_one_at_a_time():
    s = DacSizing(arch="segmented")
    prev = cell_states(s, 0)[4:]
    for code in range(16, 256, 16):
        cur = cell_states(s, code)[4:]
        assert sum(c - p for c, p in zip(cur, prev)) == 1
        assert all(c >= p for c, p in zip(cur, prev))
        prev = cur


def test_pwl_only_emits_transitions():
    txt = _pwl([0, 0, 1, 1, 0], 10e-9, 1e-10, 1.8)
    vals = txt.split()
    assert len(vals) == 2 + 4 * 2


def test_sine_codes_range():
    c = sine_codes(256, 7)
    assert min(c) >= 0 and max(c) <= 255 and len(c) == 256
