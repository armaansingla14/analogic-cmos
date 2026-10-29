"""Process corners, mismatch and device-line generation.

Global process corners are applied as threshold shifts (``delvto``) and
mobility multipliers (``mulu0``) on each instance; local mismatch follows a
Pelgrom model (sigma_Vt = A_VT / sqrt(W L), sigma_beta/beta = A_beta / sqrt(W L)).
All coefficients are approximate textbook-scale values for a 180 nm-class
process and are not foundry data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Pelgrom coefficients (V*um and fraction*um)
AVT = {"n": 4.5e-3, "p": 5.0e-3}
ABETA = {"n": 0.010, "p": 0.012}
# Resistor matching: sigma(dR/R) = AR / sqrt(W*L) with W,L in um
AR_POLY = 0.02


@dataclass(frozen=True)
class Corner:
    name: str
    dvt_n: float  # +V slows NMOS
    dvt_p: float  # +V slows PMOS (raises |Vt|)
    mu_n: float
    mu_p: float
    r_scale: float = 1.0


CORNERS: dict[str, Corner] = {
    "tt": Corner("tt", 0.0, 0.0, 1.0, 1.0),
    "ff": Corner("ff", -0.045, -0.045, 1.08, 1.08, 0.9),
    "ss": Corner("ss", 0.045, 0.045, 0.92, 0.92, 1.1),
    "fs": Corner("fs", -0.045, 0.045, 1.08, 0.92),
    "sf": Corner("sf", 0.045, -0.045, 0.92, 1.08),
}


@dataclass
class Process:
    """Holds a global corner plus an optional mismatch random generator."""

    corner: Corner = field(default_factory=lambda: CORNERS["tt"])
    rng: np.random.Generator | None = None
    mismatch_scale: float = 1.0
    record: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def at(cls, corner: str = "tt", seed: int | None = None, mismatch: bool = False) -> "Process":
        rng = np.random.default_rng(seed) if mismatch else None
        return cls(CORNERS[corner], rng)

    def _mismatch(self, kind: str, w: float, l: float, m: int) -> tuple[float, float]:
        if self.rng is None:
            return 0.0, 1.0
        area = w * l * 1e12 * m  # um^2
        s_vt = AVT[kind] / np.sqrt(area) * self.mismatch_scale
        s_b = ABETA[kind] / np.sqrt(area) * self.mismatch_scale
        return float(self.rng.normal(0, s_vt)), float(1 + self.rng.normal(0, s_b))

    def mos(self, name: str, d: str, g: str, s: str, b: str, kind: str,
            w: float, l: float, m: int = 1) -> str:
        """Return an ngspice MOSFET instance line with corner and mismatch applied."""
        if kind not in ("n", "p"):
            raise ValueError(kind)
        c = self.corner
        dvt, mu = (c.dvt_n, c.mu_n) if kind == "n" else (c.dvt_p, c.mu_p)
        dv_mm, mu_mm = self._mismatch(kind, w, l, m)
        dvt_total = dvt + dv_mm
        mu_total = mu * mu_mm
        self.record[name] = (dv_mm, mu_mm)
        # ngspice delvto is added to vth0 in model sign; a PMOS is slowed by a
        # negative shift.
        delvto = dvt_total if kind == "n" else -dvt_total
        model = "nch" if kind == "n" else "pch"
        return (f"{name} {d} {g} {s} {b} {model} w={w:.4g} l={l:.4g} m={m} "
                f"delvto={delvto:.6g} mulu0={mu_total:.6g}")

    def res(self, name: str, a: str, b: str, value: float, w_um: float = 2.0,
            l_um: float | None = None, tc: bool = True) -> str:
        """Poly resistor with corner scaling, mismatch and temperature coefficients."""
        val = value * self.corner.r_scale
        if self.rng is not None:
            l_um = l_um if l_um is not None else max(w_um, value / 500.0 * w_um)
            val *= 1 + self.rng.normal(0, AR_POLY / np.sqrt(w_um * l_um)) * self.mismatch_scale
        tcs = " tc1=-1.2e-3 tc2=1.0e-6" if tc else ""
        return f"{name} {a} {b} {val:.6g}{tcs}"


def pvt_grid(corners=("tt", "ff", "ss", "fs", "sf"), temps=(-40.0, 27.0, 125.0),
             vdd_nom: float = 1.8, vdd_tol: float = 0.1):
    """Yield (corner, temp, vdd) tuples for a PVT sweep."""
    for c in corners:
        for t in temps:
            for v in (vdd_nom * (1 - vdd_tol), vdd_nom, vdd_nom * (1 + vdd_tol)):
                yield c, t, round(v, 4)
