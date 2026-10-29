"""Thin ngspice driver.

Netlists are generated as text by the circuit modules, written into a
scratch batch directory together with the model library, and executed in
batch mode. ngspice is used natively if it is on PATH, otherwise the
``devtools`` docker image is used. Each batch runs in a single container so
that Monte Carlo sweeps do not pay container start-up cost per sample.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_LIB = REPO_ROOT / "models" / "cmos180_approx.lib"
DOCKER_IMAGE = os.environ.get("ANALOGIC_DOCKER_IMAGE", "devtools:latest")

_MEAS_RE = re.compile(r"^\s*([A-Za-z_][\w\[\]@.#]*)\s*=\s*([-+]?\d[\d.]*(?:[eE][-+]?\d+)?)")


class SpiceError(RuntimeError):
    pass


@dataclass
class SimResult:
    name: str
    log: str
    workdir: Path
    scalars: dict[str, float] = field(default_factory=dict)
    tables: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def table(self, fname: str) -> dict[str, np.ndarray]:
        """Return a ``wrdata`` table (written with wr_vecnames/wr_singlescale)."""
        try:
            return self.tables[fname]
        except KeyError:
            raise SpiceError(f"{self.name}: no output table {fname}\n{self.log[-1500:]}") from None


def read_wrdata(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        header = fh.readline().split()
        rows = [ln.split() for ln in fh if ln.strip()]
    data = np.array(rows, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise SpiceError(f"cannot match columns in {path}: {header} vs {data.shape}")
    # Complex vectors are written as two adjacent columns with the same name.
    out: dict[str, np.ndarray] = {}
    i = 0
    while i < len(header):
        nm = header[i]
        if i + 1 < len(header) and header[i + 1] == nm:
            out[nm] = data[:, i] + 1j * data[:, i + 1]
            i += 2
        else:
            out[nm] = data[:, i]
            i += 1
    return out


def parse_scalars(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in log.splitlines():
        m = _MEAS_RE.match(line)
        if m:
            out[m.group(1).lower()] = float(m.group(2))
    return out


def _have_native() -> bool:
    return shutil.which("ngspice") is not None


def _have_docker() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def ngspice_available() -> bool:
    return _have_native() or _have_docker()


PREAMBLE = """.include cmos180_approx.lib
.options savecurrents=0
"""

CONTROL_SETTINGS = "set wr_vecnames\nset wr_singlescale\noption numdgt=10\n"


class NgSpice:
    """Run batches of netlists and collect results."""

    def __init__(self, workroot: Path | None = None, parallel: int = 2, timeout: int = 1800):
        self.workroot = Path(workroot or REPO_ROOT / "sim_work")
        self.workroot.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.timeout = timeout
        if _have_native():
            self.mode = "native"
        elif _have_docker():
            self.mode = "docker"
        else:
            raise SpiceError("neither ngspice nor the devtools docker image is available")

    def run(self, jobs: dict[str, str], keep: bool = False) -> dict[str, SimResult]:
        batch = Path(tempfile.mkdtemp(prefix="batch_", dir=self.workroot))
        shutil.copy(MODEL_LIB, batch / MODEL_LIB.name)
        for name, text in jobs.items():
            if not re.fullmatch(r"[\w.-]+", name):
                raise ValueError(f"bad job name {name!r}")
            (batch / f"{name}.cir").write_text(text)
        script = (
            f"ls *.cir | xargs -P {self.parallel} -I{{}} sh -c "
            "'ngspice -b {} > $(basename {} .cir).log 2>&1'"
        )
        if self.mode == "native":
            cmd = ["sh", "-c", script]
            cwd = batch
        else:
            cmd = ["docker", "run", "--rm", "--name", f"analogic-{uuid.uuid4().hex[:8]}",
                   "--memory", "1g",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{batch}:/work", "-w", "/work", DOCKER_IMAGE, "sh", "-c", script]
            cwd = None
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise SpiceError(f"ngspice batch failed: {proc.stderr[-2000:]}")
        results = {}
        for name in jobs:
            log_path = batch / f"{name}.log"
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            if "error" in log.lower() and "simulation interrupted" in log.lower():
                raise SpiceError(f"{name}: simulation error\n{log[-1500:]}")
            tables = {p.name: read_wrdata(p) for p in batch.glob(f"{name}_*.dat")}
            results[name] = SimResult(name, log, batch, parse_scalars(log), tables)
        if not keep:
            shutil.rmtree(batch, ignore_errors=True)
        return results

    def run_one(self, name: str, text: str) -> SimResult:
        return self.run({name: text})[name]


def wrap(title: str, body: str, control: str) -> str:
    """Assemble a complete netlist with the common preamble and a control block."""
    return (f"* {title}\n{PREAMBLE}\n{body}\n.control\n{CONTROL_SETTINGS}"
            f"{control}\n.endc\n.end\n")
