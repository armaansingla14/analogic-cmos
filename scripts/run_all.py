"""Run every characterization script in sequence (OTA, bandgap, DAC, PLL)."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("run_ota.py", "run_bandgap.py", "run_dac.py", "run_pll.py", "export_netlists.py"):
    t0 = time.time()
    print(f"== {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)
    print(f"== {script} done in {time.time() - t0:.0f} s", flush=True)
