"""Result writing and plotting helpers shared by the experiment scripts."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .spice import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
PLOTS = RESULTS / "plots"

# Fixed categorical order (slot 1..7); series keep their slot across plots.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def style_axes(ax, xlabel: str, ylabel: str, title: str | None = None):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def new_fig(nrows=1, ncols=1, w=5.0, h=3.2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=90)
    return fig, axes


def save(fig, name: str):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=90)
    plt.close(fig)


def write_csv(name: str, rows: list[dict]):
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(RESULTS / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "nan"
        return f"{v:.6g}"
    return v


def md_table(rows: list[dict], cols: list[str] | None = None, digits: int = 4) -> str:
    if not rows:
        return ""
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (float, np.floating)):
                v = "nan" if math.isnan(v) else f"{v:.{digits}g}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def stats(values) -> dict:
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def write_md(name: str, text: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(text)
