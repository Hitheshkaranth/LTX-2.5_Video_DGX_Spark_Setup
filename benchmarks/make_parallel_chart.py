#!/usr/bin/env python3
"""Render benchmarks/parallel_results.json into a README chart (run with LTX-2/.venv/bin/python)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
data = json.loads((HERE / "parallel_results.json").read_text())
rows = data["results"]

BG, FG, GRID = "#111217", "#d8d9da", "#2a2d35"
WCOL = {1: "#5794F2", 2: "#73BF69", 3: "#FF9830"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 11,
                     "axes.titleweight": "bold", "axes.titlesize": 13})
sizes = list(dict.fromkeys(r["size"] for r in rows))
fig, axes = plt.subplots(1, 3, figsize=(17, 5))


def grouped(ax, key, title, ylabel, fmt):
    width = 0.26
    for si, size in enumerate(sizes):
        group = [r for r in rows if r["size"] == size]
        for gi, r in enumerate(group):
            x = si + (gi - (len(group) - 1) / 2) * width
            b = ax.bar(x, r[key], width * 0.92, color=WCOL[r["workers"]],
                       label=f"{r['workers']} at a time" if si == 0 else None)
            ax.text(x, r[key], fmt(r[key]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(sizes)), [f"{s.replace('x', '×')}\n4 s clips" for s in sizes])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


grouped(axes[0], "makespan_s", "Time to finish 4 clips", "seconds (lower is better)", lambda v: f"{v:.0f}s")
grouped(axes[1], "throughput_jobs_per_min", "Throughput", "clips per minute (higher is better)", lambda v: f"{v:.2f}")
grouped(axes[2], "min_mem_available_gib", "Lowest free memory during the batch", "GiB available", lambda v: f"{v:.0f}")
axes[2].axhline(20, color="#F2495C", ls="--", lw=1.2)
axes[2].text(axes[2].get_xlim()[1], 20, " swap risk ", color="#F2495C", va="bottom", ha="right", fontsize=9)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Parallel generation on one DGX Spark (LTX-2.5 NVFP4, 4-clip batch)", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(HERE / "parallel_benchmark.png", dpi=130)
print("wrote", HERE / "parallel_benchmark.png")
