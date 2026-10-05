#!/usr/bin/env python3
"""Render benchmarks/results.json into README charts (run with LTX-2/.venv/bin/python)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
data = json.loads((HERE / "results.json").read_text())
rows = [r for r in data["results"] if r["status"] == "done"]

BG, FG, GRID = "#111217", "#d8d9da", "#2a2d35"
COLORS = {"768x512": "#73BF69", "1024x576": "#5794F2", "1280x704": "#FF9830", "1536x1024": "#F2495C"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 11,
                     "axes.titleweight": "bold", "axes.titlesize": 13})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))

labels = [f"{r['size'].replace('x', '×')}\n{r['video_seconds']:.0f} s" for r in rows]
bars = a1.bar(labels, [r["duration_s"] for r in rows], color=[COLORS[r["size"]] for r in rows])
for b, r in zip(bars, rows):
    a1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['duration_s']:.0f}s", ha="center", va="bottom")
a1.set_title("Wall time per clip (text-to-video, incl. model load)")
a1.set_ylabel("seconds")
a1.grid(axis="y", color=GRID)
a1.set_axisbelow(True)
a1.tick_params(axis="x", labelsize=9)

bars = a2.bar(labels, [r["peak_gpu_mem_gib"] for r in rows], color=[COLORS[r["size"]] for r in rows])
for b, r in zip(bars, rows):
    a2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['peak_gpu_mem_gib']:.1f}", ha="center", va="bottom")
a2.set_title("Peak GPU memory of the LTX process (GiB)")
a2.set_ylabel("GiB")
a2.grid(axis="y", color=GRID)
a2.set_axisbelow(True)
a2.tick_params(axis="x", labelsize=9)

fig.suptitle("LTX-2.5 distilled · NVFP4 · DGX Spark (GB10)", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(HERE / "generation_benchmark.png", dpi=130)
print("wrote", HERE / "generation_benchmark.png")
