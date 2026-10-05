#!/usr/bin/env python3
"""Parallel-generation benchmark: does running K jobs at once finish a batch sooner?

For each (size, K) it renders the same batch of 4 prompts (fixed seeds) with K jobs in flight,
straight through run.sh (so ltx_job.py records per-job time and peak GPU memory), and samples
system memory, swap and GPU utilization every second. Also times one image-to-video job.

Every job start is memory-gated like the studio (MemAvailable minus the not-yet-allocated peak of
all running jobs must stay >= 50 GiB), so it is safe to run while users keep generating. Each row
records how many other (non-benchmark) jobs overlapped it: rows with "contended": true were measured
on a shared GPU. For clean numbers, pause the studio first (echo "Benchmark" > logs/webui_pause).
Writes benchmarks/parallel_results.json; plot with make_parallel_chart.py.
"""
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "bench"  # subfolder: kept out of the studio gallery
HISTORY = ROOT / "logs" / "jobs.jsonl"
PROMPTS = [
    "A red fox trots through fresh snow in a quiet birch forest at golden hour, soft breath vapor, gentle wind sound, cinematic tracking shot",
    "Ocean waves crash against black volcanic rocks at sunset, sea spray glowing orange, seagulls calling, slow aerial drone shot",
    "A steaming cup of coffee on a wooden cafe table by a rainy window, raindrops sliding down the glass, soft jazz and rain sounds, slow push-in",
    "Northern lights ripple over a frozen lake at night, stars overhead, ice crackling softly, slow pan across the horizon",
]
CONFIGS = [((768, 512), 1), ((768, 512), 2), ((768, 512), 3), ((1280, 704), 1), ((1280, 704), 2)]
FRAMES = 97  # 4 s at 24 fps
ENV = os.environ | {"LTX_JOB_SOURCE": "bench", "LTX_JOB_USER": "benchmark"}
RUNNING = ROOT / "logs" / "running"
MIN_FREE = 50 * 2**30
JOB_PEAK = 27 * 2**30  # measured per-process peak 22.5-23.4 GiB (8 s at 1280x704: 26 GiB), + margin
start_lock = threading.Lock()
starting = set()  # outputs launched whose state file hasn't appeared yet (reserved at full peak)


def running_jobs():
    jobs = []
    for f in RUNNING.glob("*.json"):
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if j.get("running") and Path(f"/proc/{j.get('pid')}").exists():
            jobs.append(j)
    return jobs


def effective_free():
    live = running_jobs()
    seen = {j.get("output") for j in live}
    starting.difference_update(seen)
    pending = sum(max(0, JOB_PEAK - j.get("gpu_mem_bytes", 0)) for j in live) + JOB_PEAK * len(starting)
    return Sampler.meminfo()["MemAvailable"] - pending


class Sampler(threading.Thread):
    """1 Hz system sampling: MemAvailable, swap traffic, GPU utilization."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.min_avail, self.util, self.swap_kb = 1 << 62, [], 0
        self.other_s, self.max_other, self.max_bench = 0, 0, 0

    @staticmethod
    def meminfo():
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            d[k] = int(v.split()[0]) * 1024
        return d

    @staticmethod
    def pswp():
        v = {k: int(n) for k, n in (l.split() for l in open("/proc/vmstat") if l.startswith(("pswpin", "pswpout")))}
        return v["pswpin"] + v["pswpout"]

    def run(self):
        sw0 = self.pswp()
        while not self.stop.is_set():
            self.min_avail = min(self.min_avail, self.meminfo()["MemAvailable"])
            live = running_jobs()
            other = sum(j.get("source") != "bench" for j in live)
            self.other_s += other > 0
            self.max_other = max(self.max_other, other)
            self.max_bench = max(self.max_bench, sum(j.get("source") == "bench" for j in live))
            try:
                u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
                self.util.append(float(u.splitlines()[0]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            time.sleep(1)
        self.swap_kb = (self.pswp() - sw0) * 4  # pages -> KiB


def render(prompt, out, w, h, seed, image=None):
    cmd = [str(ROOT / "run.sh"), prompt, str(out), "--width", str(w), "--height", str(h),
           "--num-frames", str(FRAMES), "--seed", str(seed)]
    if image:
        cmd += ["--image", str(image), "0", "1.0"]
    with start_lock:  # one start at a time, and only with memory to spare
        while effective_free() < MIN_FREE:
            time.sleep(3)
        starting.add(str(out))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
    return {"out": str(out), "rc": rc, "wall_s": round(time.time() - t0, 2)}


def records_for(outs):
    by_out = {}
    for line in open(HISTORY):
        r = json.loads(line)
        if r.get("output") in outs:
            by_out[r["output"]] = r
    return by_out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for (w, h), k in CONFIGS:
        tag = f"{w}x{h}-k{k}"
        print(f"== {tag}: {len(PROMPTS)} jobs, {k} at a time", flush=True)
        samp = Sampler()
        samp.start()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=k) as ex:
            futs = [ex.submit(render, p, OUT / f"{tag}-{i}.mp4", w, h, 100 + i) for i, p in enumerate(PROMPTS)]
            runs = [f.result() for f in futs]
        makespan = time.time() - t0
        samp.stop.set()
        samp.join()
        recs = records_for({r["out"] for r in runs})
        row = {
            "size": f"{w}x{h}", "workers": k, "jobs": len(PROMPTS), "frames": FRAMES,
            "makespan_s": round(makespan, 1),
            "throughput_jobs_per_min": round(len(PROMPTS) / makespan * 60, 2),
            "avg_job_s": round(sum(r["wall_s"] for r in runs) / len(runs), 1),
            "max_job_s": max(r["wall_s"] for r in runs),
            "failed": sum(r["rc"] != 0 for r in runs),
            "peak_gpu_mem_gib_per_job": round(max((recs[r["out"]]["peak_gpu_mem_bytes"] for r in runs if r["out"] in recs), default=0) / 2**30, 1),
            "min_mem_available_gib": round(samp.min_avail / 2**30, 1),
            "swap_traffic_mib": round(samp.swap_kb / 1024, 1),
            "gpu_util_avg_pct": round(sum(samp.util) / len(samp.util), 1) if samp.util else None,
            "max_bench_concurrent": samp.max_bench,
            "other_jobs_seconds": samp.other_s, "max_other_jobs": samp.max_other,
            "contended": samp.other_s > 0,
        }
        print("  ", row, flush=True)
        results.append(row)

    # Image-to-video sanity run (same size/length as the 768x512 serial text jobs).
    img = OUT / "i2v-input.png"
    print("== image-to-video: 1 job, 768x512", flush=True)
    r = render("Waves roll in and burst against the rocks, spray catching the low sun, seagulls and surf",
               OUT / "i2v-768x512.mp4", 768, 512, 7, image=img)
    rec = records_for({r["out"]}).get(r["out"], {})
    i2v = {"size": "768x512", "frames": FRAMES, "rc": r["rc"], "wall_s": r["wall_s"], "mode": rec.get("mode"),
           "peak_gpu_mem_gib": round(rec.get("peak_gpu_mem_bytes", 0) / 2**30, 1)}
    print("  ", i2v, flush=True)

    out = Path(__file__).parent / "parallel_results.json"
    out.write_text(json.dumps({"gpu": "NVIDIA GB10 (DGX Spark)", "pipeline": "ltx_pipelines.distilled, nvfp4-prequant",
                               "note": "makespan = wall time for the whole batch; jobs include model loading",
                               "results": results, "image_to_video": i2v}, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
