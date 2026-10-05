#!/usr/bin/env python3
"""Sweep LTX-2.5 generation time / peak memory across sizes and lengths.

Jobs go through the web UI queue (webui/server.py) so they never overlap with other users' jobs;
timings and peak GPU memory are read back from logs/jobs.jsonl as recorded by ltx_job.py.
Writes benchmarks/results.json. Run on an otherwise idle box (no LLM server holding memory) with the
studio at LTX_WORKERS=1; with parallel workers the timed jobs would overlap each other.
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8090/api"
PROMPT = ("A red fox trots through fresh snow in a quiet birch forest at golden hour, soft breath vapor, "
          "gentle wind sound, cinematic tracking shot")
SWEEP = [("768x512", 49), ("768x512", 97), ("768x512", 121), ("768x512", 193),
         ("1024x576", 97), ("1280x704", 97), ("1536x1024", 49), ("1536x1024", 97)]


def call(path, body=None):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def main():
    results = []
    for size, frames in SWEEP:
        job = call("generate", {"prompt": PROMPT, "size": size, "frames": frames, "seed": 42})
        print(f"{size} x {frames} frames: queued {job['id']}", flush=True)
        while True:
            j = next(x for x in call("status")["jobs"] if x["id"] == job["id"])
            if j["status"] in ("done", "failed"):
                break
            time.sleep(3)
        rec = None
        for line in open(ROOT / "logs" / "jobs.jsonl"):
            r = json.loads(line)
            if Path(r["output"]).name == job["file"]:
                rec = r
        w, h = map(int, size.split("x"))
        row = {"size": size, "width": w, "height": h, "frames": frames, "video_seconds": round(frames / 24, 2),
               "status": j["status"], "duration_s": rec and rec["duration_s"],
               "peak_gpu_mem_gib": rec and round(rec["peak_gpu_mem_bytes"] / 2**30, 1),
               "error": j.get("error")}
        if rec:
            row["seconds_per_video_second"] = round(rec["duration_s"] / (frames / 24), 1)
        print("   ", row, flush=True)
        results.append(row)
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({"prompt": PROMPT, "seed": 42, "fps": 24, "gpu": "NVIDIA GB10 (DGX Spark)",
                               "pipeline": "ltx_pipelines.distilled, nvfp4-prequant, 8+3 steps",
                               "results": results}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
