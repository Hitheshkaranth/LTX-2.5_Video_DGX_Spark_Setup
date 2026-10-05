#!/usr/bin/env python3
"""Run one LTX pipeline invocation and record it for the Grafana exporter.

Usage: ltx_job.py <output.mp4> -- <pipeline command...>

While the job runs, logs/running/<job_id>.json is rewritten every second with the
live stage, denoising progress and GPU memory of the pipeline process; several
jobs can run at once, each with its own file. When a job ends its file is removed
and one JSON line is appended to logs/jobs.jsonl. ltx_exporter.py and the web UI
read both. Pipeline output is still streamed to the terminal and also saved to
logs/jobs/<job_id>.log.
"""
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
RUNNING = LOGS / "running"  # one <job_id>.json per in-flight job
LEGACY = LOGS / "current.json"  # single-job file read by older web UI / exporter versions
HISTORY = LOGS / "jobs.jsonl"

# Log line markers from ltx_pipelines.utils.blocks, in pipeline order.
STAGES = [
    ("text_encoder", "Building text encoder"),
    ("text_encoder", "Prompt encoding"),
    ("denoise_stage1", "Running denoising loop (8 steps"),
    ("upsample", "spatial upsampler"),
    ("denoise_stage2", "Running denoising loop (3 steps"),
    ("decode", "Building video decoder"),
    ("decode", "Building audio decoder"),
]
TQDM = re.compile(r"(\d+)/(\d+) \[")


def arg(cmd, name, default=None):
    return cmd[cmd.index(name) + 1] if name in cmd else default


def gpu_mem_bytes(pid):
    """Unified-memory GPU allocation of pid, via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    total = 0
    for line in out.strip().splitlines():
        p, mib = (x.strip() for x in line.split(","))
        if p == str(pid):
            total += int(mib) * 2**20
    return total


def write_json(path, obj):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # per-process: parallel jobs share LEGACY
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def main():
    sep = sys.argv.index("--")
    output, cmd = sys.argv[1], sys.argv[sep + 1:]
    # pid suffix: parallel jobs can start in the same second
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    (LOGS / "jobs").mkdir(parents=True, exist_ok=True)
    RUNNING.mkdir(parents=True, exist_ok=True)
    live_file = RUNNING / f"{job_id}.json"
    width, height = int(arg(cmd, "--width", 1536)), int(arg(cmd, "--height", 1024))
    frames, fps = int(arg(cmd, "--num-frames", 121)), float(arg(cmd, "--frame-rate", 24))
    state = {
        "job_id": job_id, "running": 1, "stage": "starting", "step": 0, "steps": 0,
        "start": time.time(), "pid": None, "gpu_mem_bytes": 0, "peak_gpu_mem_bytes": 0,
        "width": width, "height": height, "frames": frames, "fps": fps, "output": output,
        "prompt": arg(cmd, "--prompt", ""),
        # --image PATH FRAME STRENGTH turns the run into image-to-video.
        "image": Path(arg(cmd, "--image")).name if "--image" in cmd else "",
        "image_path": str(Path(arg(cmd, "--image")).resolve()) if "--image" in cmd else "",
        "mode": "i2v" if "--image" in cmd else "t2v",
        # Set by webui/server.py; plain CLI runs are attributed to the shell user.
        "user": os.environ.get("LTX_JOB_USER") or os.environ.get("USER", "unknown"),
        "source": os.environ.get("LTX_JOB_SOURCE", "cli"),
    }
    lock = threading.Lock()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    state["pid"] = proc.pid
    write_json(live_file, state)
    write_json(LEGACY, state)

    def stop(signum, _frame):
        # Killed (benchmark abort, Ctrl-C): take the render down too and leave no stale state file,
        # otherwise an orphaned pipeline keeps holding ~23 GiB with nothing tracking it.
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        live_file.unlink(missing_ok=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def sample():
        while proc.poll() is None:
            mem = gpu_mem_bytes(proc.pid)
            with lock:
                state["gpu_mem_bytes"] = mem
                state["peak_gpu_mem_bytes"] = max(state["peak_gpu_mem_bytes"], mem)
                if state["running"]:
                    write_json(live_file, state)
                    write_json(LEGACY, state)
            time.sleep(1)

    threading.Thread(target=sample, daemon=True).start()

    with open(LOGS / "jobs" / f"{job_id}.log", "wb") as log:
        buf = b""
        while chunk := proc.stdout.read(4096):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            buf = (buf + chunk)[-4096:]
            text = buf.decode("utf-8", "replace")
            with lock:
                for stage, marker in STAGES:
                    if marker in text:
                        if stage != state["stage"]:
                            state["step"], state["steps"] = 0, 0
                        state["stage"] = stage
                # tqdm redraws with \r; the last match is the current step.
                if state["stage"].startswith("denoise"):
                    m = TQDM.findall(text.split("Running denoising loop")[-1])
                    if m:
                        state["step"], state["steps"] = int(m[-1][0]), int(m[-1][1])
    rc = proc.wait()

    end = time.time()
    with lock:
        ok = rc == 0 and Path(output).is_file()
        record = {
            "job_id": job_id, "status": "ok" if ok else "failed", "exit_code": rc,
            "start": state["start"], "end": end, "duration_s": round(end - state["start"], 2),
            "peak_gpu_mem_bytes": state["peak_gpu_mem_bytes"], "width": width, "height": height,
            "frames": frames, "fps": fps, "video_seconds": round(frames / fps, 3) if ok else 0,
            "output": output, "prompt": state["prompt"], "user": state["user"], "source": state["source"],
            "image": state["image"], "image_path": state["image_path"], "mode": state["mode"],
        }
        with open(HISTORY, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # parallel jobs append to the same history
            f.write(json.dumps(record) + "\n")
        state["running"] = 0
        live_file.unlink(missing_ok=True)
        if not any(RUNNING.glob("*.json")):
            write_json(LEGACY, state)
    sys.exit(rc)


if __name__ == "__main__":
    main()
