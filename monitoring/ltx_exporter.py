#!/usr/bin/env python3
"""Prometheus exporter for LTX-2.5 video generation jobs.

LTX runs as a one-shot CLI with no metrics endpoint, so run.sh goes through
ltx_job.py, which keeps logs/running/<job_id>.json per in-flight job (several can
run in parallel) and appends logs/jobs.jsonl (finished jobs). This re-exposes both as Prometheus gauges for
the "LTX-2.5 Video Generation" Grafana dashboard.
"""
import http.server
import json
import os
import time
from pathlib import Path

LTX_LOGS = Path(os.environ.get("LTX_LOGS", Path(__file__).resolve().parent.parent / "logs"))
LISTEN_PORT = int(os.environ.get("EXPORTER_PORT", "9092"))
# 0.0.0.0 so Prometheus in Docker can reach it via host.docker.internal. Metrics include requester
# logins: firewall :9092 from your LAN, or bind to the Docker bridge IP (e.g. 172.17.0.1).
LISTEN_HOST = os.environ.get("EXPORTER_HOST", "0.0.0.0")
STAGES = ["starting", "text_encoder", "denoise_stage1", "upsample", "denoise_stage2", "decode"]


def load_running():
    """In-flight jobs, oldest first. A killed job never removes its file, so trust the pid."""
    jobs = []
    for f in (LTX_LOGS / "running").glob("*.json"):
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if j.get("running") and Path(f"/proc/{j.get('pid')}").exists():
            jobs.append(j)
    if not jobs:  # job started by an older ltx_job.py that only wrote current.json
        try:
            j = json.loads((LTX_LOGS / "current.json").read_text())
            if j.get("running") and Path(f"/proc/{j.get('pid')}").exists():
                jobs.append(j)
        except (OSError, ValueError):
            pass
    return sorted(jobs, key=lambda j: j.get("start", 0))


def load_webui():
    try:
        return json.loads((LTX_LOGS / "webui_state.json").read_text())
    except (OSError, ValueError):
        return {}


def load_history():
    jobs = []
    try:
        with open(LTX_LOGS / "jobs.jsonl") as f:
            for line in f:
                try:
                    jobs.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return jobs


def render_metrics():
    live, jobs = load_running(), load_history()
    running = bool(live)
    cur = live[0] if live else {}  # oldest running job drives the single-value panels
    out = []

    def metric(name, help_, value, labels=None, kind="gauge"):
        if not any(line.startswith(f"# HELP {name} ") for line in out):
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {kind}")
        esc = lambda v: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        lbl = "{" + ",".join(f'{k}="{esc(v)}"' for k, v in labels.items()) + "}" if labels else ""
        out.append(f"{name}{lbl} {value}")

    metric("ltx_job_running", "Number of LTX jobs running right now (parallel workers).", len(live))
    for s in STAGES:
        metric("ltx_job_stage", "Running jobs currently in each pipeline stage.",
               sum(j.get("stage") == s for j in live), {"stage": s})
    steps = cur.get("steps", 0) if running else 0
    metric("ltx_job_step", "Current denoising step of the running job.", cur.get("step", 0) if running else 0)
    metric("ltx_job_steps", "Denoising steps in the current stage.", steps)
    metric("ltx_job_elapsed_seconds", "Wall time of the oldest running job.",
           round(time.time() - cur["start"], 1) if running else 0)
    metric("ltx_job_gpu_memory_bytes", "Unified GPU memory held by all running LTX processes.",
           sum(j.get("gpu_mem_bytes", 0) for j in live))
    metric("ltx_job_pixels", "Output width*height of the running job.",
           cur["width"] * cur["height"] if running else 0)
    metric("ltx_job_frames", "Output frame count of the running job.", cur["frames"] if running else 0)

    for status in ("ok", "failed"):
        metric("ltx_jobs_total", "Finished LTX jobs.", sum(j["status"] == status for j in jobs),
               {"status": status}, "counter")
    ok = [j for j in jobs if j["status"] == "ok"]
    metric("ltx_video_seconds_generated_total", "Seconds of video produced by successful jobs.",
           round(sum(j["video_seconds"] for j in ok), 3), kind="counter")
    metric("ltx_job_duration_seconds_sum", "Total wall time of successful jobs.",
           round(sum(j["duration_s"] for j in ok), 2), kind="counter")
    metric("ltx_job_duration_seconds_count", "Number of successful jobs.", len(ok), kind="counter")
    by_user = {}
    for j in jobs:
        key = (j.get("user", "unknown"), j.get("source", "cli"), j["status"])
        by_user[key] = by_user.get(key, 0) + 1
    for (user, source, status), n in sorted(by_user.items()):
        metric("ltx_jobs_by_user_total", "Finished LTX jobs by requester and entry point (cli/webui).", n,
               {"user": user, "source": source, "status": status}, "counter")
    by_mode = {}
    for j in jobs:
        key = (j.get("mode", "t2v"), j["status"])
        by_mode[key] = by_mode.get(key, 0) + 1
    for mode in ("t2v", "i2v"):
        for status in ("ok", "failed"):
            metric("ltx_jobs_by_mode_total", "Finished jobs by mode: t2v (text-to-video) or i2v (image-to-video).",
                   by_mode.get((mode, status), 0), {"mode": mode, "status": status}, "counter")
    for mode in ("t2v", "i2v"):
        metric("ltx_job_mode", "Running jobs by mode.", sum(j.get("mode", "t2v") == mode for j in live), {"mode": mode})
    who = {}
    for j in live:
        k = (j.get("source", "cli"), j.get("user", ""))
        who[k] = who.get(k, 0) + 1
    for (source, user), n in sorted(who.items()) or [(("none", ""), 0)]:
        metric("ltx_job_running_source", "Running jobs by entry point (cli/webui) and requester.",
               n, {"source": source, "user": user})

    web = load_webui()
    up = bool(web) and time.time() - web.get("heartbeat", 0) < 30
    metric("ltx_webui_up", "1 if the LTX web UI heartbeat is fresh (<30s).", int(up))
    for state in ("queued", "running"):
        metric("ltx_webui_queue_jobs", "Web UI jobs by state.", web.get(state, 0) if up else 0, {"state": state})
    metric("ltx_webui_rejected_low_memory", "Web UI jobs refused for lack of memory since the UI started.",
           web.get("rejected_low_memory", 0) if up else 0)

    if jobs:
        last = jobs[-1]
        metric("ltx_last_job_success", "1 if the most recent job succeeded.", int(last["status"] == "ok"))
        metric("ltx_last_job_duration_seconds", "Wall time of the most recent job.", last["duration_s"])
        metric("ltx_last_job_peak_gpu_memory_bytes", "Peak GPU memory of the most recent job.",
               last["peak_gpu_mem_bytes"])
        metric("ltx_last_job_end_timestamp_seconds", "Unix time the most recent job finished.", last["end"])
        metric("ltx_last_job_video_seconds", "Video length of the most recent job.", last["video_seconds"])
        rt = last["duration_s"] / last["video_seconds"] if last["video_seconds"] else 0
        metric("ltx_last_job_realtime_factor", "Seconds of compute per second of video (last job).", round(rt, 2))
    return "\n".join(out) + "\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Prometheus scrapes every 5s


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    server.serve_forever()
