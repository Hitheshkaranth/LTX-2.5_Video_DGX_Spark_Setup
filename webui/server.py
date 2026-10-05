#!/usr/bin/env python3
"""LTX-2.5 web UI: queue text-to-video jobs and browse/play the results.

Binds to 127.0.0.1 only; the tailnet reaches it through `tailscale serve`, which
also supplies the Tailscale-User-Login header used to tag who queued a job.
Jobs run one at a time through ../run.sh, so they show up in the Grafana
"LTX-2.5 Video Generation" dashboard like any CLI run.
"""
import http.server
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
CURRENT = ROOT / "logs" / "current.json"
WEBUI_STATE = ROOT / "logs" / "webui_state.json"  # read by ltx_exporter.py for Grafana
HISTORY = ROOT / "logs" / "jobs.jsonl"
INDEX = Path(__file__).resolve().parent / "index.html"
LISTEN = (os.environ.get("LTX_WEBUI_HOST", "127.0.0.1"), int(os.environ.get("LTX_WEBUI_PORT", "8090")))
# Optional second listener on the Tailscale IP itself (e.g. 100.x.y.z:8091) for tailnet
# clients whose MagicDNS is off; tailscale serve only routes by hostname, so a bare IP 404s there.
TS_LISTEN = os.environ.get("LTX_WEBUI_TS_LISTEN", "")
TAILSCALE = os.environ.get("TAILSCALE_BIN") or shutil.which("tailscale") or "tailscale"
# LLM container that shares unified memory with LTX; only used to explain a low-memory refusal.
LLM_CONTAINER = os.environ.get("LLM_CONTAINER", "vllm-ornith-a3b")
_whois_cache = {}

# Final output sizes; the distilled pipeline renders stage 1 at half size, so both
# dimensions must be multiples of 64.
SIZES = {"768x512", "512x768", "1024x576", "576x1024", "1280x704", "704x1280", "1536x1024"}
FRAMES = {49, 97, 121, 193}  # 2s, 4s, 5s, 8s at 24 fps (must be 8k+1)
# Observed peak ~29 GiB above idle; keep headroom. Override with LTX_MIN_FREE_GIB.
MIN_FREE_BYTES = int(os.environ.get("LTX_MIN_FREE_GIB", "40")) * 2**30
MAX_QUEUE = 10

lock = threading.Lock()
wake = threading.Condition(lock)
jobs = []  # web-submitted jobs, newest last


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def llm_running():
    try:
        out = subprocess.run(["docker", "ps", "--filter", f"name=^{LLM_CONTAINER}$", "--format", "{{.Status}}"],
                             capture_output=True, text=True, timeout=5).stdout
        return out.startswith("Up")
    except (OSError, subprocess.TimeoutExpired):
        return None


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-") or "clip"


def worker():
    while True:
        with lock:
            while not any(j["status"] == "queued" for j in jobs):
                wake.wait()
            job = next(j for j in jobs if j["status"] == "queued")
            free = mem_available()
            if free < MIN_FREE_BYTES:
                job.update(status="failed", finished=time.time(),
                           error=f"Not enough free memory ({free / 2**30:.0f} GiB free, need "
                                 f"{MIN_FREE_BYTES // 2**30}). Is an LLM server ({LLM_CONTAINER}) running?")
                continue
            job.update(status="running", started=time.time())
        w, h = job["size"].split("x")
        cmd = [str(ROOT / "run.sh"), job["prompt"], str(OUTPUTS / job["file"]),
               "--width", w, "--height", h, "--num-frames", str(job["frames"])]
        if job.get("seed") is not None:
            cmd += ["--seed", str(job["seed"])]
        log = ROOT / "logs" / "webui" / f"{job['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ | {"LTX_JOB_USER": job["user"], "LTX_JOB_SOURCE": "webui"}
        with open(log, "wb") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
        with lock:
            ok = rc == 0 and (OUTPUTS / job["file"]).is_file()
            job.update(status="done" if ok else "failed", finished=time.time())
            if not ok:
                tail = log.read_text(errors="replace").strip().splitlines()[-3:]
                job["error"] = f"exit {rc}: " + " | ".join(tail)[-400:]


def publish_state():
    """Heartbeat + queue snapshot for Grafana; the exporter treats a stale file as down."""
    while True:
        with lock:
            snap = {"heartbeat": time.time(),
                    "queued": sum(j["status"] == "queued" for j in jobs),
                    "running": sum(j["status"] == "running" for j in jobs),
                    "running_user": next((j["user"] for j in jobs if j["status"] == "running"), ""),
                    "rejected_low_memory": sum("Not enough free memory" in j.get("error", "") for j in jobs)}
        tmp = WEBUI_STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap))
        tmp.replace(WEBUI_STATE)
        time.sleep(2)


def live_progress():
    try:
        cur = json.loads(CURRENT.read_text())
    except (OSError, ValueError):
        return None
    if not cur.get("running"):
        return None
    return {k: cur.get(k) for k in ("stage", "step", "steps", "output")} | {
        "elapsed": round(time.time() - cur["start"], 1), "gpu_mem_gib": round(cur.get("gpu_mem_bytes", 0) / 2**30, 1)}


def history_by_file():
    out = {}
    try:
        for line in open(HISTORY):
            try:
                r = json.loads(line)
                out[Path(r["output"]).name] = r
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return out


def list_videos():
    hist = history_by_file()
    vids = []
    for p in sorted(OUTPUTS.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = hist.get(p.name, {})
        vids.append({"file": p.name, "size_mb": round(p.stat().st_size / 2**20, 1), "mtime": p.stat().st_mtime,
                     "prompt": r.get("prompt", ""), "duration_s": r.get("duration_s"),
                     "width": r.get("width"), "height": r.get("height"), "video_seconds": r.get("video_seconds")})
    return vids


class Handler(http.server.BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def user(self):
        if login := self.headers.get("Tailscale-User-Login"):
            return login
        ip = self.client_address[0]
        if ip.startswith("100."):  # direct hit on the Tailscale-IP listener
            if ip not in _whois_cache:
                try:
                    out = subprocess.run([TAILSCALE, "whois", "--json", ip], capture_output=True, text=True, timeout=5)
                    w = json.loads(out.stdout)
                    _whois_cache[ip] = w["UserProfile"]["LoginName"] + " @" + w["Node"]["Hostinfo"]["Hostname"]
                except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
                    return f"tailnet {ip}"
            return _whois_cache[ip]
        return "local"

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            with lock:
                q = [dict(j) for j in jobs[-30:]]
            free = mem_available()
            self.send_json({"jobs": q[::-1], "live": live_progress(), "mem_free_gib": round(free / 2**30, 1),
                            "can_run": free >= MIN_FREE_BYTES, "llm_running": llm_running(), "llm_container": LLM_CONTAINER,
                            "min_free_gib": MIN_FREE_BYTES // 2**30, "user": self.user()})
        elif path == "/api/videos":
            self.send_json(list_videos())
        elif path.startswith("/videos/"):
            self.send_video(unquote(path[len("/videos/"):]))
        else:
            self.send_error(404)

    def send_video(self, name):
        p = (OUTPUTS / name).resolve()
        if p.parent != OUTPUTS.resolve() or not p.is_file():
            self.send_error(404)
            return
        size = p.stat().st_size
        start, end = 0, size - 1
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if rng and (rng[1] or rng[2]):
            if rng[1]:
                start, end = int(rng[1]), int(rng[2]) if rng[2] else size - 1
            else:
                start = max(0, size - int(rng[2]))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0 and (chunk := f.read(min(1 << 20, left))):
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
            prompt = str(req["prompt"]).strip()
            size, frames = str(req["size"]), int(req["frames"])
            seed = int(req["seed"]) if str(req.get("seed", "")).strip() else None
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": "bad request"}, 400)
            return
        if not prompt or len(prompt) > 2000:
            self.send_json({"error": "Prompt must be 1-2000 characters."}, 400)
            return
        if size not in SIZES or frames not in FRAMES:
            self.send_json({"error": "Unsupported size or length."}, 400)
            return
        with lock:
            if sum(j["status"] in ("queued", "running") for j in jobs) >= MAX_QUEUE:
                self.send_json({"error": f"Queue is full ({MAX_QUEUE} jobs)."}, 429)
                return
            jid = uuid.uuid4().hex[:8]
            job = {"id": jid, "prompt": prompt, "size": size, "frames": frames, "seed": seed,
                   "user": self.user(), "status": "queued", "queued": time.time(),
                   "file": f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(prompt)}-{jid}.mp4"}
            jobs.append(job)
            del jobs[:-100]
            wake.notify()
        self.send_json(job, 201)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=publish_state, daemon=True).start()
    if TS_LISTEN:
        host, port = TS_LISTEN.rsplit(":", 1)
        ts_server = http.server.ThreadingHTTPServer((host, int(port)), Handler)
        threading.Thread(target=ts_server.serve_forever, daemon=True).start()
    http.server.ThreadingHTTPServer(LISTEN, Handler).serve_forever()
