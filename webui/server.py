#!/usr/bin/env python3
"""LTX-2.5 web UI: queue text-to-video and image-to-video jobs and browse/play the results.

Binds to 127.0.0.1 only; the tailnet reaches it through `tailscale serve`, which
also supplies the Tailscale-User-Login header used to tag who queued a job.
Jobs run one at a time through ../run.sh, so they show up in the Grafana
"LTX-2.5 Video Generation" dashboard like any CLI run.
"""
import base64
import http.server
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imageinfo  # noqa: E402

# LTX_ROOT / LTX_RUN_SCRIPT let a test instance run against a scratch dir and a stub pipeline.
ROOT = Path(os.environ.get("LTX_ROOT", Path(__file__).resolve().parent.parent))
RUN_SCRIPT = os.environ.get("LTX_RUN_SCRIPT", str(Path(__file__).resolve().parent.parent / "run.sh"))
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "uploads"  # conditioning images for image-to-video
RUNNING_DIR = ROOT / "logs" / "running"  # ltx_job.py: one <job_id>.json per in-flight job
LEGACY_CURRENT = ROOT / "logs" / "current.json"  # older ltx_job.py versions
WEBUI_STATE = ROOT / "logs" / "webui_state.json"  # read by ltx_exporter.py for Grafana
QUEUE_FILE = ROOT / "logs" / "webui_queue.json"
# While this file exists no new jobs start (running ones finish; new requests still queue).
# Its text is shown to users, e.g.:  echo "Benchmark running" > logs/webui_pause
PAUSE_FILE = ROOT / "logs" / "webui_pause"  # survives restarts: queued jobs resume, running ones are adopted
HISTORY = ROOT / "logs" / "jobs.jsonl"
INDEX = Path(__file__).resolve().parent / "index.html"
LISTEN = (os.environ.get("LTX_WEBUI_HOST", "127.0.0.1"), int(os.environ.get("LTX_WEBUI_PORT", "8090")))
# Optional second listener on the Tailscale IP itself (e.g. 100.x.y.z:8091) for tailnet
# clients whose MagicDNS is off; tailscale serve only routes by hostname, so a bare IP 404s there.
TS_LISTEN = os.environ.get("LTX_WEBUI_TS_LISTEN", "")
TAILSCALE = os.environ.get("TAILSCALE_BIN") or shutil.which("tailscale") or "tailscale"
# Optional Docker container name of an LLM server sharing unified memory with LTX; only used to
# explain a low-memory refusal. Empty = don't check.
LLM_CONTAINER = os.environ.get("LLM_CONTAINER", "")
_whois_cache = {}

# Final output sizes; the distilled pipeline renders stage 1 at half size, so both
# dimensions must be multiples of 64.
SIZES = imageinfo.SIZES | {"768x512", "512x768", "1024x576", "576x1024", "1280x704", "704x1280", "1536x1024"}
MAX_IMAGE_BYTES = 16 * 2**20
MAX_BODY_BYTES = MAX_IMAGE_BYTES * 4 // 3 + 64 * 1024  # base64 overhead + JSON
FRAMES = {49, 97, 121, 193}  # 2s, 4s, 5s, 8s at 24 fps (must be 8k+1)
# A job peaks ~29 GiB system-wide; 50 GiB free before starting leaves ~20 GiB after it loads,
# the margin below which this unified-memory box starts swapping. Override with LTX_MIN_FREE_GIB.
MIN_FREE_BYTES = int(os.environ.get("LTX_MIN_FREE_GIB", "50")) * 2**30
MAX_QUEUE = 10
# Parallel jobs. They share one GPU, so each runs slower; overlap mainly hides model loading.
WORKERS = max(1, int(os.environ.get("LTX_WORKERS", "1")))
# Per-job GPU memory peak as nvidia-smi reports it (measured 22.5-23.4 GiB; 8 s at 1280x704: 26 GiB); reserved for
# running jobs that haven't allocated it yet. Compared against gpu_mem_bytes, so not the ~29 GiB system-wide figure.
JOB_PEAK_BYTES = int(float(os.environ.get("LTX_JOB_PEAK_GIB", "27")) * 2**30)

lock = threading.Lock()
wake = threading.Condition(lock)
jobs = []  # web-submitted jobs, newest last


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def llm_running():
    if not LLM_CONTAINER:
        return None
    try:
        out = subprocess.run(["docker", "ps", "--filter", f"name=^{LLM_CONTAINER}$", "--format", "{{.Status}}"],
                             capture_output=True, text=True, timeout=5).stdout
        return out.startswith("Up")
    except (OSError, subprocess.TimeoutExpired):
        return None


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-") or "clip"


def paused_reason():
    try:
        return PAUSE_FILE.read_text().strip()[:200] or "Paused by the admin"
    except OSError:
        return None


def save_queue():
    """Persist the job list (called with `lock` held). Restart-safe with systemd KillMode=process,
    which lets in-flight renders outlive the server so a new server can adopt them."""
    tmp = QUEUE_FILE.with_name(f".{QUEUE_FILE.name}.tmp")
    tmp.write_text(json.dumps(jobs[-100:]))
    tmp.replace(QUEUE_FILE)


def load_queue():
    try:
        saved = json.loads(QUEUE_FILE.read_text())
    except (OSError, ValueError):
        return
    for j in saved:
        if j.get("status") == "running":
            j["adopted"] = True  # its render may still be going; adoption_monitor settles it
    jobs.extend(saved)


def adoption_monitor():
    """Settle running jobs inherited from a previous server once their render process is gone."""
    time.sleep(10)  # let in-flight renders refresh their state files
    while True:
        live = {Path(st.get("output", "")).name for st in running_states()}
        with lock:
            changed = False
            for j in jobs:
                if j.get("adopted") and j["status"] == "running" and j["file"] not in live:
                    ok = (OUTPUTS / j["file"]).is_file()
                    j.update(status="done" if ok else "failed", finished=time.time())
                    if not ok:
                        j["error"] = "Interrupted: the studio restarted and the render didn't finish."
                    changed = True
            if changed:
                save_queue()
                wake.notify_all()
        time.sleep(3)


def effective_free():
    """MemAvailable minus memory that running jobs will still allocate (they ramp up over ~20 s).

    A web job marked running may not have written its state file yet (the pipeline process is
    still starting), so it is reserved at the full expected peak. Called with `lock` held.
    """
    states = running_states()
    pending = sum(max(0, JOB_PEAK_BYTES - st.get("gpu_mem_bytes", 0)) for st in states)
    seen = {Path(st.get("output", "")).name for st in states}
    pending += JOB_PEAK_BYTES * sum(j["status"] == "running" and j["file"] not in seen for j in jobs)
    return mem_available() - pending


def worker():
    while True:
        with lock:
            while not any(j["status"] == "queued" for j in jobs):
                wake.wait()
            job = next(j for j in jobs if j["status"] == "queued")
            if PAUSE_FILE.exists():
                wake.wait(timeout=5)
                continue
            if sum(j["status"] == "running" for j in jobs) >= WORKERS:
                wake.wait(timeout=5)  # adopted renders from before a restart can fill the slots
                continue
            free = effective_free()
            if free < MIN_FREE_BYTES:
                if any(j["status"] == "running" for j in jobs):
                    # Another job holds the memory: wait for it instead of failing this one.
                    job["waiting"] = "memory"
                    wake.wait(timeout=5)
                    continue
                job.update(status="failed", finished=time.time(),
                           error=f"Not enough free memory ({mem_available() / 2**30:.0f} GiB free, need "
                                 f"{MIN_FREE_BYTES // 2**30}). Is another app (e.g. an LLM server"
                                 f"{f' {LLM_CONTAINER}' if LLM_CONTAINER else ''}) holding memory?")
                save_queue()
                continue
            job.pop("waiting", None)
            job.update(status="running", started=time.time())
            save_queue()
        w, h = job["size"].split("x")
        cmd = [RUN_SCRIPT, job["prompt"], str(OUTPUTS / job["file"]),
               "--width", w, "--height", h, "--num-frames", str(job["frames"])]
        if job.get("seed") is not None:
            cmd += ["--seed", str(job["seed"])]
        if job.get("image"):
            # Condition frame 0 on the uploaded still (image-to-video).
            cmd += ["--image", str(UPLOADS / job["image"]), "0", str(job["strength"])]
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
            save_queue()
            wake.notify_all()  # memory freed: let waiting workers re-check


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


def running_states():
    """Live state of every in-flight job (any entry point), oldest first."""
    states = []
    for f in RUNNING_DIR.glob("*.json"):
        try:
            st = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if st.get("running") and Path(f"/proc/{st.get('pid')}").exists():
            states.append(st)
    if not states:
        try:
            st = json.loads(LEGACY_CURRENT.read_text())
            if st.get("running") and Path(f"/proc/{st.get('pid')}").exists():
                states.append(st)
        except (OSError, ValueError):
            pass
    return sorted(states, key=lambda st: st.get("start", 0))


def live_progress():
    """Per running job: stage/progress for the activity rail, keyed by output file name."""
    now = time.time()
    return [{k: st.get(k) for k in ("stage", "step", "steps", "prompt", "user", "source", "mode", "width", "height", "frames")}
            | {"file": Path(st.get("output", "")).name, "elapsed": round(now - st["start"], 1),
               "gpu_mem_gib": round(st.get("gpu_mem_bytes", 0) / 2**30, 1)} for st in running_states()]


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


def source_image(r):
    """Thumbnail name in uploads/ for an image-to-video record, or None.

    Studio jobs already keep their image in uploads/. CLI jobs (i2v.sh) point anywhere, so the
    first time one is listed its image is copied in as src-<job_id>-<name>.
    """
    name = r.get("image")
    if not name:
        return None
    if (UPLOADS / name).is_file():
        return name
    src = Path(r.get("image_path") or "")
    if not src.is_file():
        return None
    cached = f"src-{r.get('job_id', 'job')}-{src.name}"
    if not (UPLOADS / cached).is_file():
        UPLOADS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, UPLOADS / cached)
    return cached


def recent_history(n=15):
    """Last n finished jobs from any entry point (web or CLI), newest first, for the activity rail."""
    rows = []
    try:
        with open(HISTORY) as f:
            lines = f.readlines()[-n:]
    except OSError:
        return rows
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        name = Path(r.get("output", "")).name
        rows.append({"file": name if (OUTPUTS / name).is_file() else None, "status": r.get("status"),
                     "prompt": r.get("prompt", ""), "user": r.get("user", ""), "source": r.get("source", "cli"),
                     "mode": r.get("mode", "t2v"), "image": source_image(r),
                     "width": r.get("width"), "height": r.get("height"), "frames": r.get("frames"),
                     "duration_s": r.get("duration_s"), "end": r.get("end")})
    return rows


def list_videos():
    hist = history_by_file()
    vids = []
    for p in sorted(OUTPUTS.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = hist.get(p.name, {})
        vids.append({"file": p.name, "size_mb": round(p.stat().st_size / 2**20, 1), "mtime": p.stat().st_mtime,
                     "prompt": r.get("prompt", ""), "duration_s": r.get("duration_s"),
                     "image": source_image(r), "mode": r.get("mode", "t2v"), "source": r.get("source", "cli"),
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
            lives = live_progress()
            self.send_json({"jobs": q[::-1], "lives": lives, "live": lives[0] if lives else None,
                            "workers": WORKERS, "mem_free_gib": round(free / 2**30, 1),
                            "can_run": free >= MIN_FREE_BYTES or bool(lives),  # with jobs running, new ones queue and wait "llm_running": llm_running(), "llm_container": LLM_CONTAINER,
                            "min_free_gib": MIN_FREE_BYTES // 2**30, "user": self.user(),
                            "now": time.time(), "recent": recent_history(), "paused": paused_reason()})
        elif path == "/api/videos":
            self.send_json(list_videos())
        elif path.startswith("/videos/"):
            self.send_file(OUTPUTS, unquote(path[len("/videos/"):]))
        elif path.startswith("/uploads/"):
            self.send_file(UPLOADS, unquote(path[len("/uploads/"):]))
        else:
            self.send_error(404)

    def send_file(self, folder, name):
        p = (folder / name).resolve()
        if p.parent != folder.resolve() or not p.is_file():
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
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > MAX_BODY_BYTES:
            self.send_json({"error": f"Image too large (max {MAX_IMAGE_BYTES // 2**20} MB)."}, 413)
            return
        try:
            req = json.loads(self.rfile.read(length))
            prompt = str(req["prompt"]).strip()
            size, frames = str(req["size"]), int(req["frames"])
            seed = int(req["seed"]) if str(req.get("seed", "")).strip() else None
            strength = float(req.get("strength", 1.0))
            image_data = req.get("image") or ""
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": "bad request"}, 400)
            return
        image = None
        if image_data:
            try:  # accepts a data: URL or bare base64
                raw = base64.b64decode(image_data.split(",", 1)[-1], validate=True)
            except ValueError:
                self.send_json({"error": "Image is not valid base64."}, 400)
                return
            info = imageinfo.sniff(raw)
            if len(raw) > MAX_IMAGE_BYTES or not info:
                self.send_json({"error": "Image must be a PNG, JPEG or WebP up to "
                                         f"{MAX_IMAGE_BYTES // 2**20} MB."}, 400)
                return
            if not 0.3 <= strength <= 1.0:
                self.send_json({"error": "Image strength must be between 0.3 and 1.0."}, 400)
                return
            image = {"kind": info[0], "width": info[1], "height": info[2], "raw": raw}
            if size.startswith("auto"):
                tier = size.partition(":")[2] or "fast"
                if tier not in imageinfo.TIERS:
                    self.send_json({"error": "Unknown size tier."}, 400)
                    return
                size = "%dx%d" % imageinfo.best_size(info[1], info[2], tier)
        elif size.startswith("auto"):
            self.send_json({"error": "Auto size needs an image."}, 400)
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
                   "mode": "i2v" if image else "t2v",
                   "file": f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(prompt)}-{jid}.mp4"}
            if image:
                UPLOADS.mkdir(parents=True, exist_ok=True)
                name = f"{jid}.{'jpg' if image['kind'] == 'jpeg' else image['kind']}"
                (UPLOADS / name).write_bytes(image["raw"])
                job.update(image=name, strength=round(strength, 2),
                           image_size=f"{image['width']}x{image['height']}")
            jobs.append(job)
            del jobs[:-100]
            save_queue()
            wake.notify()
        self.send_json(job, 201)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    load_queue()
    threading.Thread(target=adoption_monitor, daemon=True).start()
    for _ in range(WORKERS):
        threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=publish_state, daemon=True).start()
    if TS_LISTEN:
        host, port = TS_LISTEN.rsplit(":", 1)
        ts_server = http.server.ThreadingHTTPServer((host, int(port)), Handler)
        threading.Thread(target=ts_server.serve_forever, daemon=True).start()
    http.server.ThreadingHTTPServer(LISTEN, Handler).serve_forever()
