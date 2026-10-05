#!/usr/bin/env python3
"""Generate the "LTX-2.5 Video Generation" Grafana dashboard.

Usage: make_dashboard.py [output.json]   (default: monitoring/grafana-dashboard.json)

Metrics come from ltx_exporter.py (job state/history), dcgm-exporter (GPU) and
node-exporter (unified RAM). Re-run after editing; Grafana reloads in ~10s.
"""
import json
import sys
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def nid():
    global _id
    _id += 1
    return _id


def targets(exprs, instant=False):
    t = [{"datasource": DS, "expr": e, "legendFormat": l, "refId": chr(65 + i)} for i, (e, l) in enumerate(exprs)]
    if instant:  # label-valued stats (stage, requester): only the current series, not every one seen in range
        for x in t:
            x.update(instant=True, range=False)
    return t


def row(title, y):
    return {"type": "row", "title": title, "collapsed": False, "id": nid(),
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}


def stat(title, exprs, x, y, w=4, h=5, unit="none", mappings=None, steps=None, decimals=None, desc="", no_value=None,
         instant=False):
    p = {
        "type": "stat", "title": title, "description": desc, "datasource": DS, "id": nid(),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": targets(exprs, instant),
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "background", "graphMode": "none",
                    # label-valued stats (instant) show the series name, e.g. the stage, not its value 1
                    "textMode": "name" if instant else "value_and_name" if len(exprs) > 1 else "value",
                    "justifyMode": "center"},
        "fieldConfig": {"defaults": {
            "unit": unit, "mappings": mappings or [],
            "color": {"mode": "thresholds"},
            "thresholds": {"mode": "absolute", "steps": steps or [{"color": "blue", "value": None}]},
        }, "overrides": []},
    }
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    if no_value is not None:
        p["fieldConfig"]["defaults"]["noValue"] = no_value
    return p


def ts(title, exprs, x, y, w=12, h=8, unit="none", stack=False, maxv=None, desc="", step_after=False):
    d = {"unit": unit, "color": {"mode": "palette-classic"},
         "custom": {"lineWidth": 2, "fillOpacity": 15, "showPoints": "never", "spanNulls": True,
                    "lineInterpolation": "stepAfter" if step_after else "linear",
                    "stacking": {"mode": "normal" if stack else "none"}}}
    if maxv is not None:
        d["max"], d["min"] = maxv, 0
    return {"type": "timeseries", "title": title, "description": desc, "datasource": DS, "id": nid(),
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": targets(exprs),
            "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                        "tooltip": {"mode": "multi"}},
            "fieldConfig": {"defaults": d, "overrides": []}}


GPU = 'DCGM_FI_DEV_{}{{gpu="0"}}'
RAM_USED = "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes"
# The LLM sharing the box with LTX; edit the port if yours differs.
ORNITH_UP = 'max(up{job="vllm",instance=~".*:8004"}) or vector(0)'

panels = []
y = 0
panels.append(row("Now", y)); y += 1
panels += [
    stat("LTX status", [("ltx_job_running", "")], 0, y,
         mappings=[{"type": "value", "options": {"0": {"text": "Idle", "color": "text"},
                                                 "1": {"text": "Generating", "color": "green"}}}]),
    stat("Stage", [('max by (stage) (ltx_job_stage) == 1', "{{stage}}")], 4, y, w=5,
         steps=[{"color": "purple", "value": None}],
         desc="Pipeline stage of the running job.", no_value="Idle", instant=True),
    stat("Denoise progress", [("100 * ltx_job_step / clamp_min(ltx_job_steps, 1)", "")], 9, y, w=3,
         unit="percent", decimals=0, desc="Steps done in the current denoising stage (8 steps, then 3)."),
    stat("Elapsed", [("ltx_job_elapsed_seconds", "")], 12, y, w=3, unit="s"),
    stat("LTX GPU memory", [("ltx_job_gpu_memory_bytes", "")], 15, y, w=3, unit="bytes",
         steps=[{"color": "blue", "value": None}, {"color": "orange", "value": 40e9}, {"color": "red", "value": 60e9}]),
    stat("Ornith LLM (:8004)", [(ORNITH_UP, "")], 18, y, w=3,
         mappings=[{"type": "value", "options": {"0": {"text": "Stopped", "color": "orange"},
                                                 "1": {"text": "Running", "color": "green"}}}],
         desc="Shares the 121 GiB unified pool with LTX. Both at once will not fit."),
    stat("RAM available", [("node_memory_MemAvailable_bytes", "")], 21, y, w=3, unit="bytes",
         steps=[{"color": "red", "value": None}, {"color": "orange", "value": 20e9}, {"color": "green", "value": 40e9}],
         desc="Unified memory left for everything. Under ~20 GiB the box starts swapping."),
]
y += 5

panels.append(row("Totals", y)); y += 1
panels += [
    stat("Jobs", [('ltx_jobs_total{status="ok"}', "ok"), ('ltx_jobs_total{status="failed"}', "failed")], 0, y, w=5,
         steps=[{"color": "green", "value": None}]),
    stat("Video generated", [("ltx_video_seconds_generated_total", "")], 5, y, unit="s", decimals=1),
    stat("Avg job time", [("ltx_job_duration_seconds_sum / clamp_min(ltx_job_duration_seconds_count, 1)", "")],
         9, y, w=3, unit="s", decimals=0),
    stat("Last job", [("ltx_last_job_success", "")], 12, y, w=3,
         mappings=[{"type": "value", "options": {"0": {"text": "Failed", "color": "red"},
                                                 "1": {"text": "OK", "color": "green"}}}]),
    stat("Last job time", [("ltx_last_job_duration_seconds", "")], 15, y, w=3, unit="s", decimals=0),
    stat("Last peak memory", [("ltx_last_job_peak_gpu_memory_bytes", "")], 18, y, w=3, unit="bytes"),
    stat("Compute s per video s", [("ltx_last_job_realtime_factor", "")], 21, y, w=3, decimals=1,
         desc="Last job: wall seconds per second of output video."),
]
y += 5

panels.append(row("Activity", y)); y += 1
panels += [
    ts("Unified memory", [(RAM_USED, "system used"), ("ltx_job_gpu_memory_bytes", "LTX process")], 0, y,
       unit="bytes", desc="System-wide used memory vs the LTX process's GPU allocation."),
    ts("Job stage & progress", [("ltx_job_running", "running"),
                                ("ltx_job_step / clamp_min(ltx_job_steps, 1)", "denoise progress")], 12, y,
       unit="percentunit", maxv=1, step_after=True),
]
y += 8
panels += [
    ts("GPU utilization", [(GPU.format("GPU_UTIL"), "GPU util %")], 0, y, w=8, unit="percent", maxv=100),
    ts("GPU power", [(GPU.format("POWER_USAGE"), "W")], 8, y, w=8, unit="watt"),
    ts("GPU temperature", [(GPU.format("GPU_TEMP"), "°C")], 16, y, w=8, unit="celsius"),
]
y += 8
panels += [
    ts("Jobs finished", [('increase(ltx_jobs_total{status="ok"}[5m])', "ok / 5m"),
                         ('increase(ltx_jobs_total{status="failed"}[5m])', "failed / 5m")], 0, y,
       step_after=True),
    ts("Last job duration", [("ltx_last_job_duration_seconds", "seconds")], 12, y, unit="s", step_after=True),
]

y += 8
panels.append(row("Web UI (tailnet)", y)); y += 1
panels += [
    stat("Web UI", [("ltx_webui_up", "")], 0, y,
         mappings=[{"type": "value", "options": {"0": {"text": "Down", "color": "red"},
                                                 "1": {"text": "Up", "color": "green"}}}],
         desc="LTX Video Studio (webui/server.py)."),
    stat("Queued", [('ltx_webui_queue_jobs{state="queued"}', "")], 4, y, w=3,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 3}, {"color": "red", "value": 8}]),
    stat("Running for", [('max by (user, source) (ltx_job_running_source{source!="none"}) == 1', "{{user}} ({{source}})")],
         7, y, w=6, no_value="Nobody", steps=[{"color": "purple", "value": None}], instant=True),
    stat("Refused: low memory", [("ltx_webui_rejected_low_memory", "")], 13, y, w=4,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 1}],
         desc="Web requests refused because Ornith (or something else) held the memory."),
    stat("Web vs CLI jobs", [('sum(ltx_jobs_by_user_total{source="webui"}) or vector(0)', "web"),
                             ('sum(ltx_jobs_by_user_total{source="cli"}) or vector(0)', "cli")], 17, y, w=7,
         steps=[{"color": "blue", "value": None}]),
]
y += 5
panels += [
    {"type": "bargauge", "title": "Jobs by requester", "datasource": DS, "id": nid(),
     "gridPos": {"h": 8, "w": 12, "x": 0, "y": y},
     "targets": targets([("sum by (user) (ltx_jobs_by_user_total)", "{{user}}")]),
     "options": {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                 "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
     "fieldConfig": {"defaults": {"min": 0, "color": {"mode": "palette-classic"}}, "overrides": []}},
    ts("Web UI queue", [('ltx_webui_queue_jobs{state="queued"}', "queued"),
                        ('ltx_webui_queue_jobs{state="running"}', "running")], 12, y, step_after=True),
]

dashboard = {
    "uid": "ltx-video", "title": "LTX-2.5 Video Generation", "tags": ["ltx", "video", "gpu"],
    "timezone": "browser", "schemaVersion": 39, "version": 4, "editable": True,
    "refresh": "5s", "time": {"from": "now-1h", "to": "now"},
    "annotations": {"list": []}, "templating": {"list": []}, "panels": panels,
}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "grafana-dashboard.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
