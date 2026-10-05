<div align="center">

<img src="assets/banner.png" alt="LTX-2.5 on DGX Spark" width="100%">

# LTX-2.5 (NVFP4) on NVIDIA DGX Spark (GB10)

**Generate video with synchronized audio from a text prompt or a still image on a single DGX Spark: a 4-second 768×512 clip in about a minute, with a tailnet web studio and Grafana monitoring**

[![Model](https://img.shields.io/badge/model-Lightricks%2FLTX--2.5-blue)](https://huggingface.co/Lightricks/LTX-2.5)
[![Quant](https://img.shields.io/badge/quant-NVFP4%20DiT%20%2B%20BF16%20VAE-8A2BE2)](#configuration-reference)
[![Code](https://img.shields.io/badge/LTX--2-v1.4.2-orange)](https://github.com/Lightricks/LTX-2/tree/v1.4.2)
[![Hardware](https://img.shields.io/badge/hardware-DGX%20Spark%20(GB10%2C%20sm__121)-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20%2B%20cu132-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/scripts-Apache%202.0-green)](#credits--license)

</div>

<div align="center">
<img src="assets/studio-live.png" alt="LTX Video Studio live on the DGX Spark: two clips rendering in parallel, queue and activity rail" width="100%">
<br><sub><b>LTX Video Studio</b> live on the Spark: two clips rendering in parallel, the queue in order, and every stage tracked in the activity rail.</sub>
</div>

---

## Overview

This repository is the working setup used to run
[**Lightricks LTX-2.5**](https://huggingface.co/Lightricks/LTX-2.5), a 22B-parameter audio-video
diffusion transformer, on a single **NVIDIA DGX Spark (GB10, Blackwell, 128 GB unified memory)**.
It is a sister project to the
[Ornith-1.5 LLM deployment](https://github.com/Hitheshkaranth/Ornith-1.5_A3B_Model_DGX_Spark_Setup)
on the same box.

It gives you:

- **A working NVFP4 build for GB10.** Lightricks ships a pre-quantized 4-bit (NVFP4) distilled
  transformer, but the upstream CUDA kernels are only compiled for `sm_100a`/`sm_110a`/`sm_120a`.
  GB10 is **`sm_121`**, and arch-specific (`a`) cubins don't load on a different minor version.
  A [three-line patch](patches/ltx-kernels-sm121a.patch) adds `sm_121a`.
- **One-command install.** `install.sh` pins the tested upstream tag and lockfile, applies the patch,
  builds the kernels and downloads the 44 GiB of weights.
- **Image-to-video.** Drop in a photo and describe the motion; `i2v.sh` and the studio size the output
  to the photo's aspect ratio (EXIF rotation included) so the subject isn't cropped away.
- **LTX Video Studio**, a small web app: prompt or image in, video with sound out, served privately to your
  [Tailscale](https://tailscale.com/) tailnet, with a queue and per-user attribution.
- **Grafana monitoring:** live stage, progress and memory for every job, plus history and who asked for what.

> Every number in this README was measured on the deployment itself. See [Benchmarks](#benchmarks);
> raw data is in [`benchmarks/results.json`](benchmarks/results.json).

### Samples

All four were generated on the Spark with the commands in this README (768×512, 24 fps, with audio;
GIF previews are downscaled and silent).

| A red fox trots through fresh snow at golden hour | Coffee by a rainy café window, slow push-in | Sunset waves crash on volcanic rocks, aerial | **Image → video:** a still animated with *"waves burst against the rocks, slow push-in"* |
|:---:|:---:|:---:|:---:|
| <img src="assets/sample-fox.gif" width="100%"> | <img src="assets/sample-coffee.gif" width="100%"> | <img src="assets/sample-waves.gif" width="100%"> | <img src="assets/sample-i2v.gif" width="100%"> |
| 2 s · text | 2 s · text | 4 s · text | 4 s · from an image |

### Screenshots

| LTX Video Studio: two clips rendering in parallel, image attached | Grafana: every job's stage, memory, GPU and history |
|:---:|:---:|
| <a href="assets/studio.png"><img src="assets/studio.png" alt="LTX Video Studio" width="100%"></a> | <a href="assets/grafana.png"><img src="assets/grafana.png" alt="LTX-2.5 Grafana dashboard" width="100%"></a> |

## Table of Contents

- [Architecture](#architecture)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Image-to-Video](#image-to-video)
- [The GB10 (sm_121) Patch](#the-gb10-sm_121-patch)
- [LTX Video Studio (Web UI)](#ltx-video-studio-web-ui)
- [Monitoring with Grafana](#monitoring-with-grafana)
- [Sharing the Box with an LLM](#sharing-the-box-with-an-llm)
- [Benchmarks](#benchmarks)
- [Repository Layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Credits & License](#credits--license)

---

## Architecture

### Generation pipeline

`run.sh` drives the upstream `ltx_pipelines.distilled` pipeline: a two-pass distilled schedule
that renders at half resolution, upsamples in latent space, refines, then decodes both video and audio.

```mermaid
flowchart LR
    P(["Text prompt"])
    TE["Gemma 4 12B text encoder\nLTX fine-tune · BF16"]
    D1["Pass 1 · 8 steps\nhalf resolution\nNVFP4 22B DiT"]
    UP["Latent spatial\nupscaler ×2"]
    D2["Pass 2 · 3 steps\nfull resolution\nNVFP4 22B DiT"]
    DEC["Video VAE + Audio VAE\n+ vocoder"]
    OUT(["MP4\nH.264 + AAC stereo"])

    P ==> TE ==> D1 ==> UP ==> D2 ==> DEC ==> OUT

    classDef io fill:#5794F2,stroke:#2D5FA3,color:#ffffff,stroke-width:2px
    classDef model fill:#FF9830,stroke:#C46F1F,color:#1a1a1a,stroke-width:2px
    classDef nvfp4 fill:#73BF69,stroke:#3F7A39,color:#0a1f08,stroke-width:2px
    class P,OUT io
    class TE,UP,DEC model
    class D1,D2 nvfp4
```

Each stage loads its model, runs, and frees it before the next, so peak memory stays around
23 GiB instead of the ~44 GiB sum of all the weights.

### Serving and monitoring

```mermaid
flowchart LR
    subgraph Tailnet["Tailscale tailnet (private)"]
        U1(["Laptop / Mac\nbrowser"])
    end
    subgraph Spark["DGX Spark"]
        TS["tailscale serve\nHTTPS · identity header"]
        UI["LTX Video Studio\nwebui/server.py · :8090\nqueue · memory gate"]
        RUN["run.sh → ltx_job.py\nstage · progress · memory"]
        PIPE["ltx_pipelines.distilled\nNVFP4 on GB10"]
        LOG[("logs/current.json\nlogs/jobs.jsonl")]
        EXP["ltx_exporter.py · :9092"]
        PROM["Prometheus"]
        GF["Grafana\nLTX-2.5 dashboard"]
        DCGM["dcgm-exporter\nnode-exporter"]
    end

    U1 -->|"HTTPS · host.ts.net"| TS --> UI --> RUN --> PIPE
    RUN --> LOG --> EXP --> PROM --> GF
    DCGM --> PROM

    classDef user fill:#5794F2,stroke:#2D5FA3,color:#ffffff,stroke-width:2px
    classDef svc fill:#FF9830,stroke:#C46F1F,color:#1a1a1a,stroke-width:2px
    classDef mon fill:#B877D9,stroke:#7A4A93,color:#ffffff,stroke-width:2px
    classDef store fill:#73BF69,stroke:#3F7A39,color:#0a1f08,stroke-width:2px
    class U1 user
    class TS,UI,RUN,PIPE svc
    class EXP,PROM,GF,DCGM mon
    class LOG store
```

The command line and the web UI both go through `run.sh`, so every job, whatever started it, shows
up in Grafana.

---

## Hardware & Software Requirements

| | Tested with | Notes |
|---|---|---|
| GPU | NVIDIA GB10 (DGX Spark), compute capability **12.1** | NVFP4 needs Blackwell (≥ 10.0). `install.sh` builds for whatever it finds. |
| Memory | 121 GiB unified (CPU + GPU share one pool) | A job needs **~23 GiB** for the LTX process, **~29 GiB** system-wide at peak |
| Driver / CUDA | 580.173 (CUDA 13.0) | PyTorch's cu132 wheels run on it via minor-version compatibility |
| OS | Ubuntu 24.04 (aarch64), DGX OS | |
| Python | 3.13 (uv-managed) | System Python lacks headers needed by Triton and the kernel build |
| Disk | ~45 GiB weights + ~7 GiB env | |
| Network | Hugging Face account | `Lightricks/LTX-2.5` is gated with **auto-approval**: click *Agree* once |

## Quick Start

```bash
# 1. Accept the model terms (instant): https://huggingface.co/Lightricks/LTX-2.5
# 2. Log in to Hugging Face (opens a device-code flow; works over SSH)
uvx --from 'huggingface_hub>=2.1' hf auth login

# 3. Install: clone, patch, build env + NVFP4 kernels, download weights (~44 GiB)
curl -fsSL https://raw.githubusercontent.com/Hitheshkaranth/LTX-2.5_Video_DGX_Spark_Setup/main/install.sh | bash

# 4. Generate
cd LTX-2.5_Video_DGX_Spark_Setup
./run.sh "A red fox trots through fresh snow in a quiet birch forest at golden hour, soft breath vapor, gentle wind sound, cinematic tracking shot" \
         outputs/fox.mp4 --height 512 --width 768 --num-frames 97

# 5. Or animate a photo (output size follows the photo's aspect ratio)
./i2v.sh photo.jpg "She turns toward the camera and smiles as wind lifts her hair, soft city ambience" outputs/smile.mp4
```

Optional extras:

```bash
./setup-services.sh            # web UI on 127.0.0.1:8090 + Prometheus exporter on :9092
./setup-services.sh --tailnet  # ...and publish the UI to your tailnet over HTTPS
```

> **Prompting tip:** LTX generates sound too, so describe it. Subject, action, setting, lighting,
> camera movement *and* audio ("soft wind", "rain patter", "seagulls calling") give the best results.

## Configuration Reference

`run.sh "<prompt>" [output.mp4] [extra pipeline flags…]` wraps:

| Flag | Value | Why |
|---|---|---|
| `--quantization nvfp4-prequant` | | Load Lightricks' pre-quantized NVFP4 transformer and run FP4 tensor-core GEMMs |
| `--transformer-path` | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (17.4 GiB) | **Split layout**: use this, not `--distilled-checkpoint-path`, which expects a monolith |
| `--text-encoder-path` | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (24.5 GiB) | Gemma 4 12B fine-tuned for LTX, bundled in the model repo, so no separate Google license |
| `--video-vae-path` / `--audio-vae-path` | BF16 VAEs (1.4 / 0.3 GiB) | Docs pair NVFP4 transformers with BF16 VAEs |
| `--spatial-upsampler-path` | latent ×2 upscaler (0.9 GiB) | Drives the two-pass schedule |
| `--seed` | `42` by default | Override by passing `--seed N` after the output path |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Recommended upstream; reduces fragmentation between stages |

Useful extra flags: `--width`/`--height` (multiples of 64, since pass 1 runs at half size),
`--num-frames` (**8k+1**: 49 = 2 s, 97 = 4 s, 121 = 5 s, 193 = 8 s at 24 fps).

> `--offload cpu` gains nothing on a Spark, because CPU and GPU memory are the same physical pool.

## Image-to-Video

LTX-2.5 conditions on a still image through the same distilled NVFP4 pipeline, with no extra weights.
The image becomes frame 0 and the prompt describes what happens next.

```bash
./i2v.sh IMAGE "prompt" [out.mp4] [extra pipeline flags]

LTX_TIER=hd ./i2v.sh portrait.jpg "Slow dolly-in as she looks up, rain ambience" outputs/portrait.mp4 --num-frames 121
LTX_IMAGE_STRENGTH=0.8 ./i2v.sh street.png "Traffic starts moving, horns, evening light"
```

| Setting | Default | Effect |
|---|---|---|
| `LTX_TIER` | `fast` | Resolution tier: `fast` (~768 px long side), `hd` (~1024–1280), `max` (~1536) |
| `LTX_IMAGE_STRENGTH` | `1.0` | 1.0 keeps frame 0 identical to the image; lower values (0.3–1.0) let the model drift |
| `--width/--height` | auto | Pass your own to override the automatic size |

Why the automatic sizing matters: the pipeline **resizes and center-crops** the image to the output
size. A portrait photo rendered at 768×512 loses most of its top and bottom. `webui/imageinfo.py` reads
the image header (PNG, JPEG, WebP; standard library only) and picks the preset closest in aspect ratio:

| Image | `fast` | `hd` | `max` |
|---|---|---|---|
| 4:3 landscape (e.g. 4032×3024) | 768×512 | 1024×768 | 1536×1024 |
| 3:4 portrait phone photo | 512×768 | 768×1024 | 1024×1536 |
| 16:9 (1920×1080) | 768×512 | 1024×576 | 1536×1024 |
| Square | 640×640 | 768×768 | 1024×1024 |

Phone JPEGs are usually stored landscape with an EXIF "rotate 90°" tag. The pipeline's decoder
applies that rotation, so the size picker honours EXIF orientation too; otherwise a portrait photo
would be rendered landscape and cropped. Argument handling, EXIF rotation and the pipeline's own
preprocessing were also verified on CPU against the real parser.

### Image-to-video settings in the studio

| Control | Options | Notes |
|---|---|---|
| **Image** | Picture button, drag onto the page, or paste | PNG, JPEG or WebP, up to 16 MB. Becomes frame 0. |
| **Shape** | **Match image** (default once an image is attached), Landscape, Portrait, Square | Match image picks the preset closest to the photo's aspect ratio (table above); the others crop |
| **Quality** | Fast · HD · Max | Same tiers as `LTX_TIER`: ~768 / ~1024–1280 / ~1536 px |
| **Length** | 2 · 4 · 5 · 8 s | 49 / 97 / 121 / 193 frames at 24 fps |
| **Image strength** | 0.30–1.00 (default 1.00) | 1.0 keeps frame 0 identical to the photo; lower drifts further |
| **Seed** | optional | Same image + prompt + settings + seed reproduces a clip |
| **Prompt** | free text | Describe *motion, camera and sound*; the image already sets the look |

The API takes the same settings: `POST /api/generate` with `{"prompt", "size": "auto:fast|auto:hd|auto:max"
or "WxH", "frames", "seed", "image": "<data: URL or base64>", "strength"}`.

### Measured on the GPU

<div align="center">
<img src="assets/i2v-sample.png" alt="Input image, frame 0, 2 s and 4 s of the generated clip" width="100%">
</div>

| | Image → video | Text → video (same size) |
|---|---|---|
| Size / length | 768×512 · 4 s (auto-matched to the 768×512 input) | 768×512 · 4 s |
| Wall time | **68 s** | 61 s |
| Peak GPU memory | **22.5 GiB** | 22.5 GiB |
| Frame 0 vs input image | mean difference **2.6 / 255** (H.264 noise only) | n/a |

Image conditioning costs almost nothing extra: one more VAE encode of the still. Run on a quiet GPU
(no other jobs), `ltx_job.py` records it with `mode: i2v`.

## The GB10 (sm_121) Patch

Upstream builds its NVFP4 kernels only for arch-specific targets (`sm_100a`, `sm_110a`, `sm_120a`),
on purpose: the hardware FP32→E2M1 conversion (`cvt.rn.satfinite.e2m1x2.f32`) is only emitted for
`a` targets. Without it the quantize kernel falls back to software emulation and runs about 4× slower.
Arch-specific cubins are **not** forward-compatible across minor versions, so an `sm_120a` build
won't load on GB10's `sm_121`.

[`patches/ltx-kernels-sm121a.patch`](patches/ltx-kernels-sm121a.patch) adds `121a` to the arch list
and maps `TORCH_CUDA_ARCH_LIST=12.1` to it. The pip-installed `nvcc` 13.2 that ships with the env
supports `compute_121`. After building:

```text
$ LTX-2/.venv/bin/python -c "from ltx_kernels.nvfp4 import functional as f; print(f.unavailable_reason())"
None
```

A random 256×1024 · 1024×512 NVFP4 GEMM matches the BF16 reference to 13% relative error, which is
expected when both operands are quantized to 4 bits.

## LTX Video Studio (Web UI)

<div align="center">
<img src="assets/studio.png" alt="LTX Video Studio: gallery, activity rail with two parallel renders, composer with an image attached" width="100%">
</div>

A dark, cinematic studio served by a standard-library Python app (`webui/server.py`,
`webui/index.html`, `webui/imageinfo.py`). There's no build step and no npm.

- **Composer:** a floating prompt bar with aspect (landscape, portrait, square, *match image*),
  quality (Fast, HD, Max), length (2, 4, 5, 8 s), seed and **Ctrl/⌘ + Enter**.
- **Image → video:** attach with the button, drag onto the page or paste from the clipboard (PNG, JPEG,
  WebP up to 16 MB). Size defaults to *match image*, and a strength slider appears.
- **Activity rail:**
  - **Now rendering** shows a five-step tracker (Prompt → Pass 1 → Upscale → Pass 2 → Render) with
    elapsed time and an ETA. CLI jobs appear here too.
  - **Up next** is the numbered queue, with who asked and an estimated start time for each job.
  - **Finished** lists recent jobs (web and CLI) with status, render time and failure reasons;
    click one to play it.
  - ETAs use the median of your past jobs at the same size and length, or a fit to the benchmark
    sweep when there's no history.
- **Gallery:** a masonry feed that plays on hover, filtered by *All / Text → video / Image → video*,
  with the source-image thumbnail on clips made from images. Click a clip for a full-screen viewer with
  the prompt, render stats, **Reuse prompt**, **Animate this image again** and download.
- **Queue:** up to 10 waiting, started in order by 1 worker (or more; see [Parallel generation](#parallel-generation)).
- **Memory gate:** refuses to start a job with less than 50 GiB free (`LTX_MIN_FREE_GIB`) and tells
  you if the LLM container is the reason, instead of letting the box swap.
- **Who asked:** behind `tailscale serve`, each job is tagged with the requester's Tailscale login
  (`Tailscale-User-Login`). On the optional direct listener it uses `tailscale whois` instead.

### How to use the studio

1. **Open it.** On the Spark go to `http://127.0.0.1:8090`. From another device on your tailnet use
   `https://<host>.<tailnet>.ts.net/` (see the URL table below). The pill in the top-right corner
   should say **Ready**. If it says *Busy*, an LLM server is holding the memory; see
   [Sharing the Box](#sharing-the-box-with-an-llm).
2. **Describe the shot** in the bar at the bottom: subject, action, setting, light, camera move *and*
   sound. Example: *"A red fox trots through fresh snow in a birch forest at golden hour, soft wind,
   slow tracking shot"*.
3. **Optional: start from an image.** Click the picture button, drag a photo onto the page or paste
   one (PNG, JPEG or WebP, up to 16 MB). The photo becomes the first frame; describe how it should
   *move*. Shape switches to **Match image** so nothing gets cropped. Lower **Image strength** if you
   want the model to drift further from the photo.
4. **Pick the settings:**
   - **Shape:** Landscape, Portrait, Square, or Match image.
   - **Quality:** Fast (~768 px, about a minute for 4 s), HD (~1280 px) or Max (~1536 px, about
     3 min for 4 s).
   - **Length:** 2, 4, 5 or 8 s.
   - **Seed** (optional): reuse a seed with the same settings to reproduce a clip exactly.
5. **Generate** (or **Ctrl/⌘ + Enter**). The job appears in the Activity rail on the right:
   - **Now rendering** steps through *Prompt → Pass 1 → Upscale → Pass 2 → Render* with elapsed time
     and an ETA. With parallel workers you'll see one card per running job.
   - **Up next** lists queued jobs in order with an estimated start time. *Waiting for memory* means
     the job starts as soon as a running one frees enough.
   - **Finished** lists recent results (from the studio and the command line), with render time or the
     reason a job failed.
6. **Watch and reuse.** Finished clips appear in the gallery and play on hover. Click one (or a row
   in *Finished*) to open the viewer with sound, then **Download MP4**, **Reuse prompt** to iterate,
   or **Animate this image again** for image-to-video clips. Use the tabs to filter text-to-video and
   image-to-video clips.

Tip: iterate at **Fast · 4 s** (about a minute per try), then re-render the prompt and seed you like at
HD or Max.

### Parallel generation

The studio can render several jobs at once: set `LTX_WORKERS` (default `1`), e.g.
`LTX_WORKERS=2 ./setup-services.sh`. Each job peaks at ~23 GiB for the LTX process (~29 GiB
system-wide), so two or three fit when nothing else holds the memory.

- **Memory-safe start:** before starting a job, the studio subtracts the memory that already-running
  jobs haven't allocated yet (`LTX_JOB_PEAK_GIB`, default 30). A just-started job counts at its full
  peak, so two workers can't both see "plenty free" at the same moment and overshoot. If there isn't
  room, the job waits in the queue instead of failing.
- **What to expect:** all jobs share one GPU, and the denoising passes already keep it busy. Parallel
  jobs therefore each run slower; the gain comes from overlapping one job's model loading and text
  encoding with another's GPU work. This hasn't been benchmarked on the Spark yet, so measure your own
  mix before relying on it.
- Each running job writes its own `logs/running/<job_id>.json`, so the CLI, the studio and Grafana all
  see every parallel job. In the gallery each running job gets a live tile ("Rendering 1/2 · 83%"),
  followed by the queued jobs in order.
- **Restart-safe:** the queue is saved to `logs/webui_queue.json`, and the service uses
  `KillMode=process`, so restarting or updating the studio doesn't kill renders in progress. The new
  server adopts them (marking them done when they finish) and resumes the queue.
- **Pause switch:** `echo "Maintenance" > logs/webui_pause` stops new jobs from starting (running ones
  finish; people can still queue, and the page shows the reason). `rm logs/webui_pause` resumes.
- **Settings:**

  | Variable | Default | Meaning |
  |---|---|---|
  | `LTX_WORKERS` | `1` | Jobs rendered at once |
  | `LTX_MIN_FREE_GIB` | `50` | Free memory required to start a job (~29 GiB peak + ~20 GiB no-swap margin) |
  | `LTX_JOB_PEAK_GIB` | `30` | Memory reserved for a running job that hasn't fully loaded yet |

### Publishing to your tailnet

It binds to `127.0.0.1` only. `setup-services.sh --tailnet` publishes it at:

| URL | Path |
|---|---|
| `https://<host>.<tailnet>.ts.net/` | `tailscale serve` (HTTPS, auto cert) |
| `http://<host>:8090` | `tailscale serve --http 8090` (MagicDNS short name) |
| `http://<tailscale-ip>:8091` | Direct listener for clients without MagicDNS (serve routes by hostname, so a bare IP returns 404 there) |

All three are reachable from the tailnet only.

## Monitoring with Grafana

<div align="center">
<img src="assets/grafana.png" alt="LTX-2.5 Grafana dashboard" width="100%">
</div>

`ltx_job.py` wraps every pipeline run. Once a second it rewrites `logs/current.json` (stage, step,
GPU memory from `nvidia-smi`), and when the run ends it appends a record to `logs/jobs.jsonl`.
`monitoring/ltx_exporter.py` turns both into Prometheus metrics:

| Metric | Meaning |
|---|---|
| `ltx_job_running`, `ltx_job_stage{stage}` | How many jobs are running (parallel workers), and how many are in each stage |
| `ltx_job_step` / `ltx_job_steps` | Denoising progress in the current pass |
| `ltx_job_gpu_memory_bytes` | Live unified-memory allocation of the LTX process |
| `ltx_jobs_total{status}`, `ltx_jobs_by_user_total{user,source,status}` | Finished jobs, by requester and entry point (`cli` / `webui`) |
| `ltx_last_job_{duration_seconds,peak_gpu_memory_bytes,realtime_factor}` | Last job stats |
| `ltx_jobs_by_mode_total{mode,status}`, `ltx_job_mode{mode}` | Text-to-video (`t2v`) vs image-to-video (`i2v`) |
| `ltx_video_seconds_generated_total` | Seconds of video produced |
| `ltx_webui_up`, `ltx_webui_queue_jobs{state}`, `ltx_webui_rejected_low_memory` | Web UI health and queue |

The dashboard (`monitoring/grafana-dashboard.json`, 32 panels) combines these with
`dcgm-exporter` (GPU utilization, power, temperature) and `node-exporter` (unified memory).
Regenerate it with `python3 monitoring/make_dashboard.py [out.json]`, and add
[`monitoring/prometheus-scrape.yml`](monitoring/prometheus-scrape.yml) to your Prometheus config.

## Sharing the Box with an LLM

This Spark also serves an LLM ([Ornith-1.5-35B-A3B](https://github.com/Hitheshkaranth/Ornith-1.5_A3B_Model_DGX_Spark_Setup)
on vLLM). vLLM reserves memory up front, so the two only fit together if you shrink the LLM's reservation:

| | vLLM at `--gpu-memory-utilization 0.70` | vLLM at `0.45` |
|---|---|---|
| LLM reservation | ~85 GiB | ~55 GiB |
| LTX peak (system-wide) | ~29 GiB | ~29 GiB |
| Desktop session | ~8–10 GiB | ~8–10 GiB |
| **Total** | **~125 GiB: doesn't fit** | **~94 GiB: fits with ~27 GiB spare** |
| LLM KV cache | ~5.1M tokens (≈20 × 262K), measured | ~2.4M tokens (≈9 × 262K), estimated |

*The `0.70` column is measured; the `0.45` column is an estimate from the same numbers and hasn't
been run side by side yet.*

On unified memory, oversubscription doesn't fail cleanly: the kernel swaps and both workloads crawl.
Keep about 20 GiB available at all times. Both models also share one GPU, so LLM decoding slows
while a clip renders. This deployment runs them one at a time (stop the LLM, generate, restart it),
and the studio's memory gate enforces that.

## Benchmarks

Text-to-video, fixed prompt and seed (`benchmarks/bench.py`), submitted through the web UI queue on an
otherwise idle Spark. Wall time **includes loading every model from disk** (page-cached after the
first run): this is the real time from pressing *Generate* to having an MP4.

<div align="center">
<img src="benchmarks/generation_benchmark.png" alt="Generation time and memory" width="100%">
</div>

| Size | Clip length | Frames | Wall time | Compute s per video s | Peak GPU memory (LTX process) |
|---|---|---|---|---|---|
| 768×512 | 2 s | 49 | **44 s** | 21.6 | 22.6 GiB |
| 768×512 | 4 s | 97 | **61 s** | 15.1 | 22.5 GiB |
| 768×512 | 5 s | 121 | **70 s** | 13.8 | 22.6 GiB |
| 768×512 | 8 s | 193 | **94 s** | 11.7 | 23.3 GiB |
| 1024×576 | 4 s | 97 | **78 s** | 19.2 | 23.0 GiB |
| 1280×704 | 4 s | 97 | **107 s** | 26.5 | 23.0 GiB |
| 1536×1024 | 2 s | 49 | **97 s** | 47.5 | 22.9 GiB |
| 1536×1024 | 4 s | 97 | **165 s** | 40.8 | 23.0 GiB |

**Takeaways**

- **Longer clips are cheaper per second.** At 768×512, a 2 s clip takes 44 s and an 8 s clip 94 s: 4× the video for about 2.1× the time. Model loading and the text encoder are a fixed cost per job, so the per-second cost drops from 21.6 to 11.7.
- **Resolution is what costs.** At 4 s, going from 768×512 to 1536×1024 (4× the pixels) raises wall time from 61 s to 165 s.
- **Memory is flat at ~22.5–23.3 GiB** across every size and length tested. Stages load and free their models one at a time, so the model weights set the peak, not the video size. Even 1536×1024 fits easily.
- **For quick iteration**, use 768×512 at 4–5 s (about a minute), then re-render the prompt you like at a higher resolution.

Reproduce with `python3 benchmarks/bench.py && LTX-2/.venv/bin/python benchmarks/make_charts.py`
(with the studio at `LTX_WORKERS=1`, the setting these numbers were measured with).

### Concurrency benchmark

Does rendering several clips at once finish a batch sooner on one GB10? `benchmarks/parallel_bench.py`
renders the same 4 prompts (fixed seeds) with 1, 2 and 3 jobs in flight at 768×512 · 4 s, and with 1 and
2 at 1280×704 · 4 s. For each run it records:

- **Makespan:** wall time for the whole batch.
- **Throughput** in clips per minute.
- Per-job time and peak GPU memory (from `ltx_job.py`).
- **Lowest free memory and swap traffic** during the batch.
- Average GPU utilization.
- **Contention:** how many *other* jobs (studio users) overlapped it. Rows with `contended: true` were
  measured on a shared GPU and aren't clean scaling numbers.

Every job start passes the same memory gate as the studio, so the benchmark is safe to run while
people keep generating. For clean numbers, pause the studio first.

```bash
echo "Benchmark running" > logs/webui_pause      # optional: clean numbers
python3 benchmarks/parallel_bench.py              # ~25 min on a quiet GPU
LTX-2/.venv/bin/python benchmarks/make_parallel_chart.py
rm -f logs/webui_pause
```

<!-- PARALLEL_RESULTS -->
*The first full run is in progress on the live, shared box; the results table and chart land here when
it completes.*
<!-- /PARALLEL_RESULTS -->

## Repository Layout

```text
.
├── install.sh                 # clone LTX-2 @ v1.4.2, apply patch, build env + kernels, download weights
├── download.sh                # the 5 weight files (~44 GiB) from Lightricks/LTX-2.5
├── run.sh                     # text-to-video CLI (NVFP4 distilled pipeline)
├── i2v.sh                     # image-to-video CLI: sizes output to the image's aspect ratio
├── ltx_job.py                 # wraps each run: live stage/progress/memory + job history
├── setup-services.sh          # systemd --user units for web UI + exporter, optional tailscale serve
├── patches/
│   ├── ltx-kernels-sm121a.patch   # GB10 NVFP4 kernel build fix
│   └── uv.lock                    # exact tested dependency set (torch 2.13+cu132, natten 0.21.7, …)
├── webui/
│   ├── server.py              # LTX Video Studio API: queue, uploads, memory gate, history, range-request video
│   ├── index.html             # single-page studio: composer, activity rail, gallery, viewer
│   └── imageinfo.py           # stdlib PNG/JPEG/WebP size + EXIF orientation sniffing, aspect-matched sizes
├── monitoring/
│   ├── ltx_exporter.py        # Prometheus exporter (:9092)
│   ├── make_dashboard.py      # generates the Grafana dashboard JSON
│   ├── grafana-dashboard.json
│   └── prometheus-scrape.yml
├── benchmarks/
│   ├── bench.py · make_charts.py · results.json · generation_benchmark.png        # size/length sweep
│   ├── parallel_bench.py · make_parallel_chart.py · parallel_results.json             # 1 vs 2 vs 3 jobs at once
└── assets/                    # banner, sample GIFs, screenshots
```

Created at runtime and git-ignored: `LTX-2/` (patched upstream), `models/`, `outputs/`, `uploads/` (studio images), and `logs/`
(job history includes prompts and requester logins).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `403` downloading weights | Click *Agree* on [the model page](https://huggingface.co/Lightricks/LTX-2.5), then `hf auth login`. The device-code flow works on a headless box. |
| `nvfp4 unavailable: ... rebuild` | The kernels weren't built for your GPU. Re-run `install.sh`, or `cd LTX-2 && TORCH_CUDA_ARCH_LIST=12.1 uv sync --extra natten --group dev --group kernels` |
| `--distilled-checkpoint-path` errors with the split files | Use `--transformer-path` (as `run.sh` does); the NVFP4 file is the split layout |
| Studio says *Not enough free memory* | Something else holds unified memory (usually an LLM server). Stop it or lower its `--gpu-memory-utilization`; see [Sharing the Box](#sharing-the-box-with-an-llm) |
| Box crawls, swap climbs during a job | Same cause: available memory dropped below ~20 GiB. Never run both at full reservation. |
| `tailscale serve`: *Serve is not enabled on your tailnet* | Open the printed `login.tailscale.com/f/serve?...` link as tailnet admin and enable it |
| `tailscale serve`: *Access denied* | `sudo tailscale set --operator=$USER` once |
| Tailnet HTTPS URL times out on first visit | The first request provisions the TLS certificate; give it a minute |
| A device can't resolve `<host>.ts.net` | MagicDNS is off on that device. Use `http://<tailscale-ip>:8091` (`setup-services.sh --tailnet` sets this up) |
| Image-to-video crops off the subject | Use *Match image* in the studio, or let `i2v.sh` pick the size; a fixed size center-crops the photo |
| `ffmpeg` not installed | Not needed: the pipeline encodes through PyAV, which ships its own FFmpeg |

## Credits & License

- **Model and inference code:** [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) and
  [LTX-2.5 weights](https://huggingface.co/Lightricks/LTX-2.5), under the
  [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x). This repo
  doesn't redistribute either; `install.sh` fetches them from the source. Read the license before
  commercial use.
- **Text encoder:** Gemma 4 12B as fine-tuned and bundled by Lightricks (Gemma terms apply).
- **Scripts, web UI, monitoring and patch in this repo:** Apache License 2.0.
- Built and measured on an NVIDIA DGX Spark. Sample clips in `assets/` were generated by this setup.
