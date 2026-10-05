#!/usr/bin/env bash
# Text-to-video with LTX-2.5 distilled (NVFP4 prequant transformer) on GB10.
# Usage: ./run.sh "prompt" [out.mp4] [extra pipeline args, e.g. --height 512 --width 768 --num-frames 49]
# Unified memory: Ornith (:8004) holds ~87 GiB; stop it first or this will swap/OOM.
set -euo pipefail
cd "$(dirname "$0")"
M=models/ltx-2.5
PROMPT="${1:?prompt required}"; OUT="${2:-outputs/$(date +%Y%m%d-%H%M%S).mp4}"; shift $(( $# >= 2 ? 2 : 1 ))
mkdir -p "$(dirname "$OUT")"
# --offload cpu buys nothing here: CPU and GPU share the same 121 GiB pool.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ltx_job.py records stage/progress/memory for the Grafana "LTX-2.5 Video Generation" dashboard.
exec python3 ltx_job.py "$OUT" -- LTX-2/.venv/bin/python -m ltx_pipelines.distilled \
  --quantization nvfp4-prequant \
  --transformer-path "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" \
  --text-encoder-path "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path "$M/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --audio-vae-path "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial-upsampler-path "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --seed 42 --output-path "$OUT" --prompt "$PROMPT" "$@"
