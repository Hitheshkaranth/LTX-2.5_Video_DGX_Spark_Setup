#!/usr/bin/env bash
# Download LTX-2.5 weights for the NVFP4 distilled pipeline (~44 GiB).
# The repo is gated with auto-approval: click "Agree" at https://huggingface.co/Lightricks/LTX-2.5,
# then log in once (`hf auth login` — the browser/device flow works on a headless box too).
set -euo pipefail
cd "$(dirname "$0")"
if command -v hf >/dev/null 2>&1; then HF=(hf); else HF=(uvx --from "huggingface_hub>=2.1" hf); fi
"${HF[@]}" download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir models/ltx-2.5
