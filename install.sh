#!/usr/bin/env bash
# One-line installer for LTX-2.5 (NVFP4 distilled) on a DGX Spark (GB10).
#
#   curl -fsSL https://raw.githubusercontent.com/Hitheshkaranth/LTX-2.5_Video_DGX_Spark_Setup/main/install.sh | bash
#
# Clones this repo (or reuses it if run from inside), clones Lightricks/LTX-2 at the tested tag,
# applies the sm_121a NVFP4 patch, builds the Python env + CUDA kernels, and downloads the weights.
# Every step skips work that's already done, so it's safe to re-run.
set -euo pipefail

REPO_URL="${LTX_REPO_URL:-https://github.com/Hitheshkaranth/LTX-2.5_Video_DGX_Spark_Setup.git}"
DIR="LTX-2.5_Video_DGX_Spark_Setup"
LTX_TAG="v1.4.2"

echo "==> Checking prerequisites..."
for bin in git curl nvidia-smi; do
  command -v "$bin" >/dev/null 2>&1 || { echo "$bin is required." >&2; exit 1; }
done
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$cap" in
  12.1) ARCH=12.1 ;;            # GB10 / DGX Spark
  10.*|11.*|12.0) ARCH=$cap; echo "    Note: tested on GB10 (12.1); building for $cap." ;;
  *) echo "NVFP4 needs a Blackwell GPU (compute capability >= 10.0); found $cap." >&2; exit 1 ;;
esac
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ -f run.sh ] && [ -f ltx_job.py ]; then
  ROOT=$(pwd)
else
  if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone "$REPO_URL" "$DIR"; fi
  ROOT=$(cd "$DIR" && pwd)
fi
cd "$ROOT"

echo "==> Fetching Lightricks/LTX-2 @ $LTX_TAG..."
if [ ! -d LTX-2/.git ]; then
  git clone --branch "$LTX_TAG" --depth 1 https://github.com/Lightricks/LTX-2.git LTX-2
fi
if git -C LTX-2 apply --check ../patches/ltx-kernels-sm121a.patch 2>/dev/null; then
  git -C LTX-2 apply ../patches/ltx-kernels-sm121a.patch
  echo "    Applied sm_121a NVFP4 patch."
elif git -C LTX-2 apply --reverse --check ../patches/ltx-kernels-sm121a.patch 2>/dev/null; then
  echo "    sm_121a patch already applied."
else
  echo "sm_121a patch doesn't apply to LTX-2 (local edits?). Reset with: git -C LTX-2 checkout -- packages/ltx-kernels/setup.py" >&2
  exit 1
fi
[ -f LTX-2/uv.lock ] || cp patches/uv.lock LTX-2/uv.lock

echo "==> Building Python env + NVFP4 kernels (first run: ~10 min)..."
(cd LTX-2 && TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-8}" UV_PYTHON_PREFERENCE=only-managed \
  uv sync --extra natten --group dev --group kernels)
LTX-2/.venv/bin/python -c "from ltx_kernels.nvfp4 import functional as f; r=f.unavailable_reason(); print('    NVFP4 kernels:', r or 'OK'); raise SystemExit(bool(r))"

echo "==> Downloading weights (~44 GiB)..."
./download.sh

cat <<EOF

Done. Generate a test clip (needs ~30 GiB free unified memory):
  cd $ROOT && ./run.sh "A red fox trots through fresh snow at golden hour" outputs/fox.mp4 --height 512 --width 768 --num-frames 49

Optional web UI + metrics exporter:  ./setup-services.sh
Optional Grafana dashboard:          monitoring/stack/setup.sh   (Docker; asks you to choose a login)
EOF
