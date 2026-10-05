#!/usr/bin/env bash
# Image-to-video with LTX-2.5: animate a still (PNG/JPEG/WebP) from a text prompt.
# Usage: ./i2v.sh IMAGE "prompt" [out.mp4] [extra pipeline args, e.g. --num-frames 121]
#
# The output size follows the image's aspect ratio (the pipeline center-crops the image to the
# output size). LTX_TIER=fast|hd|max picks the resolution tier (default fast: ~768x512);
# passing --width/--height yourself overrides it.
# LTX_IMAGE_STRENGTH (default 1.0): 1.0 keeps the first frame identical to the image;
# lower values let the model drift from it.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${1:?image required}"; PROMPT="${2:?prompt required}"
OUT="${3:-outputs/$(date +%Y%m%d-%H%M%S)-i2v.mp4}"; shift $(( $# >= 3 ? 3 : 2 ))
[ -f "$IMAGE" ] || { echo "No such image: $IMAGE" >&2; exit 1; }
SIZE=()
if [[ " $* " != *" --width "* && " $* " != *" --height "* ]]; then
  read -r W H < <(python3 webui/imageinfo.py "$IMAGE" "${LTX_TIER:-fast}")
  SIZE=(--width "$W" --height "$H")
  echo "Output size ${W}x${H} (matches image aspect, tier ${LTX_TIER:-fast})"
fi
exec ./run.sh "$PROMPT" "$OUT" "${SIZE[@]}" --image "$(realpath "$IMAGE")" 0 "${LTX_IMAGE_STRENGTH:-1.0}" "$@"
