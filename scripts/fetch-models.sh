#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p public/models

OIM="https://github.com/ankandrew/open-image-models/releases/download/assets"
OCR="https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates"

fetch() { # url dest
  if [ ! -s "$2" ]; then
    echo "downloading $(basename "$2")..."
    curl -fL --retry 3 -o "$2" "$1"
  else
    echo "already present: $(basename "$2")"
  fi
  # sanity: model files must be > 1MB
  [ "$(wc -c < "$2")" -gt 1000000 ] || { echo "ERROR: $2 too small"; exit 1; }
}

fetch "$OIM/yolo-v9-t-384-license-plates-end2end.onnx" public/models/yolo-v9-t-384-license-plates-end2end.onnx
fetch "$OIM/yolo-v9-t-512-license-plates-end2end.onnx" public/models/yolo-v9-t-512-license-plates-end2end.onnx
fetch "$OCR/cct_xs_v2_global.onnx" public/models/cct_xs_v2_global.onnx
fetch "$OCR/cct_s_v2_global.onnx" public/models/cct_s_v2_global.onnx

# ONNX Runtime web assets are NOT copied to public/ — the app imports them
# with Vite `?url` imports (see src/web/ort-web.ts), which serves them in dev
# and bundles them into dist/ for prod, same-origin (never a CDN).
echo "OK"
