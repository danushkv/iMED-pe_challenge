#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIGHTGLUE_COMMIT="eb42fee2d71449efb0aa5c10549752b5d75384d8"
XFEAT_COMMIT="e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
LOFTR_COMMIT="ffd4a4644064354468eb1f0c7a3e732233cb732f"
VGGT_COMMIT="a288dd0f14786c93483e45524328726ab7b1b4ce"

clone_revision() {
  local repository="$1"
  local revision="$2"
  local destination="$3"
  if [[ -d "$destination/.git" ]]; then
    echo "Using existing checkout: $destination"
    local actual
    actual="$(git -C "$destination" rev-parse HEAD)"
    [[ "$actual" == "$revision" ]] || {
      echo "Expected $revision but found $actual in $destination" >&2
      return 1
    }
    return
  fi
  git init "$destination"
  git -C "$destination" remote add origin "$repository"
  git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach FETCH_HEAD
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]]
}

verify_sha256() {
  local expected="$1"
  local path="$2"
  [[ -f "$path" ]] || {
    echo "Missing checkpoint: $path" >&2
    return 1
  }
  echo "$expected  $path" | sha256sum -c -
}

setup_lightglue() {
  clone_revision \
    https://github.com/cvg/LightGlue.git \
    "$LIGHTGLUE_COMMIT" \
    "$ROOT_DIR/third_party/LightGlue"
  echo "LightGlue is pinned by uv; run 'uv sync' from the repository root."
}

setup_xfeat() {
  clone_revision \
    https://github.com/verlab/accelerated_features.git \
    "$XFEAT_COMMIT" \
    "$ROOT_DIR/third_party/accelerated_features"
  verify_sha256 \
    0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b \
    "$ROOT_DIR/third_party/accelerated_features/weights/xfeat.pt"
}

setup_loftr() {
  clone_revision \
    https://github.com/zju3dv/EfficientLoFTR.git \
    "$LOFTR_COMMIT" \
    "$ROOT_DIR/third_party/EfficientLoFTR"
  local checkpoint="$ROOT_DIR/third_party/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "EfficientLoFTR source is ready, but its checkpoint is not stored in Git."
    echo "Download eloftr_outdoor.ckpt from the official upstream link/model card:"
    echo "  https://github.com/zju3dv/EfficientLoFTR"
    echo "  https://huggingface.co/zju-community/efficientloftr"
    echo "Place it at: $checkpoint"
    echo "Then rerun this command to verify its SHA-256."
    return
  fi
  verify_sha256 \
    0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878 \
    "$checkpoint"
}

setup_vggt() {
  clone_revision \
    https://github.com/facebookresearch/vggt.git \
    "$VGGT_COMMIT" \
    "$ROOT_DIR/third_party/vggt"
  echo "VGGT source is ready. The facebook/VGGT-1B checkpoint is loaded through"
  echo "the official Hugging Face API; cache it before offline inference."
}

usage() {
  echo "Usage: $0 [--all|--lightglue|--xfeat|--loftr|--vggt]"
}

[[ $# -gt 0 ]] || {
  usage
  exit 2
}

for option in "$@"; do
  case "$option" in
    --all)
      setup_lightglue
      setup_xfeat
      setup_loftr
      setup_vggt
      ;;
    --lightglue) setup_lightglue ;;
    --xfeat) setup_xfeat ;;
    --loftr) setup_loftr ;;
    --vggt) setup_vggt ;;
    *) usage; exit 2 ;;
  esac
done

