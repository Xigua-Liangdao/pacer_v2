#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-mosei}"
BASE_DIR="$ROOT/data"
WGET_OPTS=(--continue --tries=3 --timeout=30 --waitretry=5)

mkdir -p "$BASE_DIR/mosei/csd" "$BASE_DIR/mosi/csd"

download_one() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[SKIP] exists: $out"
    return 0
  fi
  echo "[GET] $url"
  wget "${WGET_OPTS[@]}" -O "$out" "$url"
}

case "$DATASET" in
  mosei)
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWords.csd" \
      "$BASE_DIR/mosei/csd/CMU_MOSEI_TimestampedWords.csd"
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd" \
      "$BASE_DIR/mosei/csd/CMU_MOSEI_VisualOpenFace2.csd"
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_LabelsSentiment.csd" \
      "$BASE_DIR/mosei/csd/CMU_MOSEI_LabelsSentiment.csd"
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_LabelsEmotions.csd" \
      "$BASE_DIR/mosei/csd/CMU_MOSEI_LabelsEmotions.csd"
    ;;
  mosi)
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWords.csd" \
      "$BASE_DIR/mosi/csd/CMU_MOSI_TimestampedWords.csd"
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/visual/CMU_MOSI_OpenFace2.csd" \
      "$BASE_DIR/mosi/csd/CMU_MOSI_OpenFace2.csd"
    download_one \
      "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd" \
      "$BASE_DIR/mosi/csd/CMU_MOSI_Opinion_Labels.csd"
    ;;
  all)
    "$0" mosei
    "$0" mosi
    ;;
  *)
    echo "[ERROR] dataset must be one of: mosei | mosi | all" >&2
    exit 2
    ;;
esac

echo "[DONE] downloaded official CSD subset for: $DATASET"