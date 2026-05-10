#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/yanjing/AGDiff/bin/python}"
DATASET="${1:-all}"
VARIANT="${2:-aligned}"
MMSA_GDRIVE_URL="${MMSA_GDRIVE_URL:-https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing}"
TMP_DIR="${TMP_DIR:-$ROOT/data/_downloads/mmsa_drive}"

case "$DATASET" in
  all|mosei|mosi) ;;
  *)
    echo "[ERROR] dataset must be one of: all | mosei | mosi" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  aligned)
    FILE_NAME="aligned_50.pkl"
    ;;
  unaligned)
    FILE_NAME="unaligned_50.pkl"
    ;;
  *)
    echo "[ERROR] variant must be: aligned | unaligned" >&2
    exit 2
    ;;
esac

declare -A EXPECTED_SHA
EXPECTED_SHA["mosei:aligned_50.pkl"]="45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791"
EXPECTED_SHA["mosei:unaligned_50.pkl"]="ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
EXPECTED_SHA["mosi:aligned_50.pkl"]="d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9"
EXPECTED_SHA["mosi:unaligned_50.pkl"]="78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524"

need_download=0
want_datasets=()
if [[ "$DATASET" == "all" ]]; then
  want_datasets+=("mosei" "mosi")
else
  want_datasets+=("$DATASET")
fi

mkdir -p "$ROOT/data/mosei" "$ROOT/data/mosi" "$TMP_DIR"

verify_one() {
  local dataset="$1"
  local target="$ROOT/data/$dataset/$FILE_NAME"
  local expected="${EXPECTED_SHA["$dataset:$FILE_NAME"]}"
  if [[ ! -f "$target" ]]; then
    return 1
  fi
  local actual
  actual="$(sha256sum "$target" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[WARN] checksum mismatch: $target"
    echo "       expected: $expected"
    echo "       actual:   $actual"
    return 1
  fi
  echo "[OK] verified: $target"
  return 0
}

copy_one() {
  local dataset="$1"
  local upper
  upper="$(printf '%s' "$dataset" | tr '[:lower:]' '[:upper:]')"
  local found
  found="$(find "$TMP_DIR" -type f -path "*/$upper/Processed/$FILE_NAME" | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "[ERROR] could not locate $upper/Processed/$FILE_NAME under $TMP_DIR" >&2
    return 1
  fi
  cp -f "$found" "$ROOT/data/$dataset/$FILE_NAME"
  echo "[COPY] $found -> $ROOT/data/$dataset/$FILE_NAME"
}

for dataset in "${want_datasets[@]}"; do
  if ! verify_one "$dataset"; then
    need_download=1
  fi
done

if [[ "$need_download" -eq 1 ]]; then
  echo "[INFO] missing or invalid files detected, downloading MMSA folder snapshot..."
  "$PYTHON_BIN" -m gdown --folder --remaining-ok --continue -O "$TMP_DIR/" "$MMSA_GDRIVE_URL"
fi

for dataset in "${want_datasets[@]}"; do
  if ! verify_one "$dataset"; then
    copy_one "$dataset"
    verify_one "$dataset"
  fi
done

echo "[DONE] requested datasets are ready"
for dataset in "${want_datasets[@]}"; do
  echo "  - $ROOT/data/$dataset/$FILE_NAME"
done