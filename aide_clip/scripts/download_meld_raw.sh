#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/meld}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$DATA_DIR/MELD.Raw.tar.gz}"
EXTRACT_DIR="${EXTRACT_DIR:-$DATA_DIR/raw}"
URL="${URL:-https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz}"

mkdir -p "$DATA_DIR" "$EXTRACT_DIR"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "[GET] $URL"
  wget --continue -O "$ARCHIVE_PATH" "$URL"
else
  echo "[SKIP] archive exists: $ARCHIVE_PATH"
fi

echo "[EXTRACT] $ARCHIVE_PATH -> $EXTRACT_DIR"
tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"

echo "[DONE] MELD raw archive is available"
echo "  archive: $ARCHIVE_PATH"
echo "  extracted: $EXTRACT_DIR"