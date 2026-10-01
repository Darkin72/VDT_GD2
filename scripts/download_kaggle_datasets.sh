#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${DATA_ROOT:-/root/clearair-data}"
KAGGLE_FILE="${KAGGLE_FILE:-$HOME/.kaggle/kaggle.json}"
WORK_ROOT="${WORK_ROOT:-$DATA_ROOT/.downloads}"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }
[[ -f "$KAGGLE_FILE" ]] || { echo "Kaggle credentials not found: $KAGGLE_FILE" >&2; exit 1; }

mkdir -p "$DATA_ROOT" "$WORK_ROOT"
chmod 700 "$WORK_ROOT"
df -h "$DATA_ROOT"

NETRC="$(mktemp)"
trap 'rm -f "$NETRC"' EXIT
chmod 600 "$NETRC"
python3 - "$KAGGLE_FILE" "$NETRC" <<'PY'
import json
import sys
from pathlib import Path

credential_path, netrc_path = map(Path, sys.argv[1:])
credential = json.loads(credential_path.read_text())
username = credential.get("username")
key = credential.get("key")
if not username or not key:
    raise SystemExit("kaggle.json must contain username and key")
netrc_path.write_text(f"machine www.kaggle.com\nlogin {username}\npassword {key}\n")
PY

download_and_extract() {
  local slug="$1" name="$2"
  local destination="$DATA_ROOT/$name" archive="$WORK_ROOT/$name.zip"
  local url="https://www.kaggle.com/api/v1/datasets/download/$slug"
  mkdir -p "$destination"
  echo "=== $name: download ==="
  if [[ -f "$archive" ]]; then
    curl -fL --retry 5 --retry-delay 5 --progress-bar --netrc-file "$NETRC" -C - "$url" -o "$archive"
  else
    curl -fL --retry 5 --retry-delay 5 --progress-bar --netrc-file "$NETRC" "$url" -o "$archive"
  fi
  echo "=== $name: verify and extract ==="
  unzip -t "$archive" >/dev/null
  unzip -q -o "$archive" -d "$destination"
  find "$destination" -type d \( -name hazy -o -name clear \) -print
}

requested="${1:-all}"
case "$requested" in
  all)
    download_and_extract "balraj98/indoor-training-set-its-residestandard" ITS
    download_and_extract "brunobelloni/outdoor-training-set-ots-reside" OTS
    download_and_extract "balraj98/synthetic-objective-testing-set-sots-reside" SOTS-download
    ;;
  ITS|OTS|SOTS-download)
    case "$requested" in
      ITS) slug="balraj98/indoor-training-set-its-residestandard" ;;
      OTS) slug="brunobelloni/outdoor-training-set-ots-reside" ;;
      SOTS-download) slug="balraj98/synthetic-objective-testing-set-sots-reside" ;;
    esac
    download_and_extract "$slug" "$requested"
    ;;
  *)
    echo "Usage: $0 [all|ITS|OTS|SOTS-download]" >&2
    exit 2
    ;;
esac
echo "All Kaggle datasets downloaded and extracted."
du -sh "$DATA_ROOT"/* 2>/dev/null || true
