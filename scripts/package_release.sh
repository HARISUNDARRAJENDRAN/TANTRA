#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/release}"
mkdir -p "$OUT"
cd "$ROOT"
VERSION="$(git describe --always --dirty 2>/dev/null || date -u +%Y%m%d%H%M%S)"
ARCHIVE="$OUT/tantra-neural-transceiver-${VERSION}.tar.gz"
git archive --format=tar.gz --prefix=tantra-neural-transceiver/ -o "$ARCHIVE" HEAD
printf '%s\n' "$ARCHIVE"
