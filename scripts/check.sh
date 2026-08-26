#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p validation
: > validation/check.log
exec > >(tee -a validation/check.log) 2>&1

printf 'TANTRA validation started: %s\n' "$(date -u +%FT%TZ)"
python -m compileall -q protocol/python/tantra_protocol ml/tantra_ml
pytest -q protocol/python/tests ml/tests
python tools/benchmark_protocol.py

if command -v java >/dev/null 2>&1 && [[ -x android/gradlew ]]; then
  (cd android && ./gradlew --no-daemon testDebugUnitTest assembleDebug)
else
  echo 'Android build skipped: Java or Gradle wrapper unavailable.'
  exit 24
fi

if git grep -nE '(ak-[A-Za-z0-9]{12,}|as-[A-Za-z0-9]{12,})' -- . \
  ':!.git/*' ':!scripts/check.sh' ':!.github/workflows/ci.yml' ':!docs/SECURITY.md' >/dev/null 2>&1; then
  echo 'Potential credential pattern found in tracked content.'
  exit 25
fi

printf 'TANTRA validation passed: %s\n' "$(date -u +%FT%TZ)"
touch validation/PASS
