# Device-run artifacts

Create one immutable directory per candidate release. Include:

- APK SHA-256 and model-pack SHA-256;
- phone model, SoC, RAM, Android version and thermal state;
- dataset manifest hashes and evaluator protocol;
- raw telemetry JSONL conforming to `../telemetry.schema.json`;
- WER/CER output, TTS listening-sheet results, Android memory/CPU/battery captures;
- transport configuration, distance, interference and injected loss;
- a signed `RESULTS.md` that separates measured results from targets.

Do not commit personally identifiable speech unless its license and consent explicitly permit redistribution.
