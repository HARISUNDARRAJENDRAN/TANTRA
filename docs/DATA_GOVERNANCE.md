# Data governance

1. Use only datasets whose license and consent permit the intended training and redistribution.
2. Keep source audio outside Git. Store immutable manifests with source, revision, checksum, license, speaker split, language, and normalization version.
3. Do not mix a benchmark test split into training, pseudo-label generation, vocabulary construction, or threshold tuning.
4. Remove direct identifiers and audit accidental PII in transcripts.
5. Track speaker overlap across train/dev/test, especially when combining corpora.
6. Publish per-language hours and demographic limitations rather than only a total-hours number.
7. Teacher-generated labels retain provenance and confidence; they do not overwrite human transcripts.
8. A model pack must include its training-data summary, model card, license inventory, and SHA-256 manifest.
