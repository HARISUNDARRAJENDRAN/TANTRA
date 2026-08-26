# Third-party model and dataset register

No third-party weights are committed in this repository.

Before importing a model or dataset, add one row and archive its model card/license in the generated model pack.

| Asset | Upstream revision | License | Languages | Redistribution allowed? | Intended use | Reviewed by |
|---|---|---|---|---|---|---|
| _example_ | commit/SHA | SPDX id | list | yes/no/conditions | teacher/student/eval | name/date |

Suggested candidates to evaluate, not automatically approved:

- AI4Bharat IndicConformer / IndicWav2Vec family as ASR teachers.
- OpenAI Whisper or an open converted checkpoint as a multilingual baseline, subject to mobile latency.
- AI4Bharat IndicTTS or openly licensed VITS/Piper/MMS voices as TTS teachers.
- Mozilla Common Voice and Google FLEURS for reproducible held-out ASR evaluation, subject to each release's license and terms.

A GitHub or Hugging Face repository being public does not itself grant weight or dataset redistribution rights. The pack builder refuses a release pack unless license metadata is present.
