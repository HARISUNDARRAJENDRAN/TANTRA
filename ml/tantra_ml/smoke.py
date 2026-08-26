from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .text.vocabulary import build_vocabulary, Vocabulary


def main() -> None:
    texts = [
        "मदद की आवश्यकता है", "સહાયની જરૂર છે", "मदत हवी आहे", "ಸಹಾಯ ಬೇಕಾಗಿದೆ",
        "സഹായം ആവശ്യമാണ്", "உதவி தேவை", "సహాయం కావాలి", "ସାହାଯ୍ୟ ଦରକାର",
        "সাহায্য প্রয়োজন", "help is required",
    ]
    vocabulary = build_vocabulary(texts)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "vocab.json"
        vocabulary.save(path)
        reloaded = Vocabulary.load(path)
        assert all(reloaded.decode(reloaded.encode(text)) == text for text in texts)
    result = {"languages": 10, "vocabulary_tokens": len(vocabulary.tokens), "round_trip": True}
    try:
        import torch
        from .asr.model import AsrConfig, TantraAsr

        model = TantraAsr(AsrConfig(vocab_size=len(vocabulary.tokens), model_dim=64, layers=2, heads=4, feed_forward_dim=128, conv_channels=(16, 32, 48, 64)))
        samples = torch.randn(2, 16_000)
        logits, lengths = model(samples, torch.tensor([16_000, 14_000]), torch.tensor([1, 10]))
        assert logits.shape[0] == 2 and logits.shape[-1] == len(vocabulary.tokens)
        result.update({"asr_shape": list(logits.shape), "parameters": model.parameter_count(), "output_lengths": lengths.tolist()})
    except ImportError:
        result["torch"] = "not installed; vocabulary smoke passed"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
