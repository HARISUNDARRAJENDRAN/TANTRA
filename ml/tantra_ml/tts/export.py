from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import TantraTts, TtsConfig


def export_tts(checkpoint_path: str | Path, output_path: str | Path, opset: int = 18) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TtsConfig(**checkpoint["model_config"])
    model = TantraTts(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = torch.tensor([[2, 5, 8, 3]], dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)
    language = torch.tensor([1], dtype=torch.long)
    speaker = torch.tensor([0], dtype=torch.long)
    speed = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(
        model,
        (tokens, lengths, language, speaker, speed),
        output_path,
        input_names=["tokens", "token_lengths", "language_id", "speaker_id", "speed"],
        output_names=["audio"],
        dynamic_axes={"tokens": {1: "tokens"}, "audio": {1: "samples"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    print(export_tts(args.checkpoint, args.output, args.opset))


if __name__ == "__main__":
    main()
