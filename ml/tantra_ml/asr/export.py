from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from .model import AsrConfig, TantraAsr


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: TantraAsr) -> None:
        super().__init__()
        self.model = model

    def forward(self, samples: torch.Tensor, sample_lengths: torch.Tensor, language_id: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(samples, sample_lengths, language_id)
        return logits


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path, opset: int = 18) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = AsrConfig.from_dict(checkpoint["model_config"])
    model = TantraAsr(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    wrapper = ExportWrapper(model)
    samples = torch.zeros(1, 16_000 * 3, dtype=torch.float32)
    lengths = torch.tensor([samples.shape[1]], dtype=torch.long)
    language = torch.tensor([1], dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (samples, lengths, language),
        output_path,
        input_names=["samples", "sample_lengths", "language_id"],
        output_names=["logits"],
        dynamic_axes={
            "samples": {1: "samples"},
            "logits": {1: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "model_config": config.to_dict(),
        "parameters": model.parameter_count(),
        "opset": opset,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def quantize_dynamic(model_path: str | Path, output_path: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic as ort_quantize

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ort_quantize(
        str(model_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"MatMulConstBOnly": True},
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantized-output")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    exported = export_checkpoint(args.checkpoint, args.output, args.opset)
    if args.quantized_output:
        quantize_dynamic(exported, args.quantized_output)
    print(exported)


if __name__ == "__main__":
    main()
