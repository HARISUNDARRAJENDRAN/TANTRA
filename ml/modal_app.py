from __future__ import annotations

import json
from pathlib import Path
import subprocess

import modal

APP_NAME = "tantra-neural-transceiver"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch>=2.4", "torchaudio>=2.4", "onnx>=1.16", "onnxruntime-gpu>=1.20",
        "jiwer>=3.0", "numpy>=1.26,<3", "regex>=2024.5", "PyYAML>=6.0",
        "soundfile>=0.12", "tqdm>=4.66", "safetensors>=0.4",
    )
    .add_local_dir(Path(__file__).parent, remote_path="/root/ml", copy=True)
)

data_volume = modal.Volume.from_name("tantra-data", create_if_missing=True)
artifact_volume = modal.Volume.from_name("tantra-artifacts", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=900)
def gpu_smoke() -> dict:
    import torch
    from tantra_ml.asr.model import AsrConfig, TantraAsr

    device = torch.device("cuda")
    model = TantraAsr(AsrConfig(
        vocab_size=128, model_dim=96, layers=2, heads=4, feed_forward_dim=256,
        conv_channels=(24, 48, 72, 96),
    )).to(device).eval()
    samples = torch.randn(1, 32_000, device=device)
    with torch.inference_mode():
        logits, lengths = model(samples, torch.tensor([32_000], device=device), torch.tensor([1], device=device))
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "parameters": model.parameter_count(),
        "output": list(logits.shape),
        "lengths": lengths.tolist(),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
)
def train_asr(config_name: str = "baseline", resume: str = "") -> str:
    command = ["python", "-m", "tantra_ml.asr.train", "--config", f"/root/ml/configs/{config_name}.yaml"]
    if resume:
        command += ["--resume", resume]
    subprocess.run(command, cwd="/root/ml", check=True)
    artifact_volume.commit()
    return "ASR training finished; inspect /artifacts/asr-runs"


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
)
def export_asr(run_id: str, quantize: bool = True) -> str:
    checkpoint = f"/artifacts/asr-runs/{run_id}/best.pt"
    output = f"/artifacts/asr-runs/{run_id}/asr.onnx"
    command = ["python", "-m", "tantra_ml.asr.export", "--checkpoint", checkpoint, "--output", output]
    if quantize:
        command += ["--quantized-output", f"/artifacts/asr-runs/{run_id}/asr-int8.onnx"]
    subprocess.run(command, cwd="/root/ml", check=True)
    artifact_volume.commit()
    return output


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
)
def benchmark_asr(model_path: str, manifest_path: str, vocabulary_path: str, output_path: str) -> dict:
    from tantra_ml.asr.evaluate import evaluate

    report = evaluate(model_path, manifest_path, vocabulary_path, output_path)
    artifact_volume.commit()
    return report["languages"]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
)
def train_tts(config_name: str = "tts-baseline", resume: str = "") -> str:
    command = ["python", "-m", "tantra_ml.tts.train", "--config", f"/root/ml/configs/{config_name}.yaml"]
    if resume:
        command += ["--resume", resume]
    subprocess.run(command, cwd="/root/ml", check=True)
    artifact_volume.commit()
    return "TTS training finished; inspect /artifacts/tts-runs"


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
)
def align_tts(asr_model: str, source_manifest: str, vocabulary: str, output_manifest: str) -> str:
    subprocess.run([
        "python", "-m", "tantra_ml.tts.align_manifest",
        "--asr-model", asr_model, "--manifest", source_manifest,
        "--vocabulary", vocabulary, "--output", output_manifest,
    ], cwd="/root/ml", check=True)
    data_volume.commit()
    return output_manifest


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={"/artifacts": artifact_volume},
)
def export_tts(run_id: str) -> str:
    output = f"/artifacts/tts-runs/{run_id}/tts.onnx"
    subprocess.run([
        "python", "-m", "tantra_ml.tts.export",
        "--checkpoint", f"/artifacts/tts-runs/{run_id}/best.pt", "--output", output,
    ], cwd="/root/ml", check=True)
    artifact_volume.commit()
    return output


@app.local_entrypoint()
def main(action: str = "smoke", value: str = "") -> None:
    if action == "smoke":
        print(json.dumps(gpu_smoke.remote(), indent=2))
    elif action == "train-asr":
        print(train_asr.remote(value or "baseline"))
    elif action == "export-asr":
        if not value:
            raise SystemExit("Pass --value <run-id>")
        print(export_asr.remote(value))
    elif action == "train-tts":
        print(train_tts.remote(value or "tts-baseline"))
    elif action == "export-tts":
        if not value:
            raise SystemExit("Pass --value <run-id>")
        print(export_tts.remote(value))
    else:
        raise SystemExit(f"Unknown action: {action}")
