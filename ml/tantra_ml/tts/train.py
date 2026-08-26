from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml

from .dataset import TtsDataConfig, TtsDataset, build_speaker_map, collate_tts
from .model import TantraTts, TtsConfig
from ..manifest import read_manifest
from ..text.vocabulary import Vocabulary


def masked_l1(predicted: Tensor, target: Tensor, lengths: Tensor) -> Tensor:
    limit = min(predicted.shape[1], target.shape[1])
    predicted = predicted[:, :limit]
    target = target[:, :limit]
    mask = torch.arange(limit, device=lengths.device).unsqueeze(0) < lengths.clamp_max(limit).unsqueeze(1)
    return (predicted - target).abs().masked_select(mask.unsqueeze(-1)).mean()


def duration_loss(predicted_log: Tensor, target: Tensor, lengths: Tensor) -> Tensor:
    mask = torch.arange(target.shape[1], device=target.device).unsqueeze(0) < lengths.unsqueeze(1)
    expected = torch.log1p(target.float())
    return F.mse_loss(predicted_log.masked_select(mask), expected.masked_select(mask))


def multi_resolution_stft_loss(predicted: Tensor, target: Tensor, lengths: Tensor) -> Tensor:
    minimum = min(predicted.shape[1], target.shape[1])
    predicted = predicted[:, :minimum]
    target = target[:, :minimum]
    losses = []
    for fft, hop, window in ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)):
        if minimum < window:
            continue
        window_tensor = torch.hann_window(window, device=predicted.device)
        pred_stft = torch.stft(predicted, fft, hop, window, window_tensor, return_complex=True)
        target_stft = torch.stft(target, fft, hop, window, window_tensor, return_complex=True)
        pred_mag = pred_stft.abs().clamp_min(1e-5)
        target_mag = target_stft.abs().clamp_min(1e-5)
        spectral_convergence = (target_mag - pred_mag).norm() / target_mag.norm().clamp_min(1e-5)
        log_magnitude = F.l1_loss(pred_mag.log(), target_mag.log())
        losses.append(spectral_convergence + log_magnitude)
    if not losses:
        return F.l1_loss(predicted, target)
    return torch.stack(losses).mean()


def run(config_path: str | Path, resume: str | None = None) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 19))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = read_manifest(config["train_manifest"])
    dev_rows = read_manifest(config["dev_manifest"])
    vocabulary = Vocabulary.load(config["vocabulary"])
    speaker_map = build_speaker_map(rows + dev_rows)
    data_config = TtsDataConfig(**config.get("data", {}))
    train_data = TtsDataset(rows, vocabulary, speaker_map, data_config)
    dev_data = TtsDataset(dev_rows, vocabulary, speaker_map, data_config)
    loader = DataLoader(train_data, batch_size=int(config.get("batch_size", 4)), shuffle=True,
                        num_workers=int(config.get("workers", 4)), collate_fn=collate_tts, pin_memory=True)
    dev_loader = DataLoader(dev_data, batch_size=int(config.get("batch_size", 4)), shuffle=False,
                            num_workers=2, collate_fn=collate_tts)
    model_config = TtsConfig(
        vocab_size=len(vocabulary.tokens),
        speaker_count=len(speaker_map),
        mel_bins=data_config.mel_bins,
        hop_length=data_config.hop_length,
        sample_rate=data_config.sample_rate,
        **config.get("model", {}),
    )
    model = TantraTts(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 2e-4)), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best = float("inf")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1; best = float(state.get("best", best))
    run_dir = Path(config["output_dir"]) / (config.get("run_id") or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "speaker_map.json").write_text(json.dumps(speaker_map, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    for epoch in range(start_epoch, int(config.get("epochs", 100))):
        model.train(); total = 0.0
        for batch in loader:
            tokens = batch["tokens"].to(device)
            languages = batch["language_id"].to(device)
            speakers = batch["speaker_id"].to(device)
            durations = batch["durations"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                mel, log_duration, audio = model.forward_train(tokens, languages, speakers, durations)
                mel_loss = masked_l1(mel, batch["mel"].to(device), batch["frame_lengths"].to(device))
                dur_loss = duration_loss(log_duration, durations, batch["token_lengths"].to(device))
                wave_loss = multi_resolution_stft_loss(audio, batch["audio"].to(device), batch["audio_lengths"].to(device))
                loss = mel_loss + 0.25 * dur_loss + 0.5 * wave_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss)

        model.eval(); dev_total = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                tokens = batch["tokens"].to(device)
                durations = batch["durations"].to(device)
                mel, log_duration, audio = model.forward_train(
                    tokens, batch["language_id"].to(device), batch["speaker_id"].to(device), durations,
                )
                dev_total += float(masked_l1(mel, batch["mel"].to(device), batch["frame_lengths"].to(device)))
        dev_loss = dev_total / max(1, len(dev_loader))
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": model_config.to_dict(), "epoch": epoch, "best": min(best, dev_loss),
            "speaker_map": speaker_map,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if dev_loss < best:
            best = dev_loss; checkpoint["best"] = best; torch.save(checkpoint, run_dir / "best.pt")
        record = {"epoch": epoch, "train_loss": total / max(1, len(loader)), "dev_mel_l1": dev_loss}
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(run(args.config, args.resume))


if __name__ == "__main__":
    main()
