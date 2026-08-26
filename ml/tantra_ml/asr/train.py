from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .dataset import AsrDataset, collate_asr, language_balanced_sampler
from .decode import greedy_ctc_text
from .model import AsrConfig, TantraAsr
from ..manifest import read_manifest
from ..text.vocabulary import Vocabulary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max(1e-6, step / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model: TantraAsr, loader: DataLoader, vocabulary: Vocabulary, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0.0
    examples = 0
    previews: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            samples = batch["samples"].to(device)
            lengths = batch["sample_lengths"].to(device)
            languages = batch["language_id"].to(device)
            logits, output_lengths = model(samples, lengths, languages)
            loss = loss_fn(
                logits.log_softmax(-1).transpose(0, 1),
                batch["tokens"].to(device),
                output_lengths,
                batch["token_lengths"].to(device),
            )
            total_loss += float(loss) * samples.shape[0]
            examples += samples.shape[0]
            if len(previews) < 12:
                hypotheses = greedy_ctc_text(logits.cpu(), output_lengths.cpu(), vocabulary)
                previews.extend(
                    {"reference": reference, "hypothesis": hypothesis}
                    for reference, hypothesis in zip(batch["texts"], hypotheses, strict=True)
                )
    return {"loss": total_loss / max(1, examples), "previews": previews[:12]}


def run(config_path: str | Path, resume: str | None = None) -> Path:
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 17)))
    run_root = Path(config["output_dir"])
    run_id = config.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    vocabulary = Vocabulary.load(config["vocabulary"])
    train_rows = read_manifest(config["train_manifest"])
    dev_rows = read_manifest(config["dev_manifest"])
    data_config = config.get("data", {})
    train_data = AsrDataset(train_rows, vocabulary, max_seconds=float(data_config.get("max_seconds", 15)), training=True)
    dev_data = AsrDataset(dev_rows, vocabulary, max_seconds=float(data_config.get("max_seconds", 15)), training=False)
    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=language_balanced_sampler(train_rows),
        num_workers=int(config.get("workers", 4)),
        collate_fn=collate_asr,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.get("workers", 4)) > 0,
    )
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_asr)

    model_config = AsrConfig(vocab_size=len(vocabulary.tokens), **config.get("model", {}))
    model = TantraAsr(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=(0.9, 0.98),
    )
    epochs = int(config.get("epochs", 30))
    accumulation = int(config.get("gradient_accumulation", 1))
    total_steps = math.ceil(len(train_loader) / accumulation) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total_steps, int(config.get("warmup_steps", 1000))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    start_epoch = 0
    best_dev = float("inf")
    global_step = 0

    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dev = float(checkpoint.get("best_dev", best_dev))
        global_step = int(checkpoint.get("global_step", 0))

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            samples = batch["samples"].to(device, non_blocking=True)
            sample_lengths = batch["sample_lengths"].to(device, non_blocking=True)
            languages = batch["language_id"].to(device, non_blocking=True)
            targets = batch["tokens"].to(device, non_blocking=True)
            target_lengths = batch["token_lengths"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, output_lengths = model(samples, sample_lengths, languages)
                loss = loss_fn(logits.log_softmax(-1).transpose(0, 1), targets, output_lengths, target_lengths)
                loss = loss / accumulation
            scaler.scale(loss).backward()
            running += float(loss) * accumulation
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        dev = evaluate(model, dev_loader, vocabulary, device)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_dev": min(best_dev, dev["loss"]),
            "vocabulary": str(Path(config["vocabulary"]).resolve()),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if dev["loss"] < best_dev:
            best_dev = dev["loss"]
            checkpoint["best_dev"] = best_dev
            torch.save(checkpoint, run_dir / "best.pt")
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "dev_loss": dev["loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "parameters": model.parameter_count(),
            "previews": dev["previews"],
        }
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(run(args.config, args.resume))


if __name__ == "__main__":
    main()
