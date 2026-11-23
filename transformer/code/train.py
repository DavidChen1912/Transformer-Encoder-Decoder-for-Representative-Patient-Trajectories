"""Train either OT reconstruction or a frozen-backbone structure head."""
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from dataset import PatientDataset, load_config, to_device
from model import TrajectoryTransformer, losses


def restore(path, device):
    # Only load checkpoints produced by this project from a trusted source.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TrajectoryTransformer(len(checkpoint["state_ids"]), checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["ot", "structure"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Required OT checkpoint for structure training")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    dataset = PatientDataset(config["data_dir"], config["model"]["time_scale"])
    samples = [dataset[i] for i in range(len(dataset))]
    if args.stage == "structure":
        if args.checkpoint is None:
            parser.error("Structure training requires --checkpoint from Stage 1")
        model, checkpoint = restore(args.checkpoint, device)
        if checkpoint["stage"] != "ot":
            raise ValueError("Expected an OT reconstruction checkpoint")
        if checkpoint["state_ids"] != dataset.states or checkpoint["model_config"] != config["model"]:
            raise ValueError("Checkpoint vocabulary/model configuration does not match")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.structure_head.parameters():
            parameter.requires_grad_(True)
        model.eval()  # Frozen trunk dropout stays disabled.
    else:
        model = TrajectoryTransformer(len(dataset.states), config["model"]).to(device)
        for parameter in model.structure_head.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"][args.stage]["learning_rate"],
        weight_decay=config["training"]["weight_decay"])
    destination = Path(config["output_dir"]) / args.stage
    destination.mkdir(parents=True, exist_ok=True)
    history_path = destination / "history.jsonl"
    if history_path.exists() or (destination / "best.pt").exists():
        raise FileExistsError(f"Choose a new output_dir; training artifacts already exist: {destination}")
    best = float("inf")
    epochs = config["training"][args.stage]["epochs"]
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    for epoch in range(1, epochs + 1):
        if args.stage == "ot":
            model.train()
        optimizer.zero_grad()
        order = list(range(len(samples)))
        random.shuffle(order)
        for index in order:
            sample = to_device(samples[index], device)
            if args.stage == "structure":
                with torch.no_grad():
                    hidden = model.features(sample)
                prediction = model.predictions(hidden, sample)
            else:
                prediction = model(sample)
            objective = losses(prediction, sample, config["structure"]["lambda"])
            (objective[args.stage] / len(samples)).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config["training"]["gradient_clip"])
        optimizer.step()

        # Evaluate the actual post-update parameters, without dropout.
        model.eval()
        metrics = {"ot": 0.0, "structure": 0.0, "soft_coverage": 0.0, "complexity": 0.0}
        with torch.no_grad():
            for raw in samples:
                sample = to_device(raw, device)
                result = losses(model(sample), sample, config["structure"]["lambda"])
                for key in metrics:
                    metrics[key] += float(result[key]) / len(samples)
        record = {"epoch": epoch, "stage": args.stage, **metrics}
        with history_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": config["model"], "state_ids": dataset.states,
            "stage": args.stage, "epoch": epoch, "metrics": metrics,
            "config": config,
        }
        save_checkpoint(destination / "last.pt", payload)
        if metrics[args.stage] < best:
            best = metrics[args.stage]
            save_checkpoint(destination / "best.pt", payload)


if __name__ == "__main__":
    main()
