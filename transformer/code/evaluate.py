"""Report reconstruction and final hard-graph metrics per patient."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import PatientDataset, to_device
from extract_trajectory import build_graph, graph_metrics
from model import losses
from train import restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between zero and one")
    model, checkpoint = restore(args.checkpoint, args.device)
    dataset = PatientDataset(args.data, model.config["time_scale"])
    if dataset.states != checkpoint["state_ids"]:
        raise ValueError("Checkpoint state vocabulary mismatch")
    records = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            sample = to_device(raw, args.device)
            prediction = model(sample)
            target, predicted = sample["target"], prediction["prob"]
            metric = {
                "patient": raw["patient"],
                "kl_mean": float(losses(prediction, sample, 0)["ot"]),
                "tv_mean": float((predicted - target).abs().sum((1, 2)).mean() / 2),
                "source_marginal_max_error": float((predicted.sum(2) - target.sum(2)).abs().max()),
                "target_marginal_max_error": float((predicted.sum(1) - target.sum(1)).abs().max()),
                "tv_worst_interval": float((predicted - target).abs().sum((1, 2)).max() / 2),
            }
            if checkpoint["stage"] == "structure":
                selected, backbone = build_graph(raw, prediction["gates"].cpu().numpy(), args.threshold)
                metric.update(graph_metrics(raw, selected, backbone))
            records.append(metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(json.dumps({"checkpoint_stage": checkpoint["stage"], "patients": len(records),
                      "evaluation": "Observed-cohort reconstruction; no held-out-patient claim"}))


if __name__ == "__main__":
    main()
