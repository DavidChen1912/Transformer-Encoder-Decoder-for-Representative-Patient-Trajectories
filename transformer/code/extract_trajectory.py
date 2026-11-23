"""Assemble temporal graphs from learned gates; designate a backbone."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import PatientDataset, to_device
from train import restore


def build_graph(sample, gates, threshold):
    target = sample["target"].cpu().numpy()
    steps, k, _ = target.shape
    gates = np.asarray(gates).reshape(target.shape)
    selected = (gates >= threshold) & (target > 0)
    # Forward and backward reachability remove incomplete fragments.
    forward = np.zeros((steps + 1, k), dtype=bool)
    forward[0] = target[0].sum(1) > 0
    for t in range(steps):
        forward[t + 1] = (selected[t] & forward[t, :, None]).any(0)
    backward = np.zeros_like(forward)
    backward[-1] = target[-1].sum(0) > 0
    for t in reversed(range(steps)):
        backward[t] = (selected[t] & backward[t + 1, None, :]).any(1)
    selected &= forward[:-1, :, None] & backward[1:, None, :]
    if not selected.any() or not (forward[-1] & backward[-1]).any():
        return selected, None

    # Highest ORIGINAL Markov path mass in the retained graph.
    initial = target[0].sum(1)
    score = np.full(k, -np.inf)
    score[initial > 0] = np.log(initial[initial > 0])
    predecessors = []
    for t in range(steps):
        row_mass = target[t].sum(1, keepdims=True)
        q = np.divide(target[t], row_mass, out=np.zeros_like(target[t]), where=row_mass > 0)
        log_q = np.full((k, k), -np.inf)
        valid = selected[t] & (q > 0)
        log_q[valid] = np.log(q[valid])
        candidate = score[:, None] + log_q
        predecessor = candidate.argmax(0)  # Stable vocabulary-order tie breaking.
        score = candidate[predecessor, np.arange(k)]
        predecessors.append(predecessor)
    final = int(score.argmax())
    if not np.isfinite(score[final]):
        return selected, None
    backbone = [final]
    for predecessor in reversed(predecessors):
        backbone.append(int(predecessor[backbone[-1]]))
    backbone.reverse()

    # Keep the weak component containing the backbone, allowing merges.
    adjacency = {}
    for t, i, j in np.argwhere(selected):
        a, b = (int(t), int(i)), (int(t + 1), int(j))
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    reached = {(0, backbone[0])}
    pending = list(reached)
    while pending:
        node = pending.pop()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    for t, i, j in np.argwhere(selected):
        if (int(t), int(i)) not in reached:
            selected[t, i, j] = False
    return selected, backbone


def graph_metrics(sample, selected, backbone):
    flat = torch.as_tensor(selected.flatten(), dtype=torch.bool)
    covered = flat[sample["edge_indices"].cpu()].all(1)
    coverage = float(sample["weights"].cpu()[covered].sum())
    branching = int((selected.sum(2) > 1).sum())
    return {
        "valid": backbone is not None,
        "path_mass_coverage": coverage,
        "excluded_path_mass": 1 - coverage,
        "retained_edges": int(selected.sum()),
        "branching_nodes": branching,
        "candidate_positive_edges": int((sample["target"].cpu().numpy() > 0).sum()),
        "tie_rule": "first maximum in stored vocabulary order",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patient")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between zero and one")
    model, checkpoint = restore(args.checkpoint, args.device)
    if checkpoint["stage"] != "structure":
        raise ValueError("Extraction requires a trained structure-stage checkpoint")
    dataset = PatientDataset(args.data, model.config["time_scale"])
    if dataset.states != checkpoint["state_ids"]:
        raise ValueError("Checkpoint state vocabulary mismatch")
    model.eval()
    matched = False
    with torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            if args.patient and raw["patient"] != args.patient:
                continue
            matched = True
            sample = to_device(raw, args.device)
            prediction = model(sample)
            gates = prediction["gates"].cpu().numpy()
            selected, backbone = build_graph(raw, gates, args.threshold)
            output = args.output / raw["patient"]
            output.mkdir(parents=True, exist_ok=True)
            days = raw["days"].tolist()
            backbone_edges = set()
            if backbone is not None:
                backbone_edges = {(t, backbone[t], backbone[t + 1]) for t in range(len(backbone) - 1)}
            records = []
            target = raw["target"].numpy()
            gate_matrix = gates.reshape(target.shape)
            for t, i, j in np.argwhere(selected):
                records.append({
                    "source_day": int(days[t]), "target_day": int(days[t + 1]),
                    "source_state": dataset.states[i], "target_state": dataset.states[j],
                    "gate": float(gate_matrix[t, i, j]), "original_ot_mass": float(target[t, i, j]),
                    "is_backbone": (t, i, j) in backbone_edges,
                })
            columns = ["source_day", "target_day", "source_state", "target_state",
                       "gate", "original_ot_mass", "is_backbone"]
            pd.DataFrame(records, columns=columns).to_csv(output / "representative_edges.csv", index=False)
            main_path = [] if backbone is None else [
                {"day": int(day), "state_id": dataset.states[state]}
                for day, state in zip(days, backbone)
            ]
            pd.DataFrame(main_path, columns=["day", "state_id"]).to_csv(output / "main_path.csv", index=False)
            summary = graph_metrics(raw, selected, backbone)
            summary.update({"patient": raw["patient"], "threshold": args.threshold,
                            "checkpoint": str(args.checkpoint.resolve())})
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary))
    if not matched:
        raise ValueError("No patient matched --patient")


if __name__ == "__main__":
    main()
