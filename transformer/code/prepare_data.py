"""Export exhaustive patient paths from existing OT results."""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]



def balance_plan(plan, source, target, tolerance=1e-10, max_iter=100_000):
    """Correct finite-iteration marginal residuals in an in-memory OT copy."""
    adjusted = plan.copy()
    for iteration in range(max_iter):
        row_sum = adjusted.sum(1)
        if ((source > 0) & (row_sum == 0)).any():
            raise ValueError("OT support cannot supply a positive source state")
        adjusted *= np.divide(source, row_sum, out=np.zeros_like(source),
                              where=row_sum > 0)[:, None]
        column_sum = adjusted.sum(0)
        if ((target > 0) & (column_sum == 0)).any():
            raise ValueError("OT support cannot supply a positive target state")
        adjusted *= np.divide(target, column_sum, out=np.zeros_like(target),
                              where=column_sum > 0)[None, :]
        error = max(np.abs(adjusted.sum(1) - source).max(),
                    np.abs(adjusted.sum(0) - target).max())
        if error <= tolerance:
            return adjusted, iteration + 1
    raise ValueError(f"Marginal balancing did not converge: residual={error}")


def prepare(run, output, patient=None, max_paths=1_000_000):
    source = ROOT / "optimal_transport" / "data" / run
    centers = pd.read_csv(source / f"{run}_cluster_center.csv").sort_values("cluster_number")
    states = centers["cluster_number"].to_numpy()
    distributions = pd.read_csv(source / f"{run}_distribution.csv",
                                dtype={"patient": str, "time": str})
    output.mkdir(parents=True, exist_ok=True)
    # State features are shared across patients, so export them once per run.
    shared = centers.set_index("cluster_number").add_prefix("center_")
    weight_path = source / f"{run}_cluster_weight.csv"
    coordinate_path = source / f"{run}_cluster_coordinate.csv"
    if weight_path.exists():
        weights = pd.read_csv(weight_path).set_index("cluster_number")
        shared = shared.join(weights.loc[states].add_prefix("weight_"))
    if coordinate_path.exists():
        coordinates = pd.read_csv(coordinate_path).set_index("cluster_number")
        shared = shared.join(coordinates.loc[states, ["point_x", "point_y"]])
    shared.to_csv(output / "states.csv")
    selected = distributions if patient is None else distributions[distributions.patient == patient]
    if selected.empty:
        raise ValueError("No matching patient data")
    diagnostics = {}
    for name, rows in selected.groupby("patient"):
        if name in {".", ".."} or "/" in name or chr(92) in name:
            raise ValueError(f"Unsafe patient name: {name!r}")
        visits = rows.time.unique().tolist()
        if any(not re.fullmatch(r"day[0-9]+", v) for v in visits):
            raise ValueError(f"Invalid sampling day for {name}")
        visits.sort(key=lambda v: int(v[3:]))
        if len({int(v[3:]) for v in visits}) != len(visits):
            raise ValueError("Duplicate numeric days")
        if len(visits) < 2:
            print(f"Skipping {name}: fewer than two visits")
            continue
        n_paths = len(states) ** len(visits)
        if n_paths > max_paths:
            raise ValueError(f"{name}: {n_paths} paths exceeds --max-paths")
        proportions = np.stack([
            rows[rows.time == v].set_index("cluster_number").loc[states, "prob"].to_numpy()
            for v in visits
        ])
        plans = []
        for a, b in zip(visits, visits[1:]):
            path = ROOT / "optimal_transport" / "result" / run / name / f"{name}_{a}_to_{b}_prob.csv"
            frame = pd.read_csv(path, index_col=0)
            frame.columns = frame.columns.astype(states.dtype)
            plans.append(frame.loc[states, states].to_numpy())
        plans = np.stack(plans)
        if not np.isfinite(plans).all() or (plans < 0).any():
            raise ValueError(f"Invalid OT values for {name}")
        np.testing.assert_allclose(proportions.sum(1), 1, atol=1e-6)
        proportions = proportions / proportions.sum(1, keepdims=True)
        records = []
        for t in range(len(plans)):
            original = plans[t].copy()
            before = max(np.abs(original.sum(1) - proportions[t]).max(),
                         np.abs(original.sum(0) - proportions[t + 1]).max())
            iterations = 0
            if before > 1e-10:
                plans[t], iterations = balance_plan(original, proportions[t], proportions[t + 1])
            records.append({
                "source": visits[t], "target": visits[t + 1],
                "max_marginal_error_before": float(before),
                "max_marginal_error_after": float(max(
                    np.abs(plans[t].sum(1) - proportions[t]).max(),
                    np.abs(plans[t].sum(0) - proportions[t + 1]).max())),
                "max_entry_adjustment": float(np.abs(plans[t] - original).max()),
                "balancing_iterations": iterations,
            })
        diagnostics[name] = records
        row_mass = plans.sum(2, keepdims=True)
        conditional = np.divide(plans, row_mass, out=np.zeros_like(plans), where=row_mass > 0)
        indices = np.array(list(itertools.product(range(len(states)), repeat=len(visits))))
        table = pd.DataFrame({"path_id": np.arange(n_paths), "patient_id": name})
        for t, visit in enumerate(visits):
            table[f"state_{visit}"] = states[indices[:, t]]
        weight = proportions[0, indices[:, 0]].copy()
        for t, (a, b) in enumerate(zip(visits, visits[1:])):
            q = conditional[t, indices[:, t], indices[:, t + 1]]
            table[f"mass_{a}_to_{b}"] = plans[t, indices[:, t], indices[:, t + 1]]
            table[f"conditional_prob_{a}_to_{b}"] = q
            weight *= q
        np.testing.assert_allclose(weight.sum(), 1, atol=1e-6)
        table["path_mass"] = weight
        destination = output / name
        destination.mkdir(exist_ok=True)
        temporary = destination / "paths.csv.tmp"
        table.to_csv(temporary, index=False)
        temporary.replace(destination / "paths.csv")
        print(f"{name}: {n_paths} paths -> {destination}")
    (output / "metadata.json").write_text(json.dumps({
        "run": run, "path_assumption": "Markov chaining",
        "conditional_normalization": "OT row sums after marginal balancing to observed proportions",
        "mass_columns": "Balanced OT copy; original optimal_transport files remain intact",
        "marginal_balancing": diagnostics,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path,
                        help="Output directory; defaults to transformer/data/<run>/")
    parser.add_argument("--patient")
    parser.add_argument("--max-paths", type=int, default=1_000_000)
    args = parser.parse_args()
    output = args.output or ROOT / "transformer" / "data" / args.run
    prepare(args.run, output.resolve(), args.patient, args.max_paths)


if __name__ == "__main__":
    main()
