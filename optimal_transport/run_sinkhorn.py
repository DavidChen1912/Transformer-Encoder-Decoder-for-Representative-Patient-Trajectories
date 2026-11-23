import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# Sinkhorn (stable version)
# ============================================================
def sinkhorn(a, b, C, epsilon=0.05, max_iter=1000, tol=1e-9):
    K = np.exp(-C / epsilon)
    K[K < 1e-300] = 1e-300

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(max_iter):
        u_prev = u.copy()

        u = a / (K @ v)
        v = b / (K.T @ u)

        if np.linalg.norm(u - u_prev, 1) < tol:
            break

    P = np.diag(u) @ K @ np.diag(v)
    return P


def run_sinkhorn(data_name, tau1, tau2, epsilon, patient_col):
    run_name = f"{data_name}-{tau1}-{tau2}"
    root = Path(__file__).resolve().parent
    cost_path = root / "cost_matrix" / run_name / f"{run_name}_cost_matrix.npy"
    dist_path = root / "data" / run_name / f"{run_name}_distribution.csv"
    result_root = root / "result" / run_name

    cost = np.load(cost_path)
    distributions = pd.read_csv(dist_path, dtype={patient_col: str, "time": str})
    if distributions[patient_col].isna().any():
        raise ValueError(f"Missing patient ID in {dist_path}")

    total_plans = 0
    for patient, rows in distributions.groupby(patient_col):
        if patient in {".", ".."} or "/" in patient or chr(92) in patient:
            raise ValueError(f"Invalid patient directory name: {patient!r}")

        visits = {visit: group for visit, group in rows.groupby("time")}
        if any(not re.fullmatch(r"day[0-9]+", visit) for visit in visits):
            raise ValueError(f"Invalid visit label for {patient}")
        ordered = sorted(visits, key=lambda visit: int(visit[3:]))
        if len({int(visit[3:]) for visit in ordered}) != len(ordered):
            raise ValueError(f"Multiple labels refer to the same sampling day for {patient}")
        if len(ordered) < 2:
            print(f"[SKIP] {patient} has fewer than two visits")
            continue

        patient_dir = result_root / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        for source, target in zip(ordered, ordered[1:]):
            source_df = visits[source].sort_values("cluster_number")
            target_df = visits[target].sort_values("cluster_number")
            clusters = source_df["cluster_number"].to_numpy()
            a = source_df["prob"].to_numpy()
            b = target_df["prob"].to_numpy()
            if (
                len(set(clusters)) != len(clusters)
                or not np.array_equal(clusters, target_df["cluster_number"].to_numpy())
                or cost.shape != (len(clusters), len(clusters))
            ):
                raise ValueError(f"Cluster alignment failed for {patient}, {source} to {target}")
            if not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1):
                raise ValueError(f"Invalid distribution for {patient}, {source} to {target}")

            plan = sinkhorn(a, b, cost, epsilon=epsilon)
            count_plan = plan * source_df["count"].sum()
            prefix = f"{patient}_{source}_to_{target}"
            for kind, matrix in (("prob", plan), ("count", count_plan)):
                np.save(patient_dir / f"{prefix}_{kind}.npy", matrix)
                pd.DataFrame(matrix, index=clusters, columns=clusters).to_csv(
                    patient_dir / f"{prefix}_{kind}.csv"
                )
            print(f"[RUN] {patient}: {source} to {target}")
            total_plans += 1

    print(f"[DONE] {total_plans} plans saved to {result_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve adjacent visits for each patient.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", required=True)
    parser.add_argument("--tau2", required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()
    run_sinkhorn(args.data, args.tau1, args.tau2, args.epsilon, args.patient)
