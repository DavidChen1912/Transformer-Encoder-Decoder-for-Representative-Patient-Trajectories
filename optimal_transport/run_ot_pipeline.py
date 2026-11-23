"""Run scGHSOM and optimal transport for adjacent sampling days.
python3 optimal_transport/run_ot_pipeline.py \
  --data raw-data/CART_0320.csv \
  --tau1 0.5 --tau2 0.5 \
  --index ID --patient patient --time time \
  --lambda 0.3 --epsilon 0.05
  """

import argparse
import csv
import filecmp
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OT_DIR = ROOT / "optimal_transport"


def stage_input(source, required_columns, time_column):
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError(f"Expected an existing CSV file: {source}")

    name = source.stem
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*", name):
        raise ValueError("Use a CSV filename with letters, digits, dots, underscores, or hyphens.")

    visits = read_visits(source, time_column, required_columns)

    destination = ROOT / "raw-data" / f"{name}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not filecmp.cmp(source, destination, shallow=False):
            raise ValueError(f"{destination} already contains different data. Use a unique CSV filename.")
    else:
        shutil.copyfile(source, destination)
    return name, visits


def read_visits(path, time_column, required_columns=()):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        missing = set(required_columns) | {time_column}
        missing.difference_update(rows.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        visits = {row[time_column] for row in rows}

    invalid = [value for value in visits if not re.fullmatch(r"day[0-9]+", value or "")]
    if invalid:
        labels = ", ".join(sorted(map(repr, invalid)))
        raise ValueError(f"Invalid sampling days in {path}: {labels}. Use labels such as day0 and day90.")
    if len(visits) < 2:
        raise ValueError(f"At least two sampling days are required in {path}.")
    return sorted(visits, key=lambda visit: int(visit[3:]))


def run_pipeline(args):
    name, visits = stage_input(args.data.resolve(), (args.index, args.patient), args.time)
    print("Sampling days: " + ", ".join(visits), flush=True)
    tau1, tau2 = str(args.tau1), str(args.tau2)
    run_name = f"{name}-{tau1}-{tau2}"
    app_dir = ROOT / "applications" / run_name
    clustered_csv = app_dir / "data" / f"{name}_with_clustered_label-{tau1}-{tau2}.csv"
    if app_dir.exists() and not clustered_csv.is_file():
        # get_state.py only starts clustering when the application directory is absent.
        if any(app_dir.iterdir()):
            raise ValueError(f"Incomplete clustering in {app_dir}. Move that directory aside and retry.")
        app_dir.rmdir()

    common = [f"--data={name}", f"--tau1={tau1}", f"--tau2={tau2}"]
    time_patient = [f"--time={args.time}", f"--patient={args.patient}"]
    metadata = [f"--index={args.index}", *time_patient]
    stages = [
        ("get_state.py", metadata),
        ("cluster_center.py", metadata),
        ("cluster_weight.py", metadata),
        ("build_cluster_paths.py", []),
        ("get_distribution.py", time_patient),
        ("cost_matrix.py", [f"--lambda={args.lambda_}"]),
        ("run_sinkhorn.py", ["--patient=patient", f"--epsilon={args.epsilon}"]),
    ]

    for number, (script, options) in enumerate(stages, start=1):
        print(f"[{number}/{len(stages)}] {script}", flush=True)
        try:
            subprocess.run([sys.executable, str(OT_DIR / script), *common, *options],
                           cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{script} failed with exit code {error.returncode}.") from error

        if script == "get_state.py":
            if not clustered_csv.is_file():
                raise RuntimeError(f"get_state.py did not generate {clustered_csv}")
            with clustered_csv.open(encoding="utf-8-sig", newline="") as handle:
                columns = set(next(csv.reader(handle), []))
            if not {"cluster_number", args.patient, args.time}.issubset(columns):
                raise RuntimeError(f"Clustered CSV is missing required columns: {clustered_csv}")

            if read_visits(clustered_csv, args.time) != visits:
                raise RuntimeError(f"Sampling days in {clustered_csv} differ from the input CSV.")

    print(f"Results: {OT_DIR / 'result' / run_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the original CSV")
    parser.add_argument("--tau1", type=float, required=True, help="scGHSOM map growth threshold")
    parser.add_argument("--tau2", type=float, required=True, help="scGHSOM hierarchy growth threshold")
    parser.add_argument("--index", required=True, help="Cell ID column")
    parser.add_argument("--patient", required=True, help="Patient ID column in the original CSV")
    parser.add_argument("--time", required=True, help="Time column in the original CSV")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.3, help="Hierarchy cost weight")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn entropy regularization")
    args = parser.parse_args()

    if not math.isfinite(args.lambda_) or args.lambda_ < 0:
        parser.error("--lambda must be finite and nonnegative")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be finite and positive")
    if not all(math.isfinite(tau) for tau in (args.tau1, args.tau2)):
        parser.error("--tau1 and --tau2 must be finite")
    try:
        run_pipeline(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"OT pipeline failed: {error}\n")


if __name__ == "__main__":
    main()
