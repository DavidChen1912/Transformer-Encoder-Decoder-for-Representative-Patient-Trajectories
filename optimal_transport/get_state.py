import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import gzip
from pathlib import Path
import shutil
import subprocess
import pandas as pd

from programs.data_processing.format_ghsom_input_vector import format_ghsom_input_vector


# ============================================================
# ⭐ Pipeline Functions
# ============================================================

def create_ghsom_input_file(data, file, index, label, subnum, patient=None, time=None):
    try:
        format_ghsom_input_vector(data, file, index, label, subnum, patient, time)
        print('Success to create ghsom input file.')
    except Exception as e:
        print('Failed to create ghsom input file.')
        print('Error:', e)


def create_ghsom_prop_file(name, file, tau1=0.1, tau2=0.01,
                           sparseData='yes', isNormalized='false',
                           randomSeed=7, xSize=2, ySize=2,
                           learnRate=0.7, numIterations=20000):

    with open(f'./applications/{file}/GHSOM/{name}_ghsom.prop',
              'w', newline='', encoding='utf-8') as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(['workingDirectory=./'])
        writer.writerow([f'outputDirectory=./output/{file}'])
        writer.writerow([f'namePrefix={name}'])
        writer.writerow([f'vectorFileName=./data/{name}_ghsom.in'])
        writer.writerow([f'sparseData={sparseData}'])
        writer.writerow([f'isNormalized={isNormalized}'])
        writer.writerow([f'randomSeed={randomSeed}'])
        writer.writerow([f'xSize={xSize}'])
        writer.writerow([f'ySize={ySize}'])
        writer.writerow([f'learnRate={learnRate}'])
        writer.writerow([f'numIterations={numIterations}'])
        writer.writerow([f'tau={tau1}'])
        writer.writerow([f'tau2={tau2}'])


def ghsom_clustering(name, file):

    cmd = [
        "./programs/GHSOM/somtoolbox.sh",
        "GHSOM",
        f"./applications/{file}/GHSOM/{name}_ghsom.prop",
        "-h",
    ]
    subprocess.run(cmd, check=True)


def extract_ghsom_output(name, current_path):
    output = Path(current_path) / "applications" / name / "GHSOM" / "output" / name
    for archive in output.glob("*.gz"):
        with gzip.open(archive, "rb") as source, archive.with_suffix("").open("wb") as destination:
            shutil.copyfileobj(source, destination)


# ============================================================
# ⭐ Save cluster label
# ============================================================

def save_ghsom_cluster_label(name, tau1, tau2, index=None, time=None):

    cmd = [
        sys.executable,
        "./programs/data_processing/save_cluster_with_clustered_label_ot.py",
        f"--name={name}",
        f"--tau1={tau1}",
        f"--tau2={tau2}",
    ]
    if index is not None:
        cmd.append(f"--index={index}")
    if time is not None:
        cmd.append(f"--time={time}")

    subprocess.run(cmd, check=True)

    print('Success transfer cluster label.')


# ============================================================
# ⭐ 新增：cluster_number 生成
# ============================================================

def assign_cluster_number(data, tau1, tau2):

    file = f"{data}-{tau1}-{tau2}"

    csv_path = f"./applications/{file}/data/{data}_with_clustered_label-{tau1}-{tau2}.csv"

    print("Assigning cluster_number...")

    df = pd.read_csv(csv_path)

    cluster_map = {}
    cluster_id = 1
    cluster_numbers = []

    for label in df["x_y_label"]:

        if label not in cluster_map:
            cluster_map[label] = cluster_id
            cluster_id += 1

        cluster_numbers.append(cluster_map[label])

    df["cluster_number"] = cluster_numbers

    df.to_csv(csv_path, index=False)

    print(f"Cluster numbering finished. Total clusters = {cluster_id-1}")


# ============================================================
# ⭐ Evaluation
# ============================================================

def clustering_evaluation(name, tau1=0.1, tau2=0.01, label=None, index=None):

    cmd = [
        sys.executable,
        "./programs/evaluation/clustering_scores.py",
        f"--name={name}",
        f"--tau1={tau1}",
        f"--tau2={tau2}",
    ]
    if label is not None:
        cmd.append(f"--label={label}")
    if index is not None:
        cmd.append(f"--index={index}")

    result = subprocess.run(cmd)
    if result.returncode:
        print(f"Clustering evaluation exited with status {result.returncode}.", file=sys.stderr)
    else:
        print("Success evaluating.")


# ============================================================
# ⭐⭐ Pipeline 主流程 ⭐⭐
# ============================================================

def run_pipeline(data, tau1, tau2, index=None, label=None, time=None,
                 subnum=None, feature='mean', patient=None):

    print(f"tau1 = {tau1}, tau2 = {tau2}")
    print(f"data = {data}, index = {index}, label = {label}, time = {time}, patient = {patient}")

    file = f"{data}-{tau1}-{tau2}"

    current_path = os.getcwd()

    print("Current:", current_path)

    app_path = f'{current_path}/applications/{file}'

    if os.path.exists(app_path):

        print(f'Warning : /applications/{file} already exists.')

    else:

        print(f'Creating /applications/{file} ...')

        try:

            os.makedirs(f'{app_path}')
            os.makedirs(f'{app_path}/data')
            os.makedirs(f'{app_path}/GHSOM')
            os.makedirs(f'{app_path}/graphs')
            os.makedirs(f'{app_path}/GHSOM/data')
            os.makedirs(f'{app_path}/GHSOM/output')

            create_ghsom_input_file(data, file, index, label, subnum, patient, time)

            create_ghsom_prop_file(data, file, tau1, tau2)

            ghsom_clustering(data, file)

            extract_ghsom_output(file, current_path)

            save_ghsom_cluster_label(data, tau1, tau2, index, time)

            # ⭐ 新增 cluster number
            assign_cluster_number(data, tau1, tau2)

            clustering_evaluation(data, tau1, tau2, label, index)

        except Exception as e:

            print(f'Failed to create /applications/{file} folder due to: {str(e)}')


# ============================================================
# ⭐ main(): CLI entry
# ============================================================

def main():

    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--tau1', type=float, required=True)
    parser.add_argument('--tau2', type=float, required=True)

    parser.add_argument('--index', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--time', type=str, default=None)

    parser.add_argument('--patient', type=str, default=None)   # ⭐ 新增

    parser.add_argument('--subnum', type=int, default=None)
    parser.add_argument('--feature', type=str, default='mean')

    args = parser.parse_args()

    run_pipeline(
        data=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        index=args.index,
        label=args.label,
        time=args.time,
        subnum=args.subnum,
        feature=args.feature,
        patient=args.patient   # ⭐ 新增
    )


# ============================================================
# ⭐ Entry
# ============================================================

if __name__ == "__main__":
    main()

# python optimal_transport/get_state.py --data=CART_0320 --index=ID --patient=patient --time=time --tau1=0.5 --tau2=0.5