"""
$ python tools/compute_stats.py --csv-pattern "../local/eval/*/*/csv/version_0/metrics.csv"
"""

import glob
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--csv-pattern", type=str, required=True)
args = parser.parse_args()

for fpath in sorted(glob.glob(args.csv_pattern)):
    data = pd.read_csv(fpath)
    print("")
    print(fpath)
    for tag, data_g in data.groupby("env"):
        print(f"--- {tag} ---")
        for k in data_g.keys():
            if k == "env":
                continue
            print(f"{k}: {data_g[k].mean()} +- {data_g[k].std()}")
    print("========================")
