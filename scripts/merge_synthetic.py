"""
merge_synthetic.py
==================
Merges synthetic rare-class examples into training_data.csv
after step1 has been run. Run this AFTER step1, BEFORE step2.

Run:
    python scripts/merge_synthetic.py \
        --training training_data.csv \
        --synthetic data/synthetic/synthetic_rare_classes.csv
"""

import argparse
import pandas as pd
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training",  default="training_data.csv")
    parser.add_argument("--synthetic", default="data/synthetic/synthetic_rare_classes.csv")
    args = parser.parse_args()

    df_main = pd.read_csv(args.training)
    df_syn  = pd.read_csv(args.synthetic)

    print(f"From reviewed Excels: {len(df_main)} rows")
    print(f"From synthetic:       {len(df_syn)} rows")

    df_combined = pd.concat([df_main, df_syn], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["text"])
    df_combined.to_csv(args.training, index=False)

    print(f"Final total:          {len(df_combined)} rows (saved to {args.training})")
    print()
    print("Class distribution after merge:")
    for cls, cnt in sorted(Counter(df_combined["class_label"]).items(),
                           key=lambda x: -x[1]):
        flag = "  ← LOW" if cnt < 15 else ""
        print(f"  {cls:<24} {cnt:>4}{flag}")


if __name__ == "__main__":
    main()
