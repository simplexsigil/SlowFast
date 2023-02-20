import pandas as pd
import os
import random
import argparse


def partition_csv_file(csv_file, output_dir, p):
    """
    Partitions a CSV file into two non-overlapping groups.

    Args:
        csv_file (str): The path to the CSV file.
        output_dir (str): The path to the output directory.
        p (float): The fraction of rows to include in the first group.
    """
    df = pd.read_csv(csv_file, sep=",", header=None)
    unique_paths = df[0].unique()
    random.shuffle(unique_paths)
    split_index = int(p * len(unique_paths))
    test_paths = unique_paths[:split_index]
    train_paths = unique_paths[split_index:]
    train_df = df[df[0].isin(train_paths)]
    test_df = df[df[0].isin(test_paths)]
    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "val.csv")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Partition a CSV file into two non-overlapping groups.")
    parser.add_argument("--csv_file", type=str, default="/lsdf/data/activity/AMARV/annotations/general/train.csv",
                        help="The path to the CSV file.")
    parser.add_argument("--output_dir", type=str, default="/lsdf/data/activity/AMARV/annotations/val_as_test",
                        help="The path to the output directory.")
    parser.add_argument("--p", type=float, default=0.2, help="The fraction of rows to include in the first group.")
    return parser.parse_args()


def main():
    args = parse_args()
    partition_csv_file(args.csv_file, args.output_dir, args.p)


if __name__ == "__main__":
    main()
