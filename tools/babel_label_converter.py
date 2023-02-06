import glob
import json
import os
import csv
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--babel_file", type=str, default="/lsdf/data/activity/BABEL/babel_v1-0_release/train.json", help="Path to babel file")
parser.add_argument("--label_indices", type=str, default="/lsdf/data/activity/BABEL/category_index.csv", help="Path to label indices")
parser.add_argument("--base_path", type=str, default="/lsdf/data/activity/AMARV/run4_2023_01_27/", help="Path to base directory")
parser.add_argument("--output", type=str, default=None, help="Output file name")

args = parser.parse_args()

if args.output is None:
    args.output = os.path.splitext(os.path.split(args.babel_file)[1])[0] + ".csv"
    print(f"No output path provided, using: {args.output}")

label_indices = pd.read_csv(args.label_indices, sep="#")
label_indices = {k: v for v, k in enumerate(label_indices["category"].values)}

with open(args.babel_file, "r") as bf:
    bab = json.load(bf)

samples = glob.glob(os.path.join(args.base_path, '**/sequence_*'), recursive=True)
path_mappings = {os.path.join(*os.path.normpath(p).split(os.path.sep)[-4:-1]): p.removeprefix(args.base_path) for p in
                 samples}


def extract_label_and_set_times(sample):
    if sample["frame_ann"]:
        return sample["frame_ann"]["labels"], sample["dur"]
    else:
        labels = sample["seq_ann"]["labels"][:1]
        labels[0]["start_t"] = 0
        labels[0]["end_t"] = float(sample["dur"])

        return labels, sample["dur"]


anns = {os.path.join(*os.path.normpath(b["feat_p"][:-10]).split(os.path.sep)[1:]): extract_label_and_set_times(b) for b
        in bab.values()}

num_lines = 0
with open(args.output, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)

    for sam, (segs, dur) in anns.items():
        if sam in path_mappings:
            for seg in segs:
                action_cat_indices = [str(label_indices[a]) for a in seg["act_cat"]]
                line = [path_mappings[sam], sam, ";".join(action_cat_indices), ";".join(seg["act_cat"]), seg["proc_label"],
                     str(seg["start_t"]), str(seg["end_t"]), str(dur)]
                writer.writerow(line)
                num_lines += 1

print(f"Wrote {num_lines} lines.")