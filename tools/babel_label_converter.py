import glob
import json
import os
import csv
import pandas as pd
import argparse
import re
import natsort


def extract_label_and_set_times(sample):
    if sample["frame_ann"]:
        return sample["frame_ann"]["labels"], sample["dur"]
    else:
        labels = sample["seq_ann"]["labels"][:1]
        labels[0]["start_t"] = 0
        labels[0]["end_t"] = float(sample["dur"])

        return labels, sample["dur"]


def get_label_indices(label_indices_path):
    raw_labels = set()
    label_indices = pd.read_csv(label_indices_path, sep="#")

    for rls in label_indices["raw_labels"]:
        rls = [rl.strip() for rl in rls.split(",")]
        raw_labels.update(rls)

    return {k: v for v, k in enumerate(natsort.natsorted(list(raw_labels)))}, {k: v for v, k in enumerate(label_indices["category"].values)}


def get_path_mappings(base_path, pattern="**/sequence_*"):
    glob_pattern = os.path.join(base_path, pattern)
    print(f"Glob pattern: {glob_pattern}")
    files = glob.glob(glob_pattern, recursive=True)
    print(len(files))
    return {os.path.join(*os.path.normpath(p).split(os.path.sep)[-4:-1]): p.removeprefix(base_path) for p in files}, {p.removeprefix(base_path): os.path.join(*os.path.normpath(p).split(os.path.sep)[-4:-1]) for p in files}


def get_anns(babel_file_path):
    with open(babel_file_path, "r") as bf:
        babel_data = json.load(bf)

    anns = {os.path.join(*os.path.normpath(b["feat_p"][:-10]).split(os.path.sep)[1:]): extract_label_and_set_times(b)
            for b
            in babel_data.values()}

    return anns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--babel_file", type=str,
                        default=os.path.expandvars("$LSDF/data/activity/BABEL/babel_v1-0_release/train.json"),
                        help="Path to babel file")
    parser.add_argument("--label_indices", type=str,
                        default=os.path.expandvars("$LSDF/data/activity/BABEL/category_index.csv"),
                        help="Path to label indices")
    parser.add_argument("--base_path", type=str,
                        default=os.path.expandvars("$LSDF/data/activity/AMARV/run4_2023_01_27/"),
                        help="Path to base directory")
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    parser.add_argument('--save_index_files', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(os.path.split(args.babel_file)[1])[0] + ".csv"
        print(f"No output path provided, using: {args.output}")

    raw_act_indices, act_cat_indices = get_label_indices(args.label_indices)

    if args.save_index_files:
        with open('act_cat_indices.csv', 'w') as f:
            w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in act_cat_indices.items():
                w.writerow(row)

        with open('raw_act_indices.csv', 'w') as f:
            w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in raw_act_indices.items():
                w.writerow(row)


    _, path_mappings = get_path_mappings(args.base_path)
    print(f"Found {len(path_mappings)} paths in {args.base_path}.")

    anns = get_anns(args.babel_file)
    print(f"Found {len(anns)} annotations.")

    num_lines = 0
    with open(args.output, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)

        for path, sam in path_mappings.items():
            if sam in anns:
                segs,dur = anns[sam]
                for seg in segs:
                    action_cat_indices = [str(act_cat_indices[a]) for a in seg["act_cat"]]  if seg["act_cat"] is not None else ["-1",]
                    raw_cat_index = raw_act_indices[' '.join(seg["proc_label"].split())] if seg["proc_label"] is not None else "-1"
                    
                    line = [path_mappings[path], sam, ";".join(action_cat_indices), ";".join(seg["act_cat"]) if seg["act_cat"] else "None",
                            raw_cat_index, seg["proc_label"],
                            str(seg["start_t"]), str(seg["end_t"]), str(dur)]
                    writer.writerow(line)
                num_lines += 1
    print(f"Wrote {num_lines} paths with annotations.")
    print(f"There are {len(path_mappings) - num_lines} paths without annotations.")


if __name__ == "__main__":
    main()
