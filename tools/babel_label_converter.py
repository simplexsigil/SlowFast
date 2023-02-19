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

    return {k: v for v, k in enumerate(natsort.natsorted(list(raw_labels)))}, {k: v for v, k in enumerate(
        label_indices["category"].values)}


def get_path_mappings(base_path, pattern="**/sequence_*"):
    glob_pattern = os.path.join(base_path, pattern)
    print(f"Glob pattern: {glob_pattern}")
    files = glob.glob(glob_pattern, recursive=True)
    print(len(files))
    return {os.path.join(*os.path.normpath(p).split(os.path.sep)[-4:-1]):
                p.removeprefix(base_path).replace("_", "").replace(" ", "")
            for p in files}, \
        {os.path.split(p.removeprefix(base_path))[0]:
             os.path.join(*os.path.normpath(p).split(os.path.sep)[-4:-1]).replace("_", "").replace(" ", "")
         for p in files}


def get_anns(babel_file_path):
    with open(babel_file_path, "r") as bf:
        babel_data = json.load(bf)

    anns = {os.path.join(*os.path.normpath(b["feat_p"][:-10]).split(os.path.sep)[1:]).replace("_", "").replace(" ",
                                                                                                               ""): extract_label_and_set_times(
        b)
        for b
        in babel_data.values()}

    anns = dirty_fixes(anns)

    return anns


def dirty_fixes(anns):
    if 'MPImosh/50021/armyposes' in anns:
        anns["MPImosh/50021/army"] = anns['MPImosh/50021/armyposes']
        del anns['MPImosh/50021/armyposes']
        print(f"Applied dirty fix for MPImosh/50021/armyposes")

    if 'MPImosh/00058/armyposes' in anns:
        anns["MPImosh/00058/army"] = anns['MPImosh/00058/armyposes']
        del anns['MPImosh/00058/armyposes']
        print(f"Applied dirty fix for MPImosh/00058/armyposes")

    if "MPImosh/50022/stretchposes" in anns:
        anns["MPImosh/50022/stretch"] = anns["MPImosh/50022/stretchposes"]
        del anns["MPImosh/50022/stretchposes"]
        print(f"Applied dirty fix for MPImosh/50022/stretchposes")

    return anns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--babel_root", type=str,
                        default=os.path.expandvars("$LSDF/data/activity/BABEL/babel_v1-0_release"),
                        help="Path to babel files")
    parser.add_argument("--label_indices", type=str,
                        default=os.path.expandvars("$LSDF/data/activity/BABEL/category_index.csv"),
                        help="Path to label indices")
    parser.add_argument("--base_path", type=str,
                        default=os.path.expandvars("$LSDF/data/activity/AMARV/run4_2023_02_05/"),
                        help="Path to base directory")
    parser.add_argument("--output_directory", type=str, default=None, help="Output file name")
    parser.add_argument('--save_index_files', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.output_directory is None:
        args.output_directory = args.babel_root
        print(f"No output path provided, using: {args.output_directory}/{{train,val,test}}.csv")

    raw_act_indices, act_cat_indices = get_label_indices(args.label_indices)

    if args.save_index_files:
        with open(os.path.join(args.output_directory, 'act_cat_indices.csv'), 'w') as f:
            w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in act_cat_indices.items():
                w.writerow(row)

        with open(os.path.join(args.output_directory, 'raw_act_indices.csv'), 'w') as f:
            w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in raw_act_indices.items():
                w.writerow(row)

    train_anns = get_anns(os.path.join(args.babel_root, "train.json"))
    val_anns = get_anns(os.path.join(args.babel_root, "val.json"))
    test_anns = get_anns(os.path.join(args.babel_root, "test.json"))

    print(f"Found {len(train_anns)} train annotations.")
    print(f"Found {len(val_anns)} val annotations.")
    print(f"Found {len(test_anns)} test annotations.")

    all_anns = {}
    for anns in (train_anns, val_anns, test_anns): all_anns.update(anns)

    assert (not set(train_anns.keys()).intersection(val_anns.keys())) and (
        not set(train_anns.keys()).intersection(test_anns.keys())) and (
               not set(val_anns.keys()).intersection(test_anns.keys()))

    print(f"{len(all_anns)} annotations in total.")

    _, path_mappings = get_path_mappings(args.base_path)
    print(f"Found {len(path_mappings)} paths in {args.base_path}.")
    pd.DataFrame.from_records(list(path_mappings.items())).to_csv("paths.csv", header=False, index=False)

    i = 0
    for sams in set(path_mappings.values()):
        if sams not in all_anns:
            print(f"Path {sams} has no corresponding annotation.")
            i += 1
    print(f"{i} paths without annotations (and potentially more sequences).")

    for ann in all_anns:
        if ann not in set(path_mappings.values()):
            print(f"Annotation {ann} has no corresponding path.")
            i += 1

    for anns, outfile in zip([train_anns, val_anns, test_anns], ["train.csv", "val.csv", "test.csv"]):
        num_lines = 0
        with open(os.path.join(args.output_directory, outfile), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)

            for path, sam in path_mappings.items():
                if sam in anns:
                    segs, dur = anns[sam]
                    for seg in segs:
                        action_cat_indices = [str(act_cat_indices[a]) for a in seg["act_cat"]] if seg[
                                                                                                      "act_cat"] is not None else [
                            "-1", ]
                        raw_cat_index = raw_act_indices[' '.join(seg["proc_label"].split())] if seg[
                                                                                                    "proc_label"] is not None else "-1"

                        line = [path, ";".join(action_cat_indices),
                                ";".join(seg["act_cat"]) if seg["act_cat"] else "None",
                                raw_cat_index, seg["proc_label"] if seg["proc_label"] else "None",
                                str(seg["start_t"]), str(seg["end_t"]), str(dur)]
                        writer.writerow(line)
                    num_lines += 1

        print(f"Wrote {num_lines} paths with annotations in {outfile}")


if __name__ == "__main__":
    main()
