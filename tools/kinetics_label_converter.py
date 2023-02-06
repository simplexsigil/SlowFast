import argparse
import os
import csv
import glob
import tqdm
import sys

def main(base_path, label_file, target_file, with_header=True):
    os.makedirs(os.path.split(target_file)[0], exist_ok = True)

    labels = set()
    new_rows = []

    with open(label_file, "r") as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            if with_header and idx == 0: continue  # CSV has a header line.
            labels.add(row[0])
            new_rows.append([row[1], row[0]])

    print(f"Found {len(labels)} annotations.")

    labels = sorted(list(labels))
    label_to_index = {label: index for index, label in enumerate(labels)}

    new_rows = [(rw[0], label_to_index[rw[1]]) for rw in new_rows]

    file_list = glob.glob(os.path.join(base_path,"**/*.mp4"))

    print(f"Found {len(file_list)} files with paths like")
    print(file_list[:4])

    id_to_file_path = {}
    for file_path in file_list:
        vid_id = os.path.split(file_path)[1][:11]
        id_to_file_path[vid_id] = os.path.relpath(file_path, base_path)

    fin_rows = []

    for row in tqdm.tqdm(new_rows):
        vid_id = row[0][:11]
        if vid_id in id_to_file_path:
            fin_rows.append([id_to_file_path[vid_id], row[1]])
#        else:
#            print(f"No annotation: {row}")

    print(f"Found {len(fin_rows)} annotated files.")

    with open(target_file, "w", newline='') as file:
        writer = csv.writer(file)
        for row in fin_rows:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process annotation files.')
    parser.add_argument('-v', '--video_root', type=str, default=None,
    help='Path to the video files')
    parser.add_argument('-l', '--label_file', type=str, default='test.csv',
    help='Label file name')
    parser.add_argument('-t','--target_file', type=str, default=None, help='Target file path')
    args = parser.parse_args()

    if args.video_root is None:
        parser.print_help()
        sys.exit(1)

    if args.target_file is None:
        args.target_file = os.path.join(os.path.split(args.label_file)[0], "slowfast", os.path.split(args.label_file)[1])

    main(args.video_root, args.label_file, args.target_file)
