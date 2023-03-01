import glob
import os
import json
import pickle
from tqdm import tqdm

annotation_path = "../cache"


def prepare_annotations():
    # We use the annotations provided by pyskl: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl
    path_to_anno = '/home/zhong/Documents/datasets/NTU_60/ntu60_3danno.pkl'
    with open(path_to_anno, 'rb') as fp:
        data = pickle.load(fp)

    train = {}
    val = {}

    # Construct training annotations
    xsub_train = [{'id': id, 'label': int(id[-3:])-1}
                  for id in data['split']['xsub_train']]
    xview_train = [{'id': id, 'label': int(id[-3:])-1}
                   for id in data['split']['xview_train']]
    train['xsub'] = xsub_train
    train['xview'] = xview_train

    # Construct validation annotations
    xsub_val = [{'id': id, 'label': int(id[-3:])-1}
                for id in data['split']['xsub_val']]
    xview_val = [{'id': id, 'label': int(id[-3:])-1}
                 for id in data['split']['xview_val']]
    val['xsub'] = xsub_val
    val['xview'] = xview_val

    # Dump to jsons
    save_train_path = os.path.join(annotation_path, "train.json")
    with open(save_train_path, "w") as f:
        json.dump(train, f)
    print(f"Saved train annotations with {len(train['xsub'])} "
          f"cross subject samples and {len(train['xview'])} cross view samples"
          f"to {save_train_path}.")

    save_val_path = os.path.join(annotation_path, "validation.json")
    with open(save_val_path, "w") as f:
        json.dump(val, f)
    print(f"Saved validation annotations with {len(val['xsub'])} "
          f"cross subject samples and {len(val['xview'])} cross view samples"
          f"to {save_val_path}.")


# /hkfs/work/workspace_haic/scratch/kf3609-datasets/NTURGBD/nturgbd_depth_masked/
# NTU60: 001-017
# NTU120: 001-032
def get_ntu_path_to_videos(
        root_path='/pfs/work8/workspace/ffuc/scratch/on3546-datasets/NTURGBD',
):
    seq_prefix = 'nturgb+d_depth_masked_'
    annotation_path = os.path.join(root_path, 'annotations')
    root_path = os.path.join(root_path, 'nturgb+d_depth_masked')

    for mode in ['train', 'validation']:
        with open(os.path.join(annotation_path, f'{mode}.json'), 'r') as f:
            data = json.load(f)

        for split in ['xsub', 'xview']:
            video_list = [item['id'] for item in data[split]]
            res = {}
            i = 0
            for video in tqdm(video_list, desc=f'{mode}, {split}'):
                seq_path = os.path.join(
                    root_path,
                    '{}{}'.format(seq_prefix, video[:4].lower())
                )
                image_paths = sorted(glob.glob(os.path.join(seq_path, video, '*.png')))

                if len(image_paths) == 0:
                    i += 1
                    print(f'No pngs in {os.path.join(seq_path, video)}')

                res[video] = image_paths

            save_path = os.path.join(
                annotation_path,
                f'path_to_{mode}_{split}_videos.json')

            with open(save_path, 'w') as f:
                json.dump(res, f)

            n_videos = len(res)
            n_images = sum([len(res[key]) for key in res])
            print(f"Discarded {i} videos.")
            print(f"Saved data corresponding to {n_videos} videos and {n_images} images to {save_path}.")


def test_json():
    fpath = f"../cache/path_to_train_xsub_videos.json"
    print(f"Opening {fpath}...")
    with open(fpath, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    get_ntu_path_to_videos()
    # test_json()
    # prepare_annotations()