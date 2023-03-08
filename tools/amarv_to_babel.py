import torch
import os
from sklearn.metrics import average_precision_score, balanced_accuracy_score
import json
import pickle

from slowfast.utils.env import pathmgr


def amarv2babel():
    mapping = {}

    anno_path = '/home/zhong/Documents/datasets/amarv/annotations/val_as_test'
    # read the train, val, test csv
    for split in ['train', 'val', 'test']:
        path_to_file = os.path.join(anno_path, f'{split}.csv')
        with pathmgr.open(path_to_file, "r") as f:
            print(f"Loading data for {path_to_file}")
            rows = f.read().splitlines()

        for clip_idx, path_label in enumerate(rows[1:]):
            fetch_info = path_label.split(",")
            path, act_cats, act_cats_150, act_cats_labels, act, act_label, t_start, t_stop, dur = fetch_info

            acs, acs_150 = list(int(c) for c in act_cats.split(";")), \
                           list(int(c) for c in act_cats_150.split(";"))

            for ac, ac_150 in zip(acs, acs_150):
                if ac not in mapping:
                    mapping[ac] = ac_150
                else:
                    assert ac_150 == mapping[ac]

    save_path = os.path.join(anno_path, 'amarv_to_babel.json')
    with open(save_path, 'w') as f:
        json.dump(mapping, f)


def analyse_babel(path_to_logits):
    print(f'Loading logits from {path_to_logits}')
    anno_path = '/home/zhong/Documents/datasets/amarv/annotations/val_as_test'
    with open(os.path.join(anno_path, f'amarv_to_babel.json'), 'r') as f:
        mapping = json.load(f)

    max_cls = max([int(k) for k in mapping.keys()])
    assert max_cls <= 271
    mapping_array = torch.zeros(272, dtype=torch.int32) - 1
    for k, v in mapping.items():
        mapping_array[int(k)] = v

    for eval_top in [60, 120]:
        # load logits
        with open(path_to_logits, "rb") as f:
            data = pickle.load(f)

        logits, labels = data[:2]
        # overall top1
        predictions = logits.argmax(-1)
        top1 = (predictions == labels).sum() / len(labels)
        print(f"Amarv top1: {top1}")

        no_babel_mask = (mapping_array == -1) | (mapping_array >= eval_top)

        # logits to babel logits
        logits[:, no_babel_mask] = -float('inf')

        # we select only samples, which have valid babel labels
        label_babel = mapping_array[labels]
        eval_mask = (label_babel > -1) & (label_babel < eval_top)
        logits = logits[eval_mask, :]
        label_babel = label_babel[eval_mask]

        ks = [1, 5]
        _top_max_k_vals, top_max_k_inds = torch.topk(
            logits, max(ks), dim=1, largest=True, sorted=True
        )

        # convert amarv index to babel index
        top_max_k_inds = mapping_array[top_max_k_inds]

        # (batch_size, max_k) -> (max_k, batch_size).
        top_max_k_inds = top_max_k_inds.t()
        # (batch_size, ) -> (max_k, batch_size).
        rep_max_k_labels = label_babel.view(1, -1).expand_as(top_max_k_inds)
        # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
        top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
        # Compute the number of topk correct predictions for each k.
        topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
        topks_correct = [k / len(label_babel) for k in topks_correct]

        print(f'Babel{eval_top} top1 top5: {topks_correct}')

        preds_idxs = top_max_k_inds[0]
        bal_acc = balanced_accuracy_score(label_babel, preds_idxs)
        print(f'Babel{eval_top} balenced accuracy: {bal_acc}')


if __name__ == "__main__":
    analyse_babel(
        '/home/zhong/Documents/datasets/amarv/temp/x3d_amarv_val_as_test_results.pkl')
    analyse_babel(
        '/home/zhong/Documents/datasets/amarv/temp/mvitv2-s_amarv_val_as_test_results.pkl')