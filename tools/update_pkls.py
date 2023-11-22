import joblib
import argparse
import re
import os


def load_pickle(file_path):
    """
    Load the pickle file using joblib.
    """
    try:
        data = joblib.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def extract_info_from_path(self, path):
    """
    This is also different due to the action not being a separate folder.
    :param path:
    :return:
    """
    file = os.path.split(path)[1]
    match = self.action_pat_simple.match(file)
    if match:
        main_action = match.group(1)
        sub_action = None
        person = match.group(2)
        # r = match.group(3) -> ?
        # v = match.group(4) -> ?
        camera = match.group(5)
    else:
        match = self.action_pat_complex.match(file)
        if match:
            main_action = match.group(1)
            sub_action = match.group(2)
            person = match.group(3)
            # r = match.group(4) -> ?
            # v = match.group(5) -> ?
            camera = match.group(6)
        else:
            raise ValueError

    action = main_action if sub_action is None else main_action + "." + sub_action

    return file, action, main_action, sub_action, int(person), int(camera)


def update_meta_information(meta_dict):
    # Regular expression to match the filename pattern and extract required groups
    regex = r".*/([^/]+)_p(\d+)_r(\d+)_v(\d+)_c(\d+).*$"

    # Check if 'vid_path' key exists in the dictionary
    if "vid_path" not in meta_dict:
        print("Error: 'vid_path' key not found in the dictionary.")
        return False

    if "camera" in meta_dict:
        print("File already has camera annotation.")
        return False

    # Iterate over each file path in the 'vid_path' list
    for i, filepath in enumerate(meta_dict["vid_path"]):
        match = re.match(regex, filepath)
        if match:
            # Extracting the groups from the filename
            action, p, r, v, c = match.groups()

            # Update or create new entries in the dictionary for extracted data
            if "actions" not in meta_dict:
                meta_dict["actions"] = []
            if "person" not in meta_dict:
                meta_dict["person"] = []
            if "camera" not in meta_dict:
                meta_dict["camera"] = []

            # Append extracted data to the dictionary
            meta_dict["actions"].append(action)
            meta_dict["person"].append(int(p))
            meta_dict["camera"].append(int(c))
        else:
            meta_dict["actions"].append(None)
            meta_dict["person"].append(None)
            meta_dict["camera"].append(None)
            print(f"Filename pattern not matched for: {filepath}")

    return True


def save_pickle(data, file_path):
    """
    Save the modified data back to a pickle file.
    """
    try:
        joblib.dump(data, file_path)
        print(f"File saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")


def main(file_path):
    # Load data from the pickle file
    data = load_pickle(file_path)
    if data is not None:
        # Assuming the dictionary is the last element in the list
        meta_dict = data[-1]

        # Update the meta information dictionary
        if update_meta_information(meta_dict):
            # Save the updated data back to a pickle file
            save_pickle(data, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a pickle file.")
    parser.add_argument(
        "--file_path",
        default="/lsdf/data/activity/synthetic_privacy/features/TSH/train_real_baseline/real_train_set.pkl",
        type=str,
        required=False,
        help="Path to the pickle file",
    )
    args = parser.parse_args()

    main(args.file_path)
