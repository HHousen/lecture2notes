import os
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

classifier_data_dir = Path("../classifier-data")
output_dir = Path("../classifier-data-train-test")
output_dir_cv = Path("../classifier-data-cv")

video_dataset_path = Path("../videos-dataset.csv")
df = pd.read_csv(video_dataset_path, index_col=0)

video_ids = list(df["video_id"])

slide_ids = os.listdir("../slides/images")

random.shuffle(video_ids)

if sys.argv[1] == "cv":
    cv_splits = np.array_split(np.array(video_ids), 3)
    cv_splits_slides = np.array_split(np.array(slide_ids), 3)
    print("CV Splits: %s" % cv_splits)
    classifier_data = os.listdir(classifier_data_dir)
    for category in tqdm(
        classifier_data, total=len(classifier_data), desc="Categories"
    ):
        current_dir = classifier_data_dir / category
        if os.path.isdir(current_dir):
            current_category_data = os.listdir(current_dir)
            for image in tqdm(
                current_category_data, total=len(current_category_data), desc="Images"
            ):
                split_idx = None

                for idx, split in enumerate(cv_splits):
                    if any(x in image for x in split):
                        split_idx = str(idx)

                if split_idx is None:
                    for idx, split in enumerate(cv_splits_slides):
                        if any(x in image for x in split):
                            split_idx = str(idx)

                split_name = "split_" + split_idx
                os.makedirs(output_dir_cv / split_name / category, exist_ok=True)
                shutil.copy(
                    current_dir / image, output_dir_cv / split_name / category / image
                )

elif sys.argv[1] == "split":
    if sys.argv[2] == "auto_determine_best":
        video_ids_dict = {}
        for video_id in video_ids:
            video_ids_dict[video_id] = {}

        total_items = {}

        classifier_data = os.listdir(classifier_data_dir)
        for category in tqdm(
            classifier_data, total=len(classifier_data), desc="Categories"
        ):
            current_dir = classifier_data_dir / category
            if os.path.isdir(current_dir):
                current_category_data = os.listdir(current_dir)
                category_name = str(current_dir).split("/")[-1]
                total_items_in_category = len(current_category_data)
                total_items[category_name] = total_items_in_category
                for video_id in video_ids:
                    num_in_category = len(
                        [x for x in current_category_data if video_id in x]
                    )
                    video_ids_dict[video_id][category_name] = num_in_category

        last_deviation = 1000
        try:
            for idx, val in tqdm(enumerate(combinations(video_ids, 16))):
                total_deviations = []
                for category in classifier_data:
                    total_val_category = sum(
                        video_ids_dict[video_id][category] for video_id in val
                    )
                    val_percent = total_val_category / total_items[category]

                    deviation_from_frac = abs(0.2 - val_percent)
                    total_deviations.append(deviation_from_frac)

                avg_deviation = sum(total_deviations) / len(total_deviations)
                if avg_deviation < last_deviation:
                    last_deviation = avg_deviation
                    best_set = (val, avg_deviation)

                if idx % 200_000 == 0:
                    print("Best Deviation: %s" % last_deviation)

        except KeyboardInterrupt:
            pass

        print(best_set)

        video_ids_test = best_set[0]
        video_ids_train = [x for x in video_ids if x not in video_ids_test]

    else:
        frac = 0.2

        # generate a list of indices to exclude. Turn in into a set for O(1) lookup time
        inds = set(
            random.sample(list(range(len(video_ids))), int(frac * len(video_ids)))
        )

        # use `enumerate` to get list indices as well as elements.
        # Filter by index, but take only the elements
        video_ids_train = [n for i, n in enumerate(video_ids) if i not in inds]
        video_ids_test = [n for i, n in enumerate(video_ids) if i in inds]
        # video_ids_train = [x for x in video_ids if x not in video_ids_test]

    print("Training Video IDs: %s" % ", ".join(video_ids_train))
    print("Validation Video IDs: %s" % ", ".join(video_ids_test))

    classifier_data = os.listdir(classifier_data_dir)
    train_val_split_stats = {}
    for category in tqdm(
        classifier_data, total=len(classifier_data), desc="Categories"
    ):
        current_dir = classifier_data_dir / category
        if os.path.isdir(current_dir):
            current_category_data = os.listdir(current_dir)
            num_val = 0
            num_train = 0
            for image in tqdm(
                current_category_data, total=len(current_category_data), desc="Images"
            ):
                if any(
                    x in image for x in video_ids_test
                ):  # image should be in validation set
                    os.makedirs(output_dir / "val" / category, exist_ok=True)
                    shutil.copy(
                        current_dir / image, output_dir / "val" / category / image
                    )
                    num_val += 1
                else:  # image should be in training set
                    os.makedirs(output_dir / "train" / category, exist_ok=True)
                    shutil.copy(
                        current_dir / image, output_dir / "train" / category / image
                    )
                    num_train += 1

            percent = num_val / (num_train + num_val)
            train_val_split_stats[current_dir] = {
                "train": num_train,
                "val": num_val,
                "percent_in_val": percent,
            }

    for key, value in train_val_split_stats.items():
        print(str(key).split("/")[-1] + ": " + str(value))

print("Images Sorted Successfully")
