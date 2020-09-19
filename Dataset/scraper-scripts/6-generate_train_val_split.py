import os
import sys
import shutil
import random
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
    for category in tqdm(classifier_data, total=len(classifier_data), desc="Categories"):
        current_dir = classifier_data_dir / category
        if os.path.isdir(current_dir): 
            current_category_data = os.listdir(current_dir)
            for image in tqdm(current_category_data, total=len(current_category_data), desc="Images"):
                split_idx = None
                
                for idx, split in enumerate(cv_splits):
                    if any([x in image for x in split]):
                        split_idx = str(idx)
                
                if split_idx is None:
                    for idx, split in enumerate(cv_splits_slides):
                        if any([x in image for x in split]):
                            split_idx = str(idx)
                
                split_name = "split_" + split_idx
                os.makedirs(output_dir_cv / split_name / category, exist_ok=True)
                shutil.copy(current_dir / image, output_dir_cv / split_name / category / image)

else:
    frac = 0.2

    # generate a list of indices to exclude. Turn in into a set for O(1) lookup time
    inds = set(random.sample(list(range(len(video_ids))), int(frac * len(video_ids))))

    # use `enumerate` to get list indices as well as elements. 
    # Filter by index, but take only the elements
    video_ids_train = [n for i,n in enumerate(video_ids) if i not in inds]
    video_ids_test = [n for i,n in enumerate(video_ids) if i in inds]

    print("Training Video IDs: %s" % ", ".join(video_ids_train))
    print("Validation Video IDs: %s" % ", ".join(video_ids_test))

    classifier_data = os.listdir(classifier_data_dir)
    for category in tqdm(classifier_data, total=len(classifier_data), desc="Categories"):
        current_dir = classifier_data_dir / category
        if os.path.isdir(current_dir): 
            current_category_data = os.listdir(current_dir)
            for image in tqdm(current_category_data, total=len(current_category_data), desc="Images"):
                if any([x in image for x in video_ids_test]):  # image should be in validation set
                    os.makedirs(output_dir / "val" / category, exist_ok=True)
                    shutil.copy(current_dir / image, output_dir / "val" / category / image)
                else:  # image should be in training set
                    os.makedirs(output_dir / "train" / category, exist_ok=True)
                    shutil.copy(current_dir / image, output_dir / "train" / category / image)

print("Images Sorted Successfully")