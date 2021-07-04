import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

videos_dir = Path("../videos")
data_dir = Path("../classifier-data")
csv_path = Path("../sort_file_map.csv")

if csv_path.is_file():
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.DataFrame(columns=["video_id", "filename", "category"])

if sys.argv[1] == "make":
    # create the file mapping
    videos = os.listdir(videos_dir)
    for item in tqdm(videos, total=len(videos), desc="Creating File Mapping - Videos"):
        current_dir = videos_dir / item
        frames_sorted_dir = current_dir / "frames_sorted"
        if os.path.isdir(current_dir) and os.path.exists(frames_sorted_dir):
            frames_sorted = os.listdir(frames_sorted_dir)
            desc = "Creating File Mapping - " + str(item)
            for category in tqdm(frames_sorted, total=len(frames_sorted), desc=desc):
                category_path = frames_sorted_dir / category
                frames = os.listdir(category_path)
                desc = "Creating File Mapping - " + str(category)
                for frame in tqdm(frames, total=len(frames), desc=desc):
                    df.loc[len(df.index)] = [item, frame, category]
    df.to_csv(csv_path)

elif sys.argv[1] == "make_compiled":
    categories = os.listdir(data_dir)
    num_categories = len(categories)

    for idx, category in enumerate(categories):
        category_path = data_dir / category
        frames = os.listdir(category_path)
        desc = (
            "Creating File Mapping Category - "
            + str(category)
            + " ("
            + str(idx + 1)
            + "/"
            + str(num_categories)
            + ") "
        )
        for frame in tqdm(frames, total=len(frames), desc=desc):
            # Filename format is [video_id]-img_[frame_index].jpg where [video_id] is 11 characters long
            # This check removes images from slide PDFs that were mereged into the dataset
            if frame[11] == "-":
                video_id = frame[:11]
                df.loc[len(df.index)] = [video_id, frame, category]
    df.to_csv(csv_path)

else:  # if sys.argv[1] == "sort"
    for index, row in df.iterrows():
        video_id = row["video_id"]
        filename = row["filename"]
        category = row["category"]

        current_video_path = videos_dir / video_id
        current_file_path = current_video_path / "frames" / filename
        sorted_file_dir = current_video_path / "frames_sorted" / category
        sorted_file_path = sorted_file_dir / filename
        if not os.path.exists(sorted_file_dir):
            os.makedirs(sorted_file_dir)
        shutil.move(str(current_file_path), str(sorted_file_path))
