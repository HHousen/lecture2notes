import os, sys, shutil
from pathlib import Path
import pandas as pd

videos_dir = Path('../videos')
csv_path = Path("../sort_file_map.csv")

if csv_path.is_file():
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.DataFrame(columns=["video_id","filename","category"])

if sys.argv[1] == "make":
    # create the file mapping
    for item in os.listdir(videos_dir):
        current_dir = videos_dir / item
        frames_sorted_dir = current_dir / "frames_sorted"
        if os.path.isdir(current_dir) and os.path.exists(frames_sorted_dir):
            frames_sorted = os.listdir(frames_sorted_dir)
            for category in frames_sorted:
                category_path = frames_sorted_dir / category
                for frame in os.listdir(category_path):
                    df.loc[len(df.index)]=[item,frame,category]
    df.to_csv(csv_path)
else: # if sys.argv[1] == "sort"
    for index, row in df.iterrows():
        video_id = row['video_id']
        filename = row['filename']
        category = row['category']

        current_video_path = videos_dir / video_id
        current_file_path = current_video_path / "frames" / filename
        sorted_file_dir = current_video_path / "frames_sorted" / category
        sorted_file_path = sorted_file_dir / filename
        if not os.path.exists(sorted_file_dir):
            os.makedirs(sorted_file_dir)
        shutil.move(str(current_file_path), str(sorted_file_path))