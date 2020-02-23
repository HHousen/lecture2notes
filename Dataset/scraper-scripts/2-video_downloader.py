import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from shared_functions import download_video

METHOD = sys.argv[1]
CSV_PATH = Path("../videos-dataset.csv")
OUTPUT_DIR_YT = "../videos/%\(id\)s/%\(id\)s.%\(ext\)s" #pylint: disable=anomalous-backslash-in-string
VIDEO_DIR = Path("../videos/")

if METHOD == "csv":
    # python video_downloader.py csv
    df = pd.read_csv(CSV_PATH, index_col=0)

    not_downloaded_df = df.loc[df['downloaded'] == False]
    for index, row in tqdm(not_downloaded_df.iterrows(), total=len(not_downloaded_df.index), desc="Downloading Videos"):
        download_video(row, VIDEO_DIR, OUTPUT_DIR_YT)
        df.at[index, "downloaded"] = True
    df.to_csv(CSV_PATH)
elif METHOD == "youtube":
    # python video_downloader.py youtube 1Qws70XGSq4
    video_id = sys.argv[2]
    os.system('youtube-dl ' + video_id + ' -o ' + OUTPUT_DIR_YT)
