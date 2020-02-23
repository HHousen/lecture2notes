import os, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shared_functions import download_video

method = sys.argv[1]
csv_path = Path("../videos-dataset.csv")
output_dir_yt = "../videos/%\(id\)s/%\(id\)s.%\(ext\)s"
video_dir = Path("../videos/")

if method == "csv":
    # python video_downloader.py csv
    df = pd.read_csv(csv_path, index_col=0)
    for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Downloading Videos"):
        if row['downloaded'] == False:
            download_video(row, video_dir, output_dir_yt)
            row['downloaded'] = True # NOT WORKING
elif method == "youtube":
    # python video_downloader.py youtube 1Qws70XGSq4
    video_id = sys.argv[2]
    os.system('youtube-dl ' + video_id + ' -o ' + output_dir_yt)

df.to_csv(csv_path)
