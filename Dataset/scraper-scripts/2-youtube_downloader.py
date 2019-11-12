import os, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

method = sys.argv[1]
csv_path = Path("../videos-dataset.csv")
output_dir = "../videos/%\(id\)s/%\(id\)s.%\(ext\)s"

if method == "csv":
    # python youtube_downloader.py csv
    df = pd.read_csv(csv_path, index_col=0)
    for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Downloading Videos"):
        if row['downloaded'] == False:
            video_id = row['video_id']
            os.system('youtube-dl -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" ' + video_id + ' -o ' + output_dir)
            row['downloaded'] = True # NOT WORKING
else:
    # python youtube_downloader.py 1Qws70XGSq4
    os.system('youtube-dl ' + method + ' -o ' + output_dir)

df.to_csv(csv_path)
