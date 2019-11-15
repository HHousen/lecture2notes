import os, sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

method = sys.argv[1]
csv_path = Path("../videos-dataset.csv")
output_dir_yt = "../videos/%\(id\)s/%\(id\)s.%\(ext\)s"
video_dir = Path("../videos/")

if method == "csv":
    # python youtube_downloader.py csv
    df = pd.read_csv(csv_path, index_col=0)
    for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Downloading Videos"):
        if row['downloaded'] == False:
            video_id = row['video_id']
            if row['provider'] == "youtube":
                os.system('youtube-dl -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" ' + video_id + ' -o ' + output_dir_yt)
            elif row['provider'] == "website":
                download_link = row['download_link']
                file_extension = download_link.split(".")[-1]
                output_dir = video_dir / video_id

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file_website = output_dir / (video_id + '.' + file_extension)
                print("Saving to " + str(output_file_website))
                os.system('wget -O ' + str(output_file_website) + ' ' + download_link)
            row['downloaded'] = True # NOT WORKING
elif method == "youtube":
    # python youtube_downloader.py youtube 1Qws70XGSq4
    video_id = sys.argv[2]
    os.system('youtube-dl ' + video_id + ' -o ' + output_dir_yt)

df.to_csv(csv_path)
