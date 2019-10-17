import os
import sys
import pandas as pd

method = sys.argv[1]
csv_path = "../dataset.csv"
output_dir="../videos/%\(id\)s/%\(id\)s.%\(ext\)s"

if method == "csv":
    # python youtube_downloader.py csv
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        if row['downloaded'] == False:
            video_id = row['video_id']
            os.system('youtube-dl ' + video_id + ' -o ' + output_dir)
            row['downloaded'] = "true"
else:
    # python youtube_downloader.py 1Qws70XGSq4
    os.system('youtube-dl ' + method + ' -o ' + output_dir)