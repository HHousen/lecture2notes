import shutil, sys, os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from shared_functions import download_video
from shared_functions import get_sec, get_length, get_extract_every_x_seconds

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../../End-To-End'))
# Even hackier hack to allow imported scripts to import from "Models/slide-classifier" directory
sys.path.insert(1, os.path.join(sys.path[0], '../../Models/slide-classifier'))
from frames_extractor import extract_frames
from slide_classifier import classify_frames

download_csv_path = Path("../mass-download-list.csv")
download_df = pd.read_csv(download_csv_path, index_col=0)

results_csv_path = Path("../mass-download-results.csv")
if results_csv_path.is_file():
    results_df = pd.read_csv(results_csv_path, index_col=0)
else:
    results_df = pd.DataFrame(columns=["video_id", "average_certainty", "num_incorrect", "percent_incorrect", "certainty_array"])


for index, row in tqdm(download_df.iterrows(), total=len(download_df.index), desc="Processing Videos"):
    if not row["downloaded"]:
        video_id = row['video_id']
        print("> Mass Data Collector: Video ID is " + video_id)

        output_dir_yt = "../mass-download-temp/" + video_id + "/%\(id\)s.%\(ext\)s"
        root_process_folder = Path("../mass-download-temp/") / video_id
        
        print("> Mass Data Collector: Video Root Process Folder is " + str(root_process_folder))

        print("> Mass Data Collector: Starting video " + video_id + " download")
        download_video(row, root_process_folder, output_dir_yt)
        video_path = root_process_folder / (row['video_id'] + ".mp4")
        print("> Mass Data Collector: Video " + video_id + " downloaded to " + str(video_path))

        # Extract frames
        quality = 5
        output_path = root_process_folder / "frames"

        length = get_length(video_path)
        length_in_seconds = get_sec(length)
        extract_every_x_seconds = str(get_extract_every_x_seconds(length_in_seconds))

        print("> Mass Data Collector: Extracting frames every " + str(extract_every_x_seconds) + " seconds at quality " + str(quality) + " to " + str(output_path))
        extract_frames(video_path, quality, output_path, extract_every_x_seconds)
        print("> Mass Data Collector: Frames extraced successfully")

        # Classify frames
        print("> Mass Data Collector: Classify frames")
        _, certainties, percent_wrong = classify_frames(output_path, do_move=False)

        average_certainty = sum(certainties) / len(certainties)
        num_incorrect = len([i for i in certainties if i < 0.70])
        percent_incorrect = (num_incorrect / len(certainties)) * 100
        print("> Mass Data Collector: Frames classified successfully. Average certainty is " + str(average_certainty))

        results_df.loc[len(results_df.index)]=[video_id, average_certainty, num_incorrect, percent_incorrect, certainties]
        results_df.to_csv(results_csv_path)

        download_df.at[index, "downloaded"] = True
        download_df.to_csv(download_csv_path)

        shutil.rmtree(root_process_folder)