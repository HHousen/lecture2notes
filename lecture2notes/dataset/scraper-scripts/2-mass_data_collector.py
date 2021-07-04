import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from shared_functions import (
    download_video,
    get_extract_every_x_seconds,
    get_length,
    get_sec,
)
from tqdm import tqdm

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], "../../End-To-End"))
# Even hackier hack to allow imported scripts to import from "models/slide_classifier" directory
sys.path.insert(1, os.path.join(sys.path[0], "../../models/slide_classifier"))
from frames_extractor import extract_frames  # noqa: E402
from slide_classifier import classify_frames  # noqa: E402

parser = argparse.ArgumentParser(description="Mass Data Collector")
parser.add_argument(
    "-k",
    "--top_k",
    type=int,
    metavar="K",
    default=None,
    help="Add the top `k` most uncertain videos to the videos-dataset.",
)
parser.add_argument(
    "-nr",
    "--no_remove",
    action="store_true",
    help="""Don't remove the videos after they have been processed. This makes it
                    faster to manually look through the most uncertain videos since they don't have
                    to be redownloaded, but it will use more disk space.""",
)
parser.add_argument(
    "-r",
    "--resolution",
    type=int,
    default=None,
    help="The resolution of the videos to download. Default is maximum resolution.",
)
parser.add_argument(
    "-p",
    "--pause",
    action="store_true",
    help="Pause after each video has been processed but before deletion.",
)

args = parser.parse_args()

DOWNLOAD_CSV_PATH = Path("../mass-download-list.csv")
DOWNLOAD_DF = pd.read_csv(DOWNLOAD_CSV_PATH, index_col=0)

# Remove duplicates from `DOWNLOAD_DF`
DOWNLOAD_DF.drop_duplicates(
    subset="video_id", keep="first", inplace=True, ignore_index=True
)
DOWNLOAD_DF.to_csv(DOWNLOAD_CSV_PATH)

RESULTS_CSV_PATH = Path("../mass-download-results.csv")
if RESULTS_CSV_PATH.is_file():
    RESULTS_DF = pd.read_csv(RESULTS_CSV_PATH, index_col=0)
else:
    RESULTS_DF = pd.DataFrame(
        columns=[
            "video_id",
            "average_certainty",
            "num_incorrect",
            "percent_incorrect",
            "certainty_array",
        ]
    )


def process_videos():
    for index, row in tqdm(
        DOWNLOAD_DF.iterrows(), total=len(DOWNLOAD_DF.index), desc="Processing Videos"
    ):
        if not row["downloaded"]:
            video_id = row["video_id"]
            print("> Mass Data Collector: Video ID is " + video_id)

            output_dir_yt = "../mass-download-temp/" + video_id + "/%(id)s.%(ext)s"
            root_process_folder = Path("../mass-download-temp/") / video_id

            print(
                "> Mass Data Collector: Video Root Process Folder is "
                + str(root_process_folder)
            )

            print("> Mass Data Collector: Starting video " + video_id + " download")
            download_video(
                row, root_process_folder, output_dir_yt, resolution=args.resolution
            )
            video_path = root_process_folder / (row["video_id"] + ".mp4")
            print(
                "> Mass Data Collector: Video "
                + video_id
                + " downloaded to "
                + str(video_path)
            )

            # Extract frames
            quality = 5
            output_path = root_process_folder / "frames"

            length = get_length(video_path)
            length_in_seconds = get_sec(length)
            extract_every_x_seconds = str(
                get_extract_every_x_seconds(length_in_seconds)
            )

            print(
                "> Mass Data Collector: Extracting frames every "
                + str(extract_every_x_seconds)
                + " seconds at quality "
                + str(quality)
                + " to "
                + str(output_path)
            )
            extract_frames(video_path, quality, output_path, extract_every_x_seconds)
            print("> Mass Data Collector: Frames extraced successfully")

            # Classify frames
            print("> Mass Data Collector: Classify frames")
            _, certainties, percent_incorrect = classify_frames(
                output_path, do_move=False, incorrect_threshold=0.70
            )

            average_certainty = sum(certainties) / len(certainties)
            num_incorrect = len([i for i in certainties if i < 0.70])
            print(
                "> Mass Data Collector: Frames classified successfully. Average certainty is "
                + str(average_certainty)
            )

            RESULTS_DF.loc[len(RESULTS_DF.index)] = [
                video_id,
                average_certainty,
                num_incorrect,
                percent_incorrect,
                certainties,
            ]
            RESULTS_DF.to_csv(RESULTS_CSV_PATH)

            DOWNLOAD_DF.at[index, "downloaded"] = True
            DOWNLOAD_DF.to_csv(DOWNLOAD_CSV_PATH)

            if args.pause:
                input("Paused. Press enter to continue...")

            if not args.no_remove:
                shutil.rmtree(root_process_folder)


def add_top_k(k=10, remove=True):
    RESULTS_DF.sort_values("average_certainty", ascending=True, inplace=True)
    VIDEOS_DATASET_CSV_PATH = "../videos-dataset.csv"
    VIDEOS_DATASET_DF = pd.read_csv(VIDEOS_DATASET_CSV_PATH, index_col=0)

    most_uncertain_k = RESULTS_DF.head(k)

    for _, row in tqdm(
        most_uncertain_k.iterrows(),
        total=len(most_uncertain_k.index),
        desc="Processing Videos",
    ):
        row_to_add = DOWNLOAD_DF.loc[
            DOWNLOAD_DF["video_id"] == row["video_id"]
        ].squeeze()
        if not row_to_add.empty:
            row_to_add["downloaded"] = False
            VIDEOS_DATASET_DF.loc[len(VIDEOS_DATASET_DF.index)] = row_to_add

    VIDEOS_DATASET_DF.to_csv(VIDEOS_DATASET_CSV_PATH)

    if remove:
        RESULTS_DF.drop(RESULTS_DF.head(k).index, inplace=True)
        RESULTS_DF.reset_index(inplace=True, drop=True)
        RESULTS_DF.to_csv(RESULTS_CSV_PATH)


if args.top_k:
    add_top_k(args.top_k)
else:
    process_videos()
