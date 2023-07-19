import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from shared_functions import download_video, download_video_yt
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], "../../end_to_end"))
from transcript_downloader import TranscriptDownloader  # noqa: E402

PARSER = argparse.ArgumentParser(description="Video Downloader")

PARSER.add_argument(
    "method",
    choices=["csv", "youtube"],
    help="""`csv`: Download all videos that have not been marked as downloaded from the `videos-dataset.csv`.
                    `youtube`: download the specified video from YouTube with id ``--video_id`.""",
)
PARSER.add_argument(
    "--video_id",
    default=None,
    type=str,
    help="The YouTube video id to download if `method` is `youtube`.",
)
PARSER.add_argument(
    "--transcript",
    action="store_true",
    help="Download the transcript INSTEAD of the video for each entry in `videos-dataset.csv`. This ignores the `downloaded` column in the CSV and will not download videos.",
)
PARSER.add_argument(
    "-r",
    "--resolution",
    type=int,
    default=None,
    help="The resolution of the videos to download. Default is maximum resolution.",
)
PARSER.add_argument(
    "-l",
    "--log",
    dest="logLevel",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level (default: 'Info').",
)
ARGS = PARSER.parse_args()

logging.basicConfig(
    format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
    level=logging.getLevelName(ARGS.logLevel),
)

if ARGS.method == "youtube" and ARGS.video_id is None:
    PARSER.error(
        "If the `youtube` method is used then `--video_id` must be set to the id of the video to download."
    )

CSV_PATH = Path("../videos-dataset.csv")
OUTPUT_DIR_YT = "../videos/%(id)s/%(id)s.%(ext)s"
VIDEO_DIR = Path("../videos/")

if ARGS.transcript:
    transcript_dir = Path("../transcripts")
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)

    downloader = TranscriptDownloader()

if ARGS.method == "csv":
    # python video_downloader.py csv
    df = pd.read_csv(CSV_PATH, index_col=0)

    if ARGS.transcript:
        youtube_df = df.loc[df["provider"] == "youtube"]
        for index, row in tqdm(
            youtube_df.iterrows(),
            total=len(youtube_df.index),
            desc="Downloading Transcripts",
        ):
            video_id = row["video_id"]
            output_path = transcript_dir / (video_id + ".vtt")
            if not output_path.is_file():
                output = downloader.download(video_id, output_path)

    else:
        not_downloaded_df = df.loc[
            df["downloaded"] == False  # noqa: E712
        ]  # pylint: disable=singleton-comparison
        for index, row in tqdm(
            not_downloaded_df.iterrows(),
            total=len(not_downloaded_df.index),
            desc="Downloading Videos",
        ):
            download_video(row, VIDEO_DIR, OUTPUT_DIR_YT, resolution=ARGS.resolution)
            df.at[index, "downloaded"] = True

    df.to_csv(CSV_PATH)
elif ARGS.method == "youtube":
    # python video_downloader.py youtube 1Qws70XGSq4
    download_video_yt(ARGS.video_id, OUTPUT_DIR_YT, resolution=ARGS.resolution)
