import logging
import os
import sys
import traceback

import youtube_dl

# from io import StringIO

logger = logging.getLogger(__name__)


def download_video_yt(video_id, output_dir_yt, resolution=None):
    # youtube_dl_warnings_logger = logging.getLogger("youtube-dl")
    # log_stream = StringIO()
    # handler = logging.StreamHandler(log_stream)
    # handler.setLevel(logging.WARN)
    # youtube_dl_warnings_logger.addHandler(handler)

    if resolution:
        yt_format_string = (
            "bestvideo[height<=" + str(resolution) + "][ext=mp4]+bestaudio[ext=m4a]/mp4"
        )
    else:
        yt_format_string = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"

    ydl_opts = {
        "format": yt_format_string,
        "outtmpl": output_dir_yt,
        # "logger": youtube_dl_warnings_logger,
    }
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    tries = 0
    while tries < 3:
        try:
            ydl.download([video_id])
        except youtube_dl.utils.DownloadError:
            tries += 1
            logger.info("Try Number " + str(tries))
            if tries == 3:
                traceback.print_exc()
                sys.exit(1)
        # break out of the loop if successful
        else:
            break


def download_video(row, video_dir, output_dir_yt, resolution=None):
    video_id = row["video_id"]

    if row["provider"] == "youtube":
        download_video_yt(video_id, output_dir_yt, resolution)

    elif row["provider"] == "website":
        download_link = row["download_link"]
        file_extension = download_link.split(".")[-1]
        output_dir = video_dir / video_id

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_website = output_dir / (video_id + "." + file_extension)
        logger.info("Saving to " + str(output_file_website))
        os.system("wget -O " + str(output_file_website) + " " + download_link)


def get_sec(time_str):
    """Get Seconds from time."""
    time_str = time_str.split(".")[0]  # remove milliseconds
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_length(filename):
    command = (
        "ffmpeg -i "
        + str(filename)
        + " 2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//"
    )
    result = os.popen(command).read()
    return result


def get_extract_every_x_seconds(seconds):
    # Specifying `number of frames wanted` instead of `extract every x seconds` because
    # `number of frames wanted` will scale to the length of the video. Longer videos are
    # likely to stay focused on one subject longer than shorter videos.

    # Default is 200 frames
    num_frames_wanted = 200

    # Exception for very short videos
    if seconds < 1200:  # 20 minutes
        num_frames_wanted = 100

    # Exception for very long videos
    if seconds > 4800:  # 80 minuets
        num_frames_wanted = 300

    return seconds / num_frames_wanted
