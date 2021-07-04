import os
import sys
from pathlib import Path

from shared_functions import get_extract_every_x_seconds, get_length, get_sec
from tqdm import tqdm


def command(input_video_path, extract_every_x_seconds, quality, output_path, video_id):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    command = (
        "ffmpeg -i "
        + str(input_video_path)
        + ' -vf "fps=1/'
        + str(extract_every_x_seconds)
        + '" -q:v '
        + quality
        + " "
        + str(output_path)
        + "/"
        + video_id
        + "-img_%03d.jpg"
    )

    print("Running Command: " + command)
    os.system(command)


videos_dir = Path("../videos")

if sys.argv[1] == "auto":
    videos = os.listdir(videos_dir)
    for item in tqdm(videos, total=len(videos), desc="Extracting Frames"):
        current_dir = os.path.join(videos_dir, item)
        frames_dir = current_dir + "/frames"
        frames_sorted_dir = current_dir + "/frames_sorted"
        if (
            os.path.isdir(current_dir)
            and not os.path.exists(frames_dir)
            and not os.path.exists(frames_sorted_dir)
        ):
            print("Video Folder " + item + " Without Frames Directory Found!")
            os.makedirs(frames_dir)
            video_id = item
            input_video_dir = videos_dir / video_id
            files_in_dir = [
                filename
                for filename in os.listdir(input_video_dir)
                if filename.startswith(video_id)
            ]
            input_video_path = input_video_dir / files_in_dir[0]

            length = get_length(input_video_path)
            length_in_seconds = get_sec(length)
            extract_every_x_seconds = str(
                get_extract_every_x_seconds(length_in_seconds)
            )

            quality = str(5)  # 2 is best
            output_path = videos_dir / video_id / "frames"
            command(
                input_video_path,
                extract_every_x_seconds,
                quality,
                output_path,
                video_id,
            )

            if len(sys.argv) > 2 and sys.argv[2] == "remove":
                os.remove(input_video_path)

else:
    video_id = sys.argv[1]
    input_video_path = videos_dir / video_id / (video_id + ".mp4")
    extract_every_x_seconds = sys.argv[2]
    quality = sys.argv[3]  # 2 is best
    output_path = videos_dir / video_id / "frames"

    # num_frames = os.popen('ffprobe -show_streams ' + input_video_path + ' | grep "^nb_frames" | cut -d "=" -f 2').read()
    # https://stackoverflow.com/questions/8679390/ffmpeg-extracting-20-images-from-a-video-of-variable-length
    length = get_length(input_video_path)

    length_in_seconds = get_sec(length)

    if extract_every_x_seconds == "auto":
        extract_every_x_seconds = get_extract_every_x_seconds(length_in_seconds)

    num_frames_to_be_extracted = length_in_seconds / extract_every_x_seconds

    length_good_check = input(
        "Input Video: "
        + str(input_video_path)
        + "\nOutput Path: "
        + str(output_path)
        + "\nVideo Length: "
        + str(length)
        + "\nSelected Quality: "
        + quality
        + "\nExtracing Every "
        + str(extract_every_x_seconds)
        + " seconds"
        + "\nNumber of Frames to be Extracted: "
        + str(num_frames_to_be_extracted)
        + "\nContinue? "
    )

    if length_good_check == "y":
        command(
            input_video_path, extract_every_x_seconds, quality, output_path, video_id
        )

        # python frame_extractor.py VT2o4KCEbes 20 5
    else:
        print("Exiting...")
