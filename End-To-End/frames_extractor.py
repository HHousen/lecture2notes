import os
from pathlib import Path
from helpers import make_dir_if_not_exist

def extract_frames(input_video_path, quality, output_path, extract_every_x_seconds):
    """Extracts frames from `input_video_path` at quality level `quality` (best quality is 2) every `extract_every_x_seconds seconds` and saves them to `output_path`"""
    quality = str(quality)
    output_path = str(output_path)
    extract_every_x_seconds = str(extract_every_x_seconds)
    print("> Frames Extractor: Received inputs\ninput_video_path=" + str(input_video_path) + "\nquality=" + str(quality) + "\noutput_path=" + str(output_path))
    make_dir_if_not_exist(output_path)
    command = 'ffmpeg -i ' + str(input_video_path) + ' -vf "fps=1/' + str(extract_every_x_seconds) + '" -q:v ' + str(quality) + ' ' + str(output_path) + '/img_%05d.jpg'

    print("> Frames Extractor: Running command: " + command)
    os.system(command)
    print("> Frames Extractor: Frame extraction successful. Returning output_path=" + str(output_path))
    return output_path
