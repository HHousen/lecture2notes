import os
import sys
from distutils.dir_util import copy_tree
from pathlib import Path

from tqdm import tqdm

videos_dir = Path("../videos")
slides_dir = Path("../slides/images")
data_dir = Path("../classifier-data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if sys.argv[1] == "all" or sys.argv[1] == "videos":
    videos = os.listdir(videos_dir)
    for item in tqdm(videos, total=len(videos), desc="Compiling Videos"):
        current_dir = videos_dir / item
        frames_sorted_dir = current_dir / "frames_sorted"
        if os.path.isdir(current_dir) and os.path.exists(frames_sorted_dir):
            frames_sorted = os.listdir(frames_sorted_dir)
            for category in frames_sorted:
                video_category_path = frames_sorted_dir / category
                data_category_path = data_dir / category
                copy_tree(str(video_category_path), str(data_category_path))

if sys.argv[1] == "all" or sys.argv[1] == "slides":
    slide_images = os.listdir(slides_dir)
    for item in tqdm(
        slide_images, total=len(slide_images), desc="Compiling Slideshow Images"
    ):
        current_dir = slides_dir / item
        data_dir_slide = data_dir / "slide"
        copy_tree(str(current_dir), str(data_dir_slide))
