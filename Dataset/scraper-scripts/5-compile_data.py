import os
import subprocess as sbp
from pathlib import Path

videos_dir = Path('../videos')
slides_dir = Path('../slides/images')
data_dir = Path('../classifier-data')

for item in os.listdir(videos_dir):
    current_dir = videos_dir / item
    frames_sorted_dir = current_dir / "frames_sorted"
    if os.path.isdir(current_dir) and os.path.exists(frames_sorted_dir):
        frames_sorted = os.listdir(frames_sorted_dir)
        for category in frames_sorted:
            category_path = os.path.join(frames_sorted_dir, category)
            copy_command = 'cp -r ' + str(category_path) + ' ' + str(data_dir) + '/.'
            sbp.Popen(copy_command, shell=True)

for item in os.listdir(slides_dir):
    current_dir = os.path.join(slides_dir, item)
    copy_command = 'cp -r ' + str(current_dir) + '/. ' + str(data_dir) + '/slide/.'
    sbp.Popen(copy_command, shell=True)
