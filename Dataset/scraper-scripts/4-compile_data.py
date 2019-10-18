import os
import subprocess as sbp

videos_dir = '../videos'
slides_dir = '../slides/images'
data_dir = '../classifier-data'

for item in os.listdir(videos_dir):
    current_dir = os.path.join(videos_dir, item)
    frames_sorted_dir = current_dir + "/frames_sorted"
    if os.path.isdir(current_dir) and os.path.exists(frames_sorted_dir):
        frames_sorted = os.listdir(frames_sorted_dir)
        for frame in frames_sorted:
            category_path = os.path.join(frames_sorted_dir, frame)
            copy_command = 'cp -r ' + category_path + ' ' + data_dir + '/.'
            sbp.Popen(copy_command, shell=True)

for item in os.listdir(slides_dir):
    current_dir = os.path.join(slides_dir, item)
    copy_command = 'cp -r ' + current_dir + '/. ' + data_dir + '/slide/.'
    sbp.Popen(copy_command, shell=True)
