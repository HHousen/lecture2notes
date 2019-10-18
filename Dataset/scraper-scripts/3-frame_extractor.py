import os
import sys
import subprocess

def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    for x in result.stdout.readlines():
        if b'Duration' in x:
            return x.decode("utf-8")[12:23]
    return False

input_video_path = "../videos/" + sys.argv[1] + "/" + sys.argv[1] + ".mp4"
extract_every_x_frames = sys.argv[2]
quality = sys.argv[3] # 2 is best
output_path = "../videos/" + sys.argv[1] + "/frames"

length = getLength(input_video_path)
length_good_check = input("Input Video: " + input_video_path + "\nOutput Path: " + output_path + "\nVideo Length: " + length + "\nSelected Quality: " + quality + "\nExtracing Every " + extract_every_x_frames + " frames" + "\nContinue? ")

if length_good_check == 'y':
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    command = 'ffmpeg -i ' + input_video_path + ' -vf "fps=1/' + extract_every_x_frames + '" -q:v ' + quality + ' ' + output_path + '/' + sys.argv[1] + '-img_%03d.jpg'

    print("Running Command: " + command)
    os.system(command)

    # python frame_extractor.py VT2o4KCEbes 20 5
else:
    print("Exiting...")