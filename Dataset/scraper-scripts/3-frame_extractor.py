import os
import sys
import subprocess

def command(input_video_path, extract_every_x_seconds, quality, output_path, video_id):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    command = 'ffmpeg -i ' + input_video_path + ' -vf "fps=1/' + extract_every_x_seconds + '" -q:v ' + quality + ' ' + output_path + '/' + video_id + '-img_%03d.jpg'

    print("Running Command: " + command)
    os.system(command)

def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    for x in result.stdout.readlines():
        if b'Duration' in x:
            return x.decode("utf-8")[12:23]
    return False

videos_dir = '../videos'

if sys.argv[1] == "auto":
    for item in os.listdir(videos_dir):
        current_dir = os.path.join(videos_dir, item)
        frames_dir = current_dir + "/frames"
        frames_sorted_dir = current_dir + "/frames_sorted"
        if os.path.isdir(current_dir) and not os.path.exists(frames_dir) and not os.path.exists(frames_sorted_dir):
            print("Video Folder " + item + " Without Frames Directory Found!")
            os.makedirs(frames_dir)
            video_id = item
            input_video_path = "../videos/" + video_id + "/" + video_id + ".mp4"
            extract_every_x_seconds = str(20)
            quality = str(5) # 2 is best
            output_path = "../videos/" + video_id + "/frames"
            command(input_video_path, extract_every_x_seconds, quality, output_path, video_id)

else:
    video_id = sys.argv[1]
    input_video_path = "../videos/" + video_id + "/" + video_id + ".mp4"
    extract_every_x_seconds = sys.argv[2]
    quality = sys.argv[3] # 2 is best
    output_path = "../videos/" + video_id + "/frames"

    #num_frames = os.popen('ffprobe -show_streams ' + input_video_path + ' | grep "^nb_frames" | cut -d "=" -f 2').read()
    # https://stackoverflow.com/questions/8679390/ffmpeg-extracting-20-images-from-a-video-of-variable-length
    length = getLength(input_video_path)
    length_good_check = input("Input Video: " + input_video_path + "\nOutput Path: " + output_path + "\nVideo Length: " + length + "\nSelected Quality: " + quality + "\nExtracing Every " + extract_every_x_frames + " frames" + "\nContinue? ")

    if length_good_check == 'y':
        command(input_video_path, extract_every_x_seconds, quality, output_path, video_id)

        # python frame_extractor.py VT2o4KCEbes 20 5
    else:
        print("Exiting...")