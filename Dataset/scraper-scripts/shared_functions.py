import os

def download_video(row, video_dir, output_dir_yt, resolution=None):
    video_id = row['video_id']

    if resolution:
        yt_format_string = "bestvideo[height<=" + str(resolution) + "][ext=mp4]+bestaudio[ext=m4a]/mp4"
    else:
        yt_format_string = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"

    if row['provider'] == "youtube":
        os.system('youtube-dl -f "' + yt_format_string + '" -o ' + output_dir_yt + ' -- ' + video_id)
    elif row['provider'] == "website":
        download_link = row['download_link']
        file_extension = download_link.split(".")[-1]
        output_dir = video_dir / video_id

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_website = output_dir / (video_id + '.' + file_extension)
        print("Saving to " + str(output_file_website))
        os.system('wget -O ' + str(output_file_website) + ' ' + download_link)

def get_sec(time_str):
    """Get Seconds from time."""
    time_str = time_str.split(".")[0] # remove milliseconds
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def get_length(filename):
    command = "ffmpeg -i " + str(filename) + " 2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//"
    result = os.popen(command).read()
    return result

def get_extract_every_x_seconds(seconds):
    # Specifying `number of frames wanted` instead of `extract every x seconds` because
    # `number of frames wanted` will scale to the length of the video. Longer videos are
    # likely to stay focused on one subject longer than shorter videos.

    # Default is 200 frames
    num_frames_wanted = 200

    # Exception for very short videos
    if seconds < 1200: # 20 minutes
        num_frames_wanted = 100

    # Exception for very long videos
    if seconds > 4800: # 80 minuets
        num_frames_wanted = 300

    return seconds / num_frames_wanted
