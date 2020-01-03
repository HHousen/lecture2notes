# Main process to convert video to notes (end-to-end)
# 1. Extract frames
# 2. Classify slides
# 3. OCR slides

import sys, os
import argparse
from shutil import rmtree
from pathlib import Path
from helpers import *


# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from custom_nnmodules import *

parser = argparse.ArgumentParser(description='End-to-End Conversion of Lecture Videos to Notes using ML')
parser.add_argument('video_path', metavar='DIR',
                    help='path to video')
parser.add_argument('-s', '--skip-to', default=0, type=int, metavar='N',
                    help='set to > 0 to skip specific processing steps')
parser.add_argument('-d', '--process_dir', default='./', type=str, metavar='PATH',
                    help='path to the proessing directory (where extracted frames and other files are saved), set to "automatic" to use the video\'s folder (default: ./)')
parser.add_argument('-id', '--auto-id', dest='auto_id', action='store_true',
                    help='automatically create a subdirectory in `process_dir` with a unique id for the video and change `process_dir` to this new directory')
parser.add_argument('-rm', '--remove', dest='remove', action='store_true',
                    help='remove `process_dir` once conversion is complete')

args = parser.parse_args()

if args.process_dir == "automatic":
    root_process_folder = os.path.dirname(args.video_path)
else:
    root_process_folder = Path(args.process_dir)
if args.auto_id:
    unique_id = gen_unique_id(args.video_path, 12)
    root_process_folder = root_process_folder / unique_id

# 1. Extract frames
if args.skip_to <= 1: 
    from frames_extractor import extract_frames
    quality = 5
    output_path = root_process_folder / "frames"
    extract_every_x_seconds = 1
    extract_frames(args.video_path, quality, output_path, extract_every_x_seconds)

# 2. Classify slides
if args.skip_to <= 2:
    from slide_classifier import classify_frames
    frames_dir = root_process_folder / "frames"
    frames_sorted_dir = classify_frames(frames_dir)

# 3. Cluster slides
if args.skip_to <= 3: 
    if args.skip_to >= 3: # if step 2 (classify slides) was skipped
        frames_sorted_dir = root_process_folder / "frames_sorted"
    slides_dir = frames_sorted_dir / "slide"
    from cluster import make_clusters
    cluster_dir = make_clusters(slides_dir)

# 4. OCR slides
if args.skip_to <= 4: 
    if args.skip_to >= 4: # if step 3 (cluster slides) was skipped
        frames_sorted_dir = root_process_folder / "frames_sorted"
    import ocr
    slides_folder = frames_sorted_dir / "slide"
    save_file = root_process_folder / "ocr.txt"
    results = ocr.all_in_folder(slides_folder)
    ocr.write_to_file(results, save_file)

if args.skip_to <= 5:
    import transcribe
    audio_input_path = args.video_path
    audio_output_path = root_process_folder / "audio.wav"
    transcript_output_file = root_process_folder / "audio.txt"
    transcribe.extract_audio(audio_input_path, audio_output_path)
    try:
        transcript = transcribe.transcribe_audio(audio_output_path)
    except:
        print("Audio transcription failed. Retry by running this script with the skip_to parameter set to 5")
    transcribe.write_to_file(transcript, transcript_output_file)

# if args.remove:
#     rmtree(root_process_folder)