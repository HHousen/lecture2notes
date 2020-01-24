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
parser.add_argument('-c', '--chunk', dest='chunk', action='store_true',
                    help='split the audio into small chunks on silence')
parser.add_argument('-tm', '--transcription_method', default="deepspeech", choices=["sphinx", "google", "youtube", "deepspeech"],
                    help='''specify the program that should be used for transcription. 
                    CMU Sphinx: use pocketsphinx (works offline)
                    Google Speech Recognition: probably will require chunking
                    YouTube: pull a video transcript from YouTube based on video_id
                    DeepSpeech: Use the deepspeech library (works offline with great accuracy)''')
parser.add_argument('--video_id', type=str, metavar='ID',
                    help='id of youtube video to get subtitles from')
parser.add_argument('--yt_convert_to_str', action='store_true',
                    help='if the method is `youtube` and this option is specified then the transcript will be saved as a txt file instead of a srt file.')
parser.add_argument('--deepspeech_model_dir', type=str, metavar='DIR',
                    help='path containing the DeepSpeech model files. See the documentation for details.')

args = parser.parse_args()

if args.transcription_method == "deepspeech" and args.deepspeech_model_dir is None:
    parser.error("DeepSpeech method requires --deepspeech_model_dir to be set to the directory containing the deepspeech models. See the documentation for details.")

if args.process_dir == "automatic":
    root_process_folder = Path(os.path.dirname(args.video_path))
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
    from cluster import ClusterFilesystem
    cluster_filesystem = ClusterFilesystem(slides_dir, algorithm_name="affinity_propagation", preference=-8, damping=0.72, max_iter=1000)
    cluster_filesystem.extract_and_add_features()
    cluster_dir, best_samples_dir= cluster_filesystem.transfer_to_filesystem()
    best_samples_dir = cluster_dir / "best_samples"
    # cluster_dir = make_clusters(slides_dir)

# 4. OCR slides
if args.skip_to <= 4: 
    if args.skip_to >= 4: # if step 3 (cluster slides) was skipped
        cluster_dir = root_process_folder / "slide_clusters"
        best_samples_dir = cluster_dir / "best_samples"
    import ocr
    save_file = root_process_folder / "ocr.txt"
    results = ocr.all_in_folder(best_samples_dir)
    ocr.write_to_file(results, save_file)

if args.skip_to <= 5:
    import transcribe
    extract_from_video = args.video_path
    audio_path = root_process_folder / "audio.wav"
    transcript_output_file = root_process_folder / "audio.txt"

    if args.transcription_method == "youtube":
        yt_output_file = root_process_folder / "audio.srt"
        try:
            transcript_path = transcribe.get_youtube_transcript(args.video_id, yt_output_file)
            if args.yt_convert_to_str:
                transcript = transcribe.srt_to_string(transcript_path)
                transcribe.write_to_file(transcript, transcript_output_file)
        except:
            youtube_transcription_failed = True
            args.transcription_method = parser.get_default("transcription_method")
            print("> Main Process: Error detected in grabbing transcript from YouTube. Falling back to " + parser.get_default("transcription_method") + " transcription.")
    if args.transcription_method != "youtube" or youtube_transcription_failed:
        transcribe.extract_audio(extract_from_video, audio_path)
        try:
            if args.chunk:
                chunk_dir = root_process_folder / "chunks"
                transcribe.create_chunks(audio_path, chunk_dir, 5, 2000)
                if args.transcription_method == "deepspeech":
                    transcribe.process_chunks(chunk_dir, transcript_output_file, model_dir=args.deepspeech_model_dir, method=args.transcription_method)
                else:
                    transcribe.process_chunks(chunk_dir, transcript_output_file, method=args.transcription_method)
            else:
                if args.transcription_method == "deepspeech":
                    transcript = transcribe.transcribe_audio_deepspeech(audio_path, args.deepspeech_model_dir)
                else:
                    transcript = transcribe.transcribe_audio(audio_path, method=args.transcription_method)
                transcribe.write_to_file(transcript, transcript_output_file)
        except:
            print("Audio transcription failed. Retry by running this script with the skip_to parameter set to 5.")

# if args.remove:
#     rmtree(root_process_folder)