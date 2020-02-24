# Main process to convert video to notes (end-to-end)
# 1. Extract frames
# 2. Classify slides
# 3. OCR slides

import sys
import os
import argparse
from shutil import rmtree
from pathlib import Path
from helpers import *


# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from custom_nnmodules import * #pylint: disable=import-error,wildcard-import,wrong-import-position
sys.path.insert(1, os.path.join(sys.path[0], '../Models/summarizer'))

PARSER = argparse.ArgumentParser(description='End-to-End Conversion of Lecture Videos to Notes using ML')
PARSER.add_argument('video_path', metavar='DIR',
                    help='path to video')
PARSER.add_argument('-s', '--skip-to', default=0, type=int, metavar='N',
                    help='set to > 0 to skip specific processing steps')
PARSER.add_argument('-d', '--process_dir', default='./', type=str, metavar='PATH',
                    help='path to the proessing directory (where extracted frames and other files are saved), set to "automatic" to use the video\'s folder (default: ./)')
PARSER.add_argument('-id', '--auto-id', dest='auto_id', action='store_true',
                    help='automatically create a subdirectory in `process_dir` with a unique id for the video and change `process_dir` to this new directory')
PARSER.add_argument('-rm', '--remove', dest='remove', action='store_true',
                    help='remove `process_dir` once conversion is complete')
PARSER.add_argument('-c', '--chunk', dest='chunk', action='store_true',
                    help='split the audio into small chunks on silence')
PARSER.add_argument('-tm', '--transcription_method', default="deepspeech", choices=["sphinx", "google", "youtube", "deepspeech"],
                    help='''specify the program that should be used for transcription. 
                    CMU Sphinx: use pocketsphinx (works offline)
                    Google Speech Recognition: probably will require chunking
                    YouTube: pull a video transcript from YouTube based on video_id
                    DeepSpeech: Use the deepspeech library (works offline with great accuracy)''')
PARSER.add_argument('--video_id', type=str, metavar='ID',
                    help='id of youtube video to get subtitles from')
PARSER.add_argument('--yt_convert_to_str', action='store_true',
                    help='if the method is `youtube` and this option is specified then the transcript will be saved as a txt file instead of a srt file.')
PARSER.add_argument('--deepspeech_model_dir', type=str, metavar='DIR',
                    help='path containing the DeepSpeech model files. See the documentation for details.')

ARGS = PARSER.parse_args()

if ARGS.transcription_method == "deepspeech" and ARGS.deepspeech_model_dir is None:
    PARSER.error("DeepSpeech method requires --deepspeech_model_dir to be set to the directory containing the deepspeech models. See the documentation for details.")

if ARGS.process_dir == "automatic":
    ROOT_PROCESS_FOLDER = Path(os.path.dirname(ARGS.video_path))
else:
    ROOT_PROCESS_FOLDER = Path(ARGS.process_dir)
if ARGS.auto_id:
    UNIQUE_ID = gen_unique_id(ARGS.video_path, 12)
    ROOT_PROCESS_FOLDER = ROOT_PROCESS_FOLDER / UNIQUE_ID

# 1. Extract frames
if ARGS.skip_to <= 1: 
    from frames_extractor import extract_frames
    QUALITY = 5
    OUTPUT_PATH = ROOT_PROCESS_FOLDER / "frames"
    EXTRACT_EVERY_X_SECONDS = 1
    extract_frames(ARGS.video_path, QUALITY, OUTPUT_PATH, EXTRACT_EVERY_X_SECONDS)

# 2. Classify slides
if ARGS.skip_to <= 2:
    from slide_classifier import classify_frames
    FRAMES_DIR = ROOT_PROCESS_FOLDER / "frames"
    FRAMES_SORTED_DIR, _, _ = classify_frames(FRAMES_DIR)

# 3. Cluster slides
if ARGS.skip_to <= 3: 
    if ARGS.skip_to >= 3: # if step 2 (classify slides) was skipped
        FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
    SLIDES_DIR = FRAMES_SORTED_DIR / "slide"
    from cluster import ClusterFilesystem
    CLUSTER_FILESYSTEM = ClusterFilesystem(SLIDES_DIR, algorithm_name="affinity_propagation", preference=-8, damping=0.72, max_iter=1000)
    CLUSTER_FILESYSTEM.extract_and_add_features()
    CLUSTER_DIR, BEST_SAMPLES_DIR = CLUSTER_FILESYSTEM.transfer_to_filesystem()
    BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
    # cluster_dir = make_clusters(slides_dir)

# 4. OCR slides
if ARGS.skip_to <= 4: 
    if ARGS.skip_to >= 4: # if step 3 (cluster slides) was skipped
        CLUSTER_DIR = ROOT_PROCESS_FOLDER / "slide_clusters"
        BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
    import ocr
    save_file = ROOT_PROCESS_FOLDER / "ocr.txt"
    results = ocr.all_in_folder(BEST_SAMPLES_DIR)
    ocr.write_to_file(results, save_file)

# 5. Transcribe Audio
if ARGS.skip_to <= 5:
    import transcribe
    EXTRACT_FROM_VIDEO = ARGS.video_path
    AUDIO_PATH = ROOT_PROCESS_FOLDER / "audio.wav"
    transcript_output_file = ROOT_PROCESS_FOLDER / "audio.txt"

    if ARGS.transcription_method == "youtube":
        yt_output_file = ROOT_PROCESS_FOLDER / "audio.srt"
        try:
            transcript_path = transcribe.get_youtube_transcript(ARGS.video_id, yt_output_file)
            if ARGS.yt_convert_to_str:
                transcript = transcribe.caption_file_to_string(transcript_path)
                transcribe.write_to_file(transcript, transcript_output_file)
        except:
            youtube_transcription_failed = True
            ARGS.transcription_method = PARSER.get_default("transcription_method")
            print("> Main Process: Error detected in grabbing transcript from YouTube. Falling back to " + PARSER.get_default("transcription_method") + " transcription.")
    if ARGS.transcription_method != "youtube" or youtube_transcription_failed:
        transcribe.extract_audio(EXTRACT_FROM_VIDEO, AUDIO_PATH)
        try:
            if ARGS.chunk:
                chunk_dir = ROOT_PROCESS_FOLDER / "chunks"
                transcribe.create_chunks(AUDIO_PATH, chunk_dir, 5, 2000)
                if ARGS.transcription_method == "deepspeech":
                    transcribe.process_chunks(chunk_dir, transcript_output_file, model_dir=ARGS.deepspeech_model_dir, method=ARGS.transcription_method)
                else:
                    transcribe.process_chunks(chunk_dir, transcript_output_file, method=ARGS.transcription_method)
            else:
                if ARGS.transcription_method == "deepspeech":
                    transcript = transcribe.transcribe_audio_deepspeech(AUDIO_PATH, ARGS.deepspeech_model_dir)
                else:
                    transcript = transcribe.transcribe_audio(AUDIO_PATH, method=ARGS.transcription_method)
                transcribe.write_to_file(transcript, transcript_output_file)
        except:
            print("Audio transcription failed. Retry by running this script with the skip_to parameter set to 5.")

# 6. Summarize Transcript
if ARGS.skip_to <= 6:
    if ARGS.skip_to >= 6: # if step 5 transcription was skipped
        import transcribe
        transcript_output_file = ROOT_PROCESS_FOLDER / "audio.txt"
        transcript_file = open(transcript_output_file, "r")
        transcript = transcript_file.read()
        transcript_file.close()
    transcript_summarized_output_file = ROOT_PROCESS_FOLDER / "audio_summarized.txt"
    import bart_sum
    bart = bart_sum.load_bart()

    transcript_length = len(transcript.split())
    min_len = int(transcript_length/6)
    if min_len > 500:
        # If the length is too long the model will start to repeat
        min_len = 500
    transcript_summarized = bart_sum.summarize(bart, transcript, min_len=min_len, max_len_b=min_len+200)
    transcribe.write_to_file(transcript_summarized, transcript_summarized_output_file)

if ARGS.remove:
    rmtree(ROOT_PROCESS_FOLDER)
