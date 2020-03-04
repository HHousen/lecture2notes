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
PARSER.add_argument('-sm', '--sum_model', default="bart", choices=["bart", "presumm"],
                    help='summarization model')
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

# 4. Perspective crop images of slides to contain only the slide (helps OCR)
if ARGS.skip_to <= 4:
    if ARGS.skip_to >= 4: # if step 3 (cluster slides) was skipped
        FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
        CLUSTER_DIR = FRAMES_SORTED_DIR / "slide_clusters"
        BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
    import corner_crop_transform
    cropped_imgs_paths = corner_crop_transform.all_in_folder(BEST_SAMPLES_DIR, remove_original=False)
input("end")
# 5. OCR slides
if ARGS.skip_to <= 5: 
    if ARGS.skip_to >= 5: # if step 4 (perspective crop) was skipped
        CLUSTER_DIR = ROOT_PROCESS_FOLDER / "slide_clusters"
        BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
    import ocr
    save_file = ROOT_PROCESS_FOLDER / "ocr.txt"
    results = ocr.all_in_folder(BEST_SAMPLES_DIR)
    ocr.write_to_file(results, save_file)

# 6. Transcribe Audio
if ARGS.skip_to <= 6:
    import transcribe
    EXTRACT_FROM_VIDEO = ARGS.video_path
    AUDIO_PATH = ROOT_PROCESS_FOLDER / "audio.wav"
    TRANSCRIPT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.txt"

    if ARGS.transcription_method == "youtube":
        YT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.srt"
        try:
            TRANSCRIPT_PATH = transcribe.get_youtube_transcript(ARGS.video_id, YT_OUTPUT_FILE)
            if ARGS.yt_convert_to_str:
                TRANSCRIPT = transcribe.caption_file_to_string(TRANSCRIPT_PATH)
                transcribe.write_to_file(TRANSCRIPT, TRANSCRIPT_OUTPUT_FILE)
        except:
            YT_TRANSCRIPTION_FAILED = True
            ARGS.transcription_method = PARSER.get_default("transcription_method")
            print("> Main Process: Error detected in grabbing transcript from YouTube. Falling back to " + PARSER.get_default("transcription_method") + " transcription.")
    if ARGS.transcription_method != "youtube" or YT_TRANSCRIPTION_FAILED:
        transcribe.extract_audio(EXTRACT_FROM_VIDEO, AUDIO_PATH)
        try:
            if ARGS.chunk:
                CHUNK_DIR = ROOT_PROCESS_FOLDER / "chunks"
                transcribe.create_chunks(AUDIO_PATH, CHUNK_DIR, 5, 2000)
                if ARGS.transcription_method == "deepspeech":
                    transcribe.process_chunks(CHUNK_DIR, TRANSCRIPT_OUTPUT_FILE, model_dir=ARGS.deepspeech_model_dir, method=ARGS.transcription_method)
                else:
                    transcribe.process_chunks(CHUNK_DIR, TRANSCRIPT_OUTPUT_FILE, method=ARGS.transcription_method)
            else:
                if ARGS.transcription_method == "deepspeech":
                    TRANSCRIPT = transcribe.transcribe_audio_deepspeech(AUDIO_PATH, ARGS.deepspeech_model_dir)
                else:
                    TRANSCRIPT = transcribe.transcribe_audio(AUDIO_PATH, method=ARGS.transcription_method)
                transcribe.write_to_file(TRANSCRIPT, TRANSCRIPT_OUTPUT_FILE)
        except:
            print("Audio transcription failed. Retry by running this script with the skip_to parameter set to 5.")

# 7. Summarize Transcript
if ARGS.skip_to <= 7:
    if ARGS.skip_to >= 7: # if step 6 transcription was skipped
        import transcribe
        TRANSCRIPT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.txt"
        TRANSCRIPT_FILE = open(TRANSCRIPT_OUTPUT_FILE, "r")
        TRANSCRIPT = TRANSCRIPT_FILE.read()
        TRANSCRIPT_FILE.close()
    TRANSCRIPT_SUMMARIZED_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio_summarized.txt"
    if ARGS.sum_model == "bart":
        import bart_sum
        SUMMARIZER = bart_sum.BartSumSummarizer()
    elif ARGS.sum_model == "presumm":
        import presumm.presumm as presumm
        SUMMARIZER = presumm.PreSummSummarizer()

    TRANSCRIPT_LENGTH = len(TRANSCRIPT.split())
    MIN_LEN = int(TRANSCRIPT_LENGTH/6)
    if MIN_LEN > 500:
        # If the length is too long the model will start to repeat
        MIN_LEN = 500
    TRANSCRIPT_SUMMARIZED = SUMMARIZER.summarize_string(TRANSCRIPT, min_len=MIN_LEN, max_len_b=MIN_LEN+200)
    transcribe.write_to_file(TRANSCRIPT_SUMMARIZED, TRANSCRIPT_SUMMARIZED_OUTPUT_FILE)

if ARGS.remove:
    rmtree(ROOT_PROCESS_FOLDER)
