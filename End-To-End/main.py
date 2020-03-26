# Main process to convert video to notes (end-to-end)

import sys
import os
import argparse
import logging
import spacy
from shutil import rmtree
from pathlib import Path
from helpers import *
from summarization_approaches import (full_sents,
                                      keyword_based_ext,
                                      get_complete_sentences,
                                      generic_abstractive,
                                      cluster,
                                      generic_extractive_sumy)

logger = logging.getLogger(__name__)

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from custom_nnmodules import * #pylint: disable=import-error,wildcard-import,wrong-import-position

def main(ARGS):
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

    # 3. Perspective crop images of presenter_slide to contain only the slide (helps OCR)
    if ARGS.skip_to <= 3:
        if ARGS.skip_to >= 3: # if step 2 (classify slides) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
        PRESENTER_SLIDE_DIR = FRAMES_SORTED_DIR / "presenter_slide"
        IMGS_TO_CLUSTER_DIR = FRAMES_SORTED_DIR / "imgs_to_cluster"
        if os.path.exists(PRESENTER_SLIDE_DIR):
            import corner_crop_transform
            cropped_imgs_paths = corner_crop_transform.all_in_folder(PRESENTER_SLIDE_DIR, remove_original=False)
            copy_all(cropped_imgs_paths, IMGS_TO_CLUSTER_DIR)

    # 4. Cluster slides
    if ARGS.skip_to <= 4:
        if ARGS.skip_to >= 4: # if step 3 (perspective crop) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
            IMGS_TO_CLUSTER_DIR = FRAMES_SORTED_DIR / "imgs_to_cluster"
        SLIDES_DIR = FRAMES_SORTED_DIR / "slide"

        copy_all(SLIDES_DIR, IMGS_TO_CLUSTER_DIR)

        from cluster import ClusterFilesystem
        CLUSTER_FILESYSTEM = ClusterFilesystem(IMGS_TO_CLUSTER_DIR, algorithm_name="affinity_propagation", preference=-8, damping=0.72, max_iter=1000)
        CLUSTER_FILESYSTEM.extract_and_add_features()
        if ARGS.tensorboard:
            CLUSTER_FILESYSTEM.visualize(ARGS.tensorboard)
        CLUSTER_DIR, BEST_SAMPLES_DIR = CLUSTER_FILESYSTEM.transfer_to_filesystem()
        BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
        # cluster_dir = make_clusters(slides_dir)

    # 5. OCR slides
    if ARGS.skip_to <= 5: 
        if ARGS.skip_to >= 5: # if step 4 (perspective crop) was skipped
            CLUSTER_DIR = ROOT_PROCESS_FOLDER / "slide_clusters"
            BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
        import ocr
        OCR_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "ocr.txt"
        OCR_RESULTS = ocr.all_in_folder(BEST_SAMPLES_DIR)
        ocr.write_to_file(OCR_RESULTS, OCR_OUTPUT_FILE)

    # 6. Transcribe Audio
    if ARGS.skip_to <= 6:
        import transcribe
        EXTRACT_FROM_VIDEO = ARGS.video_path
        AUDIO_PATH = ROOT_PROCESS_FOLDER / "audio.wav"
        TRANSCRIPT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.txt"

        YT_TRANSCRIPTION_FAILED = False

        if ARGS.transcription_method == "youtube":
            YT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.vtt"
            try:
                TRANSCRIPT_PATH = transcribe.get_youtube_transcript(ARGS.video_id, YT_OUTPUT_FILE)
                TRANSCRIPT = transcribe.caption_file_to_string(TRANSCRIPT_PATH)
                transcribe.write_to_file(TRANSCRIPT, TRANSCRIPT_OUTPUT_FILE)
            except:
                YT_TRANSCRIPTION_FAILED = True
                ARGS.transcription_method = PARSER.get_default("transcription_method")
                logger.error("Error detected in grabbing transcript from YouTube. Falling back to " + PARSER.get_default("transcription_method") + " transcription.")
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
                logger.error("Audio transcription failed. Retry by running this script with the skip_to parameter set to 5.")

    # 7. Summarization
    if ARGS.skip_to <= 7:
        if ARGS.skip_to >= 6: # if step 6 transcription or step 5 ocr was skipped
            OCR_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "ocr.txt"
            OCR_FILE = open(OCR_OUTPUT_FILE, "r")
            OCR_RESULTS_FLAT = OCR_FILE.read()
            OCR_FILE.close()

            import transcribe
            TRANSCRIPT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.txt"
            TRANSCRIPT_FILE = open(TRANSCRIPT_OUTPUT_FILE, "r")
            TRANSCRIPT = TRANSCRIPT_FILE.read()
            TRANSCRIPT_FILE.close()
        else:
            OCR_RESULTS_FLAT = " ".join(OCR_RESULTS) # converts list of strings into one string where each item is separated by a space
        LECTURE_SUMMARIZED_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "summarized.txt"

        OCR_RESULTS_FLAT = OCR_RESULTS_FLAT.replace('\n', ' ').replace('\r', '') # remove line breaks

        # Combination Algorithm
        if ARGS.combination_algo == "only_asr":
            SUMMARIZED_COMBINED = TRANSCRIPT
        elif ARGS.combination_algo == "only_slides":
            SUMMARIZED_COMBINED = OCR_RESULTS_FLAT
        elif ARGS.combination_algo == "concat":
            SUMMARIZED_COMBINED = OCR_RESULTS_FLAT + TRANSCRIPT
        elif ARGS.combination_algo == "full_sents":
            SUMMARIZED_COMBINED = full_sents(OCR_RESULTS_FLAT, TRANSCRIPT)
        elif ARGS.combination_algo == "keyword_based":
            SUMMARIZED_COMBINED = keyword_based_ext(OCR_RESULTS_FLAT, TRANSCRIPT)
        else: # if no combination algorithm was specified, which should never happen since argparse checks
            logger.warn("No combination algorithm selected. Defaulting to `concat`.")
            SUMMARIZED_COMBINED = OCR_RESULTS_FLAT + TRANSCRIPT

        # Modifications
        if "full_sents" in ARGS.summarization_mods:
            SUMMARIZED_MOD = get_complete_sentences(SUMMARIZED_COMBINED, return_string=True)
        else:
            SUMMARIZED_MOD = SUMMARIZED_COMBINED
            logger.debug("Skipping summarization_mods")
        
        # Extractive Summarization
        if ARGS.summarization_ext != "none" and ARGS.summarization_ext is not None: # if extractive method was specified
            if "cluster" in ARGS.summarization_ext:
                SUMMARIZED_EXT = cluster(SUMMARIZED_MOD, title_generation=True, cluster_summarizer="abstractive")
            else: # one of the generic options was specified
                SUMMARIZED_EXT = generic_extractive_sumy(SUMMARIZED_MOD, algorithm=ARGS.summarization_ext)
        else:
            logger.debug("Skipping summarization_ext")

        # Abstractive Summarization
        if ARGS.summarization_abs != "none" and ARGS.summarization_abs is not None: # if abstractive method was specified
            LECTURE_SUMMARIZED = generic_abstractive(SUMMARIZED_EXT, ARGS.summarization_abs)
        else: # if no abstractive summarization method was specified
            LECTURE_SUMMARIZED = SUMMARIZED_EXT
            logger.debug("Skipping summarization_abs")

        transcribe.write_to_file(LECTURE_SUMMARIZED, LECTURE_SUMMARIZED_OUTPUT_FILE)

    if ARGS.remove:
        rmtree(ROOT_PROCESS_FOLDER)

if __name__ == "__main__":
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
    PARSER.add_argument('-ca', '--combination_algo', default="keyword_based", choices=["only_asr", "concat", "full_sents", "keyword_based"],
                        help='which extractive summarization approach to use. more information in documentation.')
    PARSER.add_argument('-sm', '--summarization_mods', default=None, choices=["none", "full_sents"], nargs="+",
                        help='modifications to perform during summarization process. each modification is run between the combination and extractive stages. more information in documentation.')
    PARSER.add_argument('-sx', '--summarization_ext', default="text_rank", choices=["none", "cluster", "lsa", "luhn", "lex_rank", "text_rank", "edmundson", "random"],
                        help='which extractive summarization approach to use. more information in documentation.')
    PARSER.add_argument('-sa', '--summarization_abs', default="bart", choices=["none", "bart", "presumm"],
                        help='which abstractive summarization approach/model to use. more information in documentation.')
    PARSER.add_argument('-tm', '--transcription_method', default="deepspeech", choices=["sphinx", "google", "youtube", "deepspeech"],
                        help='''specify the program that should be used for transcription. 
                        CMU Sphinx: use pocketsphinx (works offline)
                        Google Speech Recognition: probably will require chunking
                        YouTube: pull a video transcript from YouTube based on video_id
                        DeepSpeech: Use the deepspeech library (works offline with great accuracy)''')
    PARSER.add_argument('--video_id', type=str, metavar='ID',
                        help='id of youtube video to get subtitles from')
    PARSER.add_argument('--deepspeech_model_dir', type=str, metavar='DIR',
                        help='path containing the DeepSpeech model files. See the documentation for details.')
    PARSER.add_argument('--tensorboard', default='', type=str, metavar='PATH',
                        help='Path to tensorboard logdir. Tensorboard not used if not set. Tensorboard only used to visualize cluster primarily for debugging.')
    PARSER.add_argument('--bart_checkpoint', default=None, type=str, metavar='PATH',
                        help='[BART Abstractive Summarizer Only] Path to optional checkpoint. Semsim is better model but will use more memory and is an additional 5GB download. (default: none, recommended: semsim)')
    PARSER.add_argument('--bart_state_dict_key', default='model', type=str, metavar='PATH',
                        help='[BART Abstractive Summarizer Only] model state_dict key to load from pickle file specified with --bart_checkpoint (default: "model")')
    PARSER.add_argument('--bart_fairseq', action='store_true',
                        help='[BART Abstractive Summarizer Only] Use fairseq model from torch hub instead of huggingface transformers library models. Can not use --bart_checkpoint if this option is supplied.')
    PARSER.add_argument("-l", "--log", dest="logLevel", default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level (default: 'Info').")

    ARGS = PARSER.parse_args()

    # Perform argument checks
    if ARGS.transcription_method == "deepspeech" and ARGS.deepspeech_model_dir is None:
        PARSER.error("DeepSpeech method requires --deepspeech_model_dir to be set to the directory containing the deepspeech models. See the documentation for details.")
    
    if (ARGS.summarization_mods is not None) and ("none" in ARGS.summarization_mods and len(ARGS.summarization_mods) > 1): # None and another option were specified
        PARSER.error("If 'none' is specified in --summarization_mods then no other options can be selected.")

    # Setup logging config
    logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s> %(message)s", level=logging.getLevelName(ARGS.logLevel))

    main(ARGS)