# Main process to convert video to notes (end-to-end)

import sys
import os
import json
import argparse
import logging
import spacy
from shutil import rmtree
from pathlib import Path
from helpers import *
from timeit import default_timer as timer
from spell_check import SpellChecker
from summarization_approaches import (
    full_sents,
    keyword_based_ext,
    get_complete_sentences,
    generic_abstractive,
    cluster,
    generic_extractive_sumy,
    structured_joined_sum,
)

logger = logging.getLogger(__name__)

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(os.getcwd(), "../Models/slide-classifier"))
from custom_nnmodules import *  # pylint: disable=import-error,wildcard-import,wrong-import-position


def main(ARGS):
    if ARGS.process_dir == "automatic":
        ROOT_PROCESS_FOLDER = Path(os.path.dirname(ARGS.video_path))
    else:
        ROOT_PROCESS_FOLDER = Path(ARGS.process_dir)
    if ARGS.auto_id:
        UNIQUE_ID = gen_unique_id(ARGS.video_path, 12)
        ROOT_PROCESS_FOLDER = ROOT_PROCESS_FOLDER / UNIQUE_ID

    if ARGS.spell_check:
        spell_checker = SpellChecker()

    # 1. Extract frames
    if ARGS.skip_to <= 1:
        from frames_extractor import extract_frames

        start_time = timer()

        QUALITY = 5
        OUTPUT_PATH = ROOT_PROCESS_FOLDER / "frames"
        EXTRACT_EVERY_X_SECONDS = 1
        extract_frames(ARGS.video_path, QUALITY, OUTPUT_PATH, EXTRACT_EVERY_X_SECONDS)

        end_time = timer() - start_time
        logger.info("Stage 1 (Extract Frames) took %s", end_time)

    # 2. Classify slides
    if ARGS.skip_to <= 2:
        from slide_classifier import classify_frames

        start_time = timer()

        FRAMES_DIR = ROOT_PROCESS_FOLDER / "frames"
        FRAMES_SORTED_DIR, _, _ = classify_frames(FRAMES_DIR)

        end_time = timer() - start_time
        logger.info("Stage 2 (Classify Slides) took %s", end_time)

    # 3. Black border detection and removal
    if ARGS.skip_to <= 3:
        start_time = timer()
        if ARGS.skip_to >= 3:  # if step 2 (Classify Slides) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"

        SLIDES_DIR = FRAMES_SORTED_DIR / "slide"
        SLIDES_NOBORDER_DIR = FRAMES_SORTED_DIR / "slides_noborder"

        if os.path.exists(SLIDES_DIR):
            os.makedirs(SLIDES_NOBORDER_DIR, exist_ok=True)

            if ARGS.remove_duplicates:
                import imghash

                images_hashed = imghash.sort_by_duplicates(SLIDES_DIR)
                imghash.remove_duplicates(SLIDES_DIR, images_hashed)

            import border_removal

            REMOVED_BORDERS_PATHS = border_removal.all_in_folder(SLIDES_DIR)
            copy_all(REMOVED_BORDERS_PATHS, SLIDES_NOBORDER_DIR)

        end_time = timer() - start_time
        logger.info("Stage 3 (Border Removal) took %s", end_time)

    # 4. Perspective crop images of presenter_slide to contain only the slide (helps OCR)
    if ARGS.skip_to <= 4:
        if ARGS.skip_to >= 4:  # if step 3 (border removal) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
            SLIDES_NOBORDER_DIR = FRAMES_SORTED_DIR / "slides_noborder"
        PRESENTER_SLIDE_DIR = FRAMES_SORTED_DIR / "presenter_slide"
        IMGS_TO_CLUSTER_DIR = FRAMES_SORTED_DIR / "imgs_to_cluster"

        start_time = timer()

        if os.path.exists(PRESENTER_SLIDE_DIR):

            if ARGS.remove_duplicates:
                import imghash

                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Remove 'presenter_slide' duplicates"
                )
                imghash_start_time = timer()

                images_hashed = imghash.sort_by_duplicates(PRESENTER_SLIDE_DIR)
                imghash.remove_duplicates(PRESENTER_SLIDE_DIR, images_hashed)

                imghash_end_time = timer() - imghash_start_time
                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Remove 'presenter_slide' duplicates took %s",
                    imghash_end_time,
                )

            import sift_matcher

            logger.info("Stage 4 (Duplicate Removal & Perspective Crop): SIFT Matching")
            siftmatch_start_time = timer()

            (
                non_unique_presenter_slides,
                transformed_image_paths,
            ) = sift_matcher.match_features(SLIDES_NOBORDER_DIR, PRESENTER_SLIDE_DIR)
            siftmatch_end_time = timer() - siftmatch_start_time
            logger.info(
                "Stage 4 (Duplicate Removal & Perspective Crop): SIFT Matching took %s",
                siftmatch_end_time,
            )

            # Remove all 'presenter_slide' images that are duplicates of 'slide' images
            # and all 'slide' images that are better represented by a 'presenter_slide' image
            for x in non_unique_presenter_slides:
                try:
                    os.remove(x)
                except OSError:
                    pass

            # If there are transformed images then the camera motion was steady and we
            # do not have to run `corner_crop_transform`. If camera motion was detected
            # then the `transformed_image_paths` list will be empty and `PRESENTER_SLIDE_DIR`
            # will contain potentially unique 'presenter_slide' images that do not appear
            # in any images of the 'slide' class.
            if transformed_image_paths:
                copy_all(transformed_image_paths, IMGS_TO_CLUSTER_DIR)
            else:
                import corner_crop_transform

                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Corner Crop Transform"
                )
                cornercrop_start_time = timer()

                cropped_imgs_paths = corner_crop_transform.all_in_folder(
                    PRESENTER_SLIDE_DIR, remove_original=False
                )
                copy_all(cropped_imgs_paths, IMGS_TO_CLUSTER_DIR)

                cornercrop_end_time = timer() - cornercrop_start_time
                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Corner Crop Transform took %s",
                    cornercrop_end_time,
                )

        end_time = timer() - start_time
        logger.info("Stage 4 (Perspective Crop) took %s", end_time)

    # 5. Cluster slides
    if ARGS.skip_to <= 5:
        start_time = timer()
        if ARGS.skip_to >= 5:  # if step 4 (perspective crop) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
            IMGS_TO_CLUSTER_DIR = FRAMES_SORTED_DIR / "imgs_to_cluster"
            SLIDES_NOBORDER_DIR = FRAMES_SORTED_DIR / "slides_noborder"

        copy_all(SLIDES_NOBORDER_DIR, IMGS_TO_CLUSTER_DIR)

        if ARGS.remove_duplicates:
            import imghash

            images_hashed = imghash.sort_by_duplicates(IMGS_TO_CLUSTER_DIR)
            imghash.remove_duplicates(IMGS_TO_CLUSTER_DIR, images_hashed)

        if ARGS.cluster_method == "normal":
            from cluster import ClusterFilesystem

            CLUSTER_FILESYSTEM = ClusterFilesystem(
                IMGS_TO_CLUSTER_DIR,
                algorithm_name="affinity_propagation",
                preference=-8,
                damping=0.72,
                max_iter=1000,
            )
            CLUSTER_FILESYSTEM.extract_and_add_features()
            if ARGS.tensorboard:
                CLUSTER_FILESYSTEM.visualize(ARGS.tensorboard)
            CLUSTER_DIR, BEST_SAMPLES_DIR = CLUSTER_FILESYSTEM.transfer_to_filesystem()
            # BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
            # cluster_dir = make_clusters(slides_dir)
        elif ARGS.cluster_method == "segment":
            from segment_cluster import SegmentCluster

            SEGMENT_CLUSTER = SegmentCluster(IMGS_TO_CLUSTER_DIR)
            SEGMENT_CLUSTER.extract_and_add_features()
            CLUSTER_DIR, BEST_SAMPLES_DIR = SEGMENT_CLUSTER.transfer_to_filesystem()

        end_time = timer() - start_time
        logger.info("Stage 5 (Cluster Slides) took %s", end_time)

    # 6. Slide Structure Analysis (SSA) and OCR Slides
    if ARGS.skip_to <= 6:
        start_time = timer()
        if ARGS.skip_to >= 6:  # if step 5 (cluster slides) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
            CLUSTER_DIR = FRAMES_SORTED_DIR / "slide_clusters"
            BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
        import slide_structure_analysis

        OCR_RAW_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "slide-ocr.txt"
        OCR_JSON_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "slide-ssa.json"
        OCR_RAW_TEXT, OCR_JSON_DATA = slide_structure_analysis.all_in_folder(
            BEST_SAMPLES_DIR
        )
        if "ocr" in ARGS.spell_check:
            OCR_RAW_TEXT = spell_checker.check_all(OCR_RAW_TEXT)
        slide_structure_analysis.write_to_file(
            OCR_RAW_TEXT, OCR_JSON_DATA, OCR_RAW_OUTPUT_FILE, OCR_JSON_OUTPUT_FILE
        )

        end_time = timer() - start_time
        logger.info("Stage 6 (SSA and OCR Slides) took %s", end_time)

    # 7. Extract figures
    if ARGS.skip_to <= 7:
        start_time = timer()
        if ARGS.skip_to >= 7:  # if step 6 (ssa and ocr) was skipped
            FRAMES_SORTED_DIR = ROOT_PROCESS_FOLDER / "frames_sorted"
            CLUSTER_DIR = FRAMES_SORTED_DIR / "slide_clusters"
            BEST_SAMPLES_DIR = CLUSTER_DIR / "best_samples"
            OCR_JSON_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "slide-ssa.json"

        FIGURES_DIR = CLUSTER_DIR / "best_samples_figures"
        os.makedirs(FIGURES_DIR, exist_ok=True)

        import figure_detection

        FIGURE_PATHS = figure_detection.all_in_folder(BEST_SAMPLES_DIR)
        copy_all(FIGURE_PATHS, FIGURES_DIR, move=True)

        if os.path.isfile(OCR_JSON_OUTPUT_FILE):
            with open(OCR_JSON_OUTPUT_FILE, "r") as ssa_json_file:
                ssa = json.load(ssa_json_file)
                ssa = figure_detection.add_figures_to_ssa(ssa, FIGURES_DIR)
            with open(OCR_JSON_OUTPUT_FILE, "w") as ssa_json_file:
                json.dump(ssa, ssa_json_file)

        end_time = timer() - start_time
        logger.info("Stage 7 (Extract Figures) took %s", end_time)

    # 8. Transcribe Audio
    if ARGS.skip_to <= 8:
        from transcribe import transcribe_main as transcribe

        start_time = timer()

        EXTRACT_FROM_VIDEO = ARGS.video_path
        AUDIO_PATH = ROOT_PROCESS_FOLDER / "audio.wav"
        TRANSCRIPT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.txt"
        TRANSCRIPT_JSON_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.json"
        TRANSCRIPT_JSON = None

        YT_TRANSCRIPTION_FAILED = False

        if ARGS.transcription_method == "youtube":
            YT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.vtt"
            try:
                TRANSCRIPT_PATH = transcribe.get_youtube_transcript(
                    ARGS.video_id, YT_OUTPUT_FILE
                )
                TRANSCRIPT = transcribe.caption_file_to_string(TRANSCRIPT_PATH)
            except:
                YT_TRANSCRIPTION_FAILED = True
                ARGS.transcription_method = PARSER.get_default("transcription_method")
                logger.error(
                    "Error detected in grabbing transcript from YouTube. Falling back to "
                    + PARSER.get_default("transcription_method")
                    + " transcription."
                )

        if ARGS.transcription_method != "youtube" or YT_TRANSCRIPTION_FAILED:
            transcribe.extract_audio(EXTRACT_FROM_VIDEO, AUDIO_PATH)
            try:
                if ARGS.chunk == "silence":
                    CHUNK_DIR = ROOT_PROCESS_FOLDER / "chunks"
                    transcribe.chunk_by_silence(AUDIO_PATH, CHUNK_DIR)
                    TRANSCRIPT, TRANSCRIPT_JSON = transcribe.process_chunks(
                        CHUNK_DIR,
                        model_dir=ARGS.transcribe_model_dir,
                        method=ARGS.transcription_method,
                    )

                    if ARGS.transcribe_segment_sentences:
                        TRANSCRIPT, TRANSCRIPT_JSON = transcribe.segment_sentences(
                            TRANSCRIPT, TRANSCRIPT_JSON
                        )

                elif ARGS.chunk == "speech":
                    stt_model = transcribe.load_model(
                        ARGS.transcription_method, ARGS.transcribe_model_dir
                    )
                    
                    # Only DeepSpeech has a `sampleRate()` method but `stt_model` could contain
                    # a DeepSpeech or Vosk model
                    try:
                        desired_sample_rate = stt_model.sampleRate()
                    except AttributeError:
                        # default sample rate to convert to is 16000
                        desired_sample_rate = 16000
                    
                    segments, _, audio_length = transcribe.chunk_by_speech(
                        AUDIO_PATH, desired_sample_rate=desired_sample_rate
                    )
                    TRANSCRIPT, TRANSCRIPT_JSON = transcribe.process_segments(
                        segments, stt_model, method=ARGS.transcription_method, audio_length=audio_length, do_segment_sentences=ARGS.transcribe_segment_sentences
                    )

                else:  # if not chunking
                    TRANSCRIPT, TRANSCRIPT_JSON = transcribe.transcribe_audio(
                        AUDIO_PATH, method=ARGS.transcription_method, model=ARGS.transcribe_model_dir
                    )

                    if ARGS.transcribe_segment_sentences:
                        TRANSCRIPT, TRANSCRIPT_JSON = transcribe.segment_sentences(
                            TRANSCRIPT, TRANSCRIPT_JSON
                        )
            
            except Exception as e:
                logger.error(
                    "Audio transcription failed. Retry by running this script with the skip_to parameter set to 6."
                )
                raise

        if "transcript" in ARGS.spell_check:
            TRANSCRIPT = spell_checker.check(TRANSCRIPT)

        transcribe.write_to_file(
            TRANSCRIPT,
            TRANSCRIPT_OUTPUT_FILE,
            TRANSCRIPT_JSON,
            TRANSCRIPT_JSON_OUTPUT_FILE,
        )

        end_time = timer() - start_time
        logger.info("Stage 8 (Transcribe Audio) took %s", end_time)

    # 9. Summarization
    if ARGS.skip_to <= 9:
        start_time = timer()

        if ARGS.skip_to >= 7:  # if step 8 transcription or step 6 ocr was skipped
            OCR_RAW_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "slide-ocr.txt"
            with open(OCR_RAW_OUTPUT_FILE, "r") as OCR_FILE:
                OCR_RESULTS_FLAT = OCR_FILE.read()
            OCR_JSON_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "slide-ssa.json"

            from transcribe import transcribe_main as transcribe

            TRANSCRIPT_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.txt"
            with open(TRANSCRIPT_OUTPUT_FILE, "r") as TRANSCRIPT_FILE:
                TRANSCRIPT = TRANSCRIPT_FILE.read()
            TRANSCRIPT_JSON_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "audio.json"

            EXTRACT_EVERY_X_SECONDS = 1
        else:
            OCR_RESULTS_FLAT = " ".join(
                OCR_RAW_TEXT
            )  # converts list of strings into one string where each item is separated by a space
        LECTURE_SUMMARIZED_OUTPUT_FILE = ROOT_PROCESS_FOLDER / "summarized.txt"
        LECTURE_SUMMARIZED_STRUCTURED_OUTPUT_FILE = (
            ROOT_PROCESS_FOLDER / "summarized.json"
        )

        OCR_RESULTS_FLAT = OCR_RESULTS_FLAT.replace("\n", " ").replace(
            "\r", ""
        )  # remove line breaks

        if (
            ARGS.summarization_structured != "none"
            and ARGS.summarization_structured is not None
        ):
            logger.info("Stage 9 (Summarization): Structured Summarization")
            ss_start_time = timer()

            if ARGS.summarization_structured == "structured_joined":
                structured_joined_sum(
                    OCR_JSON_OUTPUT_FILE,
                    TRANSCRIPT_JSON_OUTPUT_FILE,
                    frame_every_x=EXTRACT_EVERY_X_SECONDS,
                    ending_char=".",
                    to_json=LECTURE_SUMMARIZED_STRUCTURED_OUTPUT_FILE,
                )

            ss_end_time = timer() - ss_start_time
            logger.info("Stage 9 (Summarization): Structured took %s", ss_end_time)
        else:
            logger.info("Skipping structured summarization.")

        # Combination Algorithm
        logger.info("Stage 9 (Summarization): Combination Algorithm")
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
        else:  # if no combination algorithm was specified, which should never happen since argparse checks
            logger.warn("No combination algorithm selected. Defaulting to `concat`.")
            SUMMARIZED_COMBINED = OCR_RESULTS_FLAT + TRANSCRIPT

        # Modifications
        logger.info("Stage 9 (Summarization): Modifications")
        if ARGS.summarization_mods != "none" and ARGS.summarization_mods is not None:
            if "full_sents" in ARGS.summarization_mods:
                SUMMARIZED_MOD = get_complete_sentences(
                    SUMMARIZED_COMBINED, return_string=True
                )
        else:
            SUMMARIZED_MOD = SUMMARIZED_COMBINED
            logger.debug("Skipping summarization_mods")

        # Extractive Summarization
        logger.info("Stage 9 (Summarization): Extractive")
        ext_start_time = timer()
        if (
            ARGS.summarization_ext != "none" and ARGS.summarization_ext is not None
        ):  # if extractive method was specified
            if ARGS.summarization_ext == "cluster":
                SUMMARIZED_EXT = cluster(
                    SUMMARIZED_MOD,
                    title_generation=False,
                    cluster_summarizer="abstractive",
                )
            else:  # one of the generic options was specified
                SUMMARIZED_EXT = generic_extractive_sumy(
                    SUMMARIZED_MOD, algorithm=ARGS.summarization_ext
                )
        else:
            logger.debug("Skipping summarization_ext")
        ext_end_time = timer() - ext_start_time
        logger.info("Stage 9 (Summarization): Extractive took %s", ext_end_time)

        # Abstractive Summarization
        logger.info("Stage 9 (Summarization): Abstractive")
        abs_start_time = timer()
        if (
            ARGS.summarization_abs != "none" and ARGS.summarization_abs is not None
        ):  # if abstractive method was specified
            LECTURE_SUMMARIZED = generic_abstractive(
                SUMMARIZED_EXT, ARGS.summarization_abs
            )
        else:  # if no abstractive summarization method was specified
            LECTURE_SUMMARIZED = SUMMARIZED_EXT
            logger.debug("Skipping summarization_abs")
        abs_end_time = timer() - abs_start_time
        logger.info("Stage 9 (Summarization): Abstractive took %s", abs_end_time)

        transcribe.write_to_file(LECTURE_SUMMARIZED, LECTURE_SUMMARIZED_OUTPUT_FILE)

        end_time = timer() - start_time
        logger.info("Stage 9 (Summarization) took %s", end_time)

    if ARGS.remove:
        rmtree(ROOT_PROCESS_FOLDER)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="End-to-End Conversion of Lecture Videos to Notes using ML"
    )
    PARSER.add_argument("video_path", metavar="DIR", help="path to video")
    PARSER.add_argument(
        "-s",
        "--skip_to",
        default=0,
        type=int,
        metavar="N",
        help="set to > 0 to skip specific processing steps",
    )
    PARSER.add_argument(
        "-d",
        "--process_dir",
        default="./",
        type=str,
        metavar="PATH",
        help='path to the proessing directory (where extracted frames and other files are saved), set to "automatic" to use the video\'s folder (default: ./)',
    )
    PARSER.add_argument(
        "-id",
        "--auto_id",
        action="store_true",
        help="automatically create a subdirectory in `process_dir` with a unique id for the video and change `process_dir` to this new directory",
    )
    PARSER.add_argument(
        "-rm",
        "--remove",
        action="store_true",
        help="remove `process_dir` once conversion is complete",
    )
    PARSER.add_argument(
        "-c",
        "--chunk",
        default="none",
        choices=["silence", "speech", "none"],
        help="split the audio into small chunks on `silence` using PyDub or voice activity `speech` using py-webrtcvad. set to 'none' to disable. Recommend 'speech' for DeepSpeech and 'none' for Vosk. (default: 'none').",
    )
    PARSER.add_argument(
        "-rd",
        "--remove_duplicates",
        action="store_true",
        help="remove duplicate slides before perspective cropping and before clustering (helpful when `--cluster_method` is `segment`)",
    )
    PARSER.add_argument(
        "-cm",
        "--cluster_method",
        default="segment",
        choices=["normal", "segment"],
        help="which clustering method to use. `normal` uses a clustering algorithm from scikit-learn and `segment` uses the special method that iterates through frames in order and splits based on large visual differences",
    )
    PARSER.add_argument(
        "-ca",
        "--combination_algo",
        default="keyword_based",
        choices=["only_asr", "concat", "full_sents", "keyword_based"],
        help="which extractive summarization approach to use. more information in documentation.",
    )
    PARSER.add_argument(
        "-sm",
        "--summarization_mods",
        default=None,
        choices=["none", "full_sents"],
        nargs="+",
        help="modifications to perform during summarization process. each modification is run between the combination and extractive stages. more information in documentation.",
    )
    PARSER.add_argument(
        "-sx",
        "--summarization_ext",
        default="text_rank",
        choices=[
            "none",
            "cluster",
            "lsa",
            "luhn",
            "lex_rank",
            "text_rank",
            "edmundson",
            "random",
        ],
        help="which extractive summarization approach to use. more information in documentation.",
    )
    PARSER.add_argument(
        "-sa",
        "--summarization_abs",
        default="bart",
        choices=["none", "bart", "presumm"],
        help="which abstractive summarization approach/model to use. more information in documentation.",
    )
    PARSER.add_argument(
        "-ss",
        "--summarization_structured",
        default="structured_joined",
        choices=["structured_joined", "none"],
        help="""An additional summarization algorithm that creates a structured summary with 
                figures, slide content (with bolded area), and summarized transcript content 
                from the SSA (Slide Structure Analysis) and transcript JSON data.""",
    )
    PARSER.add_argument(
        "-tm",
        "--transcription_method",
        default="vosk",
        choices=["sphinx", "google", "youtube", "deepspeech", "vosk"],
        help="""specify the program that should be used for transcription. 
                        CMU Sphinx: use pocketsphinx
                        Google Speech Recognition: probably will require chunking (online, free, max 1 minute chunks)
                        YouTube: download a video transcript from YouTube based on `--video_id`
                        DeepSpeech: Use the deepspeech library (fast with good GPU)
                        Vosk: Use the vosk library (extremely small low-resource model with great accuracy, this is the default)""",
    )
    PARSER.add_argument(
        "--transcribe_segment_sentences",
        action="store_false",
        help="Disable DeepSegment automatic sentence boundary detection. Specifying this option will output transcripts without punctuation."
    )
    PARSER.add_argument(
        "-sc",
        "--spell_check",
        default=["ocr"],
        choices=["ocr", "transcript"],
        nargs="+",
        help="option to perform spell checking on the ocr results of the slides or the voice transcript or both",
    )
    PARSER.add_argument(
        "--video_id",
        type=str,
        metavar="ID",
        help="id of youtube video to get subtitles from. set `--transcription_method` to `youtube` for this argument to take effect.",
    )
    PARSER.add_argument(
        "--transcribe_model_dir",
        type=str,
        metavar="DIR",
        help="path containing the model files for Vosk/DeepSpeech if `--transcription_method` is set to one of those models. See the documentation for details.",
    )
    PARSER.add_argument(
        "--tensorboard",
        default="",
        type=str,
        metavar="PATH",
        help="Path to tensorboard logdir. Tensorboard not used if not set. Tensorboard only used to visualize cluster primarily for debugging.",
    )
    PARSER.add_argument(
        "--bart_checkpoint",
        default=None,
        type=str,
        metavar="PATH",
        help="[BART Abstractive Summarizer Only] Path to optional checkpoint. Semsim is better model but will use more memory and is an additional 5GB download. (default: none, recommended: semsim)",
    )
    PARSER.add_argument(
        "--bart_state_dict_key",
        default="model",
        type=str,
        metavar="PATH",
        help='[BART Abstractive Summarizer Only] model state_dict key to load from pickle file specified with --bart_checkpoint (default: "model")',
    )
    PARSER.add_argument(
        "--bart_fairseq",
        action="store_true",
        help="[BART Abstractive Summarizer Only] Use fairseq model from torch hub instead of huggingface transformers library models. Can not use --bart_checkpoint if this option is supplied.",
    )
    PARSER.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    ARGS = PARSER.parse_args()

    # Perform argument checks
    if (ARGS.transcription_method == "deepspeech" or ARGS.transcription_method == "vosk") and ARGS.transcribe_model_dir is None:
        PARSER.error(
            "DeepSpeech and Vosk methods requires --transcribe_model_dir to be set to the directory containing the deepspeech/vosk models. See the documentation for details."
        )

    if (ARGS.summarization_mods is not None) and (
        "none" in ARGS.summarization_mods and len(ARGS.summarization_mods) > 1
    ):  # None and another option were specified
        PARSER.error(
            "If 'none' is specified in --summarization_mods then no other options can be selected."
        )

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(ARGS.logLevel),
    )

    main(ARGS)
