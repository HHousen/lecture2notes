# Main process to convert video to notes (end-to-end)

import argparse
import logging

from ..models.slide_classifier.custom_nnmodules import *  # noqa: F403,F401
from .summarizer_class import LectureSummarizer

logger = logging.getLogger(__name__)


def main(ARGS):
    summarizer = LectureSummarizer(ARGS)
    summarizer.run_all()


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
        "--custom_id",
        type=str,
        default=None,
        help="same as `--auto_id` but will create a subdirectory using this value instead of a random id",
    )
    PARSER.add_argument(
        "-rm",
        "--remove",
        action="store_true",
        help="remove `process_dir` once conversion is complete",
    )
    PARSER.add_argument(
        "--extract_frames_quality",
        type=int,
        default=5,
        help="ffmpeg quality of extracted frames",
    )
    PARSER.add_argument(
        "--extract_every_x_seconds",
        type=int,
        default=1,
        help="how many seconds between extracted frames",
    )
    PARSER.add_argument(
        "--slide_classifier_model_path",
        type=str,
        default="./lecture2notes/end_to_end/model_best.ckpt",
        help="path to the slide classification model checkpoint",
    )
    PARSER.add_argument(
        "--east_path",
        type=str,
        default="./lecture2notes/end_to_end/frozen_east_text_detection.pb",
        help="path to the EAST text detector model",
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
        help="which combination algorithm to use. more information in documentation.",
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
        default="allenai/led-large-16384-arxiv",
        choices=[
            "none",
            "presumm",
            "sshleifer/distilbart-cnn-12-6",
            "patrickvonplaten/bert2bert_cnn_daily_mail",
            "facebook/bart-large-cnn",
            "allenai/led-large-16384-arxiv",
            "patrickvonplaten/led-large-16384-pubmed",
            "google/pegasus-billsum",
            "google/pegasus-cnn_dailymail",
            "google/pegasus-pubmed",
            "google/pegasus-arxiv",
            "google/pegasus-wikihow",
            "google/pegasus-big_patent",
        ],
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
        "--structured_joined_summarization_method",
        default="abstractive",
        choices=["none", "abstractive", "extractive"],
        help="The summarization method to use during `structured_joined` summarization.",
    )
    PARSER.add_argument(
        "--structured_joined_abs_summarizer",
        default="facebook/bart-large-cnn",
        choices=[
            "presumm",
            "sshleifer/distilbart-cnn-12-6",
            "patrickvonplaten/bert2bert_cnn_daily_mail",
            "facebook/bart-large-cnn",
            "allenai/led-large-16384-arxiv",
            "patrickvonplaten/led-large-16384-pubmed",
            "google/pegasus-billsum",
            "google/pegasus-cnn_dailymail",
            "google/pegasus-pubmed",
            "google/pegasus-arxiv",
            "google/pegasus-wikihow",
            "google/pegasus-big_patent",
        ],
        help="The abstractive summarizer to use during `structured_joined` summarization (to create summaries of each slide) if `structured_joined_summarization_method` is 'abstractive'.",
    )
    PARSER.add_argument(
        "--structured_joined_ext_summarizer",
        default="text_rank",
        choices=[
            "lsa",
            "luhn",
            "lex_rank",
            "text_rank",
            "edmundson",
            "random",
        ],
        help="The extractive summarizer to use during `structured_joined` summarization (to create summaries of each slide) if `--structured_joined_summarization_method` is 'extractive'.",
    )
    PARSER.add_argument(
        "-tm",
        "--transcription_method",
        default="vosk",
        choices=["sphinx", "google", "youtube", "deepspeech", "vosk", "wav2vec"],
        help="""specify the program that should be used for transcription.
                        CMU Sphinx: use pocketsphinx
                        Google Speech Recognition: probably will require chunking (online, free, max 1 minute chunks)
                        YouTube: download a video transcript from YouTube based on `--video_id`
                        DeepSpeech: Use the deepspeech library (fast with good GPU)
                        Vosk: Use the vosk library (extremely small low-resource model with great accuracy, this is the default)
                        Wav2Vec: State-of-the-art speech-to-text model through the `huggingface/transformers` library.""",
    )
    PARSER.add_argument(
        "--transcribe_segment_sentences",
        action="store_false",
        help="Disable DeepSegment automatic sentence boundary detection. Specifying this option will output transcripts without punctuation.",
    )
    PARSER.add_argument(
        "--custom_transcript_check",
        type=str,
        default=None,
        help="Check if a transcript file (follwed by an extension of vtt, srt, or sbv) with the specified name is in the processing folder and use it instead of running speech-to-text.",
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
        default="./vosk-models/vosk-model-en-us-small-0.3/",
        help="path containing the model files for Vosk/DeepSpeech if `--transcription_method` is set to one of those models. See the documentation for details.",
    )
    PARSER.add_argument(
        "--abs_hf_api",
        action="store_true",
        help="use the huggingface inference API for abstractive summarization tasks",
    )
    PARSER.add_argument(
        "--abs_hf_api_overall",
        action="store_true",
        help="use the huggingface inference API for final overall abstractive summarization task",
    )
    PARSER.add_argument(
        "--tensorboard",
        default="",
        type=str,
        metavar="PATH",
        help="Path to tensorboard logdir. Tensorboard not used if not set. Tensorboard only used to visualize cluster primarily for debugging.",
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
    if (
        ARGS.transcription_method == "deepspeech" or ARGS.transcription_method == "vosk"
    ) and ARGS.transcribe_model_dir is None:
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
