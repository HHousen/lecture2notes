import argparse
import logging
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from timeit import default_timer as timer

import jiwer
import yt_dlp as youtube_dl
from tqdm import tqdm

from ..end_to_end.transcribe import transcribe_main as transcribe

logger = logging.getLogger(__name__)

PARSER = argparse.ArgumentParser(
    description="Word Error Rate (WER) for Transcripts with DeepSpeech"
)

PARSER.add_argument(
    "mode",
    type=str,
    choices=["transcribe", "calc"],
    help="`transcribe` each video and create a transcript using ML models or use `calc` to compute the WER for the created transcripts",
)
PARSER.add_argument(
    "--transcripts_dir",
    type=str,
    default="./transcripts/",
    help="path to the directory containing transcripts downloaded with 2-video_downloader.py",
)
PARSER.add_argument(
    "--model_dir",
    type=str,
    default="../deepspeech-models",
    help="path to the directory containing the models for `--method`",
)
PARSER.add_argument(
    "--method",
    type=str,
    default=None,
    help="Method to use to transcribe. Any method allowed by `transcribe.transcribe_audio()` will work. Defaults to 'deepspeech'.",
)
PARSER.add_argument(
    "--audio_format",
    type=str,
    default="wav",
    help="The format to convert downloaded audio to. Raw WAV is required for most recognizers.",
)
PARSER.add_argument(
    "--suffix",
    type=str,
    default="_deepspeech",
    help="string added after the video id and before the extension in the transcript output from the ML model",
)
PARSER.add_argument(
    "--no_chunk", action="store_true", help="Disable audio chunking by voice activity."
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

logging.basicConfig(
    format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
    level=logging.getLevelName(ARGS.logLevel),
)

TRANSCRIPTS_DIR = Path(ARGS.transcripts_dir)
transcripts = os.listdir(TRANSCRIPTS_DIR)

if ARGS.mode == "transcribe":
    OUTPUT_DIR_YT = TRANSCRIPTS_DIR / "temp/"
    OUTPUT_DIR_YT_FORMAT = str(OUTPUT_DIR_YT) + "/%(id)s/%(id)s.%(ext)s"
    ydl_opts = {
        "format": "bestaudio[ext=m4a]",
        "outtmpl": OUTPUT_DIR_YT_FORMAT,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": ARGS.audio_format,
                "preferredquality": "192",
            }
        ],
    }
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # Remove the transcripts created by the ML model
    transcripts = [x for x in transcripts if ARGS.suffix not in x]

    # Remove entries that are not files
    transcripts = [x for x in transcripts if os.path.isfile(TRANSCRIPTS_DIR / x)]

    # Create DeepSpeech or Vosk model if using 'deepspeech' or 'vosk' method
    model = None
    if ARGS.method == "deepspeech":
        model = transcribe.load_deepspeech_model(ARGS.model_dir)
        desired_sample_rate = model.sampleRate()
    elif ARGS.method == "vosk":
        model = transcribe.load_vosk_model(ARGS.model_dir)
        desired_sample_rate = 16000
    elif ARGS.method == "whispercpp":
        model = transcribe.load_whispercpp_model(ARGS.model_dir)
        desired_sample_rate = 16000

    for transcript in tqdm(transcripts, desc="Transcribing"):
        video_id = transcript.split(".")[0]
        transcript_ml_path = TRANSCRIPTS_DIR / (transcript[:-4] + ARGS.suffix + ".txt")
        process_folder = OUTPUT_DIR_YT / video_id

        # Check to make sure the file has not already been transcribed using the ML model
        if not os.path.isfile(transcript_ml_path):
            # Download audio
            start_time = timer()
            tries = 0
            while tries < 3:
                try:
                    ydl.download([video_id])
                except youtube_dl.utils.DownloadError as e:
                    tries += 1
                    logger.info("Try Number " + str(tries))
                    logger.error("Full error: " + str(e))
                    if tries == 3:
                        traceback.print_exc()
                        sys.exit(1)
                # Break out of the loop if successful
                else:
                    break

            audio_path = process_folder / (video_id + "." + ARGS.audio_format)

            end_time = timer() - start_time
            logger.info("Stage 1 (Download and Convert Audio) took %s", end_time)

            # Transcribe
            start_time = timer()
            if ARGS.no_chunk:
                transcript, _ = transcribe.transcribe_audio(
                    audio_path, method=ARGS.method, model=model
                )
            else:
                segments, _, audio_length = transcribe.chunk_by_speech(
                    audio_path, desired_sample_rate=desired_sample_rate
                )
                transcript, _ = transcribe.process_segments(
                    segments,
                    model,
                    method=ARGS.method,
                    audio_length=audio_length,
                    do_segment_sentences=False,
                )

            end_time = timer() - start_time
            logger.info("Stage 2 (Transcribe) took %s", end_time)

            transcribe.write_to_file(transcript, transcript_ml_path)
            logger.info(
                "Wrote transcript to disk at location " + str(transcript_ml_path)
            )

            shutil.rmtree(process_folder)
            logger.info("Removed temp folder")

elif ARGS.mode == "calc":
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(word_delimiter=" "),
            jiwer.RemoveEmptyStrings(),
        ]
    )

    # Only select transcripts created by the ML model
    transcripts = [x for x in transcripts if ARGS.suffix in x]

    errors = []
    transcripts_tqdm = tqdm(transcripts, desc="Calculating WER")
    for transcript_ml_path in transcripts_tqdm:
        video_id = re.search("(.*)" + ARGS.suffix, transcript_ml_path)
        video_id = video_id.group(1)

        transcript_ml_path = TRANSCRIPTS_DIR / transcript_ml_path

        transcript_path = str(transcript_ml_path)[: -(4 + len(ARGS.suffix))] + ".vtt"
        transcript_ground_truth = transcribe.caption_file_to_string(
            transcript_path, remove_speakers=True
        )[0]

        with open(transcript_ml_path, "r") as file:
            transcript_prediction = file.read()

        measures = jiwer.compute_measures(
            transcript_ground_truth,
            transcript_prediction,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )
        measures["token_count_ground_truth"] = len(
            transformation(transcript_ground_truth)
        )
        measures["token_count_prediction"] = len(transformation(transcript_prediction))

        transcripts_tqdm.write(
            video_id
            + " WER: "
            + str(measures["wer"])
            + "  MER: "
            + str(measures["mer"])
            + "  WIL: "
            + str(measures["wil"])
            + "  GT Tokens: "
            + str(measures["token_count_ground_truth"])
            + "  Prediction Tokens: "
            + str(measures["token_count_prediction"])
        )
        errors.append(measures)

    num_errors = len(errors)
    average_wer = sum(x["wer"] for x in errors) / num_errors
    average_mer = sum(x["mer"] for x in errors) / num_errors
    average_wil = sum(x["wil"] for x in errors) / num_errors
    average_token_count_ground_truth = (
        sum(x["token_count_ground_truth"] for x in errors) / num_errors
    )
    average_token_count_prediction = (
        sum(x["token_count_prediction"] for x in errors) / num_errors
    )

    logger.info("Average WER: " + str(average_wer))
    logger.info("Average MER: " + str(average_mer))
    logger.info("Average WIL: " + str(average_wil))
    logger.info(
        "Average # Ground Truth Tokens: " + str(average_token_count_ground_truth)
    )
    logger.info("Average # Prediction Tokens: " + str(average_token_count_prediction))
