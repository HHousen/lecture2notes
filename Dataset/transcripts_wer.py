import os
import re
import sys
import shutil
import logging
import argparse
import traceback
import youtube_dl
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer

import jiwer

sys.path.insert(1, os.path.join(sys.path[0], "../End-To-End"))
from transcribe import transcribe_main as transcribe

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
    "--deepspeech_dir",
    type=str,
    default="../deepspeech-models",
    help="path to the directory containing the DeepSpeech models",
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
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # Remove the transcripts created by the ML model
    transcripts = [x for x in transcripts if ARGS.suffix not in x]

    # Remove entries that are not files
    transcripts = [x for x in transcripts if os.path.isfile(TRANSCRIPTS_DIR / x)]

    # Create DeepSpeech model
    ds_model = transcribe.load_deepspeech_model(ARGS.deepspeech_dir)
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

            audio_path = process_folder / (video_id + ".wav")

            end_time = timer() - start_time
            logger.info("Stage 1 (Download and Convert Audio) took %s" % end_time)

            # Transcribe
            start_time = timer()
            if ARGS.no_chunk:
                transcript = transcribe.transcribe_audio_deepspeech(
                    audio_path, ds_model
                )
            else:
                desired_sample_rate = ds_model.sampleRate()
                segments, _, audio_length = transcribe.chunk_by_speech(
                    audio_path, desired_sample_rate=desired_sample_rate
                )
                transcript = transcribe.process_segments(
                    segments,
                    ds_model,
                    audio_length=audio_length,
                    do_segment_sentences=False,
                )

            end_time = timer() - start_time
            logger.info("Stage 2 (Transcribe) took %s" % end_time)

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

    errors = list()
    transcripts_tqdm = tqdm(transcripts, desc="Calculating WER")
    for transcript_ml_path in transcripts_tqdm:
        video_id = re.search("(.*)" + ARGS.suffix, transcript_ml_path)
        video_id = video_id.group(1)

        transcript_ml_path = TRANSCRIPTS_DIR / transcript_ml_path

        transcript_path = str(transcript_ml_path)[: -(4 + len(ARGS.suffix))] + ".vtt"
        transcript_ground_truth = transcribe.caption_file_to_string(
            transcript_path, remove_speakers=True
        )

        with open(transcript_ml_path, "r") as file:
            transcript_prediction = file.read()

        measures = jiwer.compute_measures(
            transcript_ground_truth,
            transcript_prediction,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )

        transcripts_tqdm.write(
            video_id
            + " WER: "
            + str(measures["wer"])
            + "  MER: "
            + str(measures["mer"])
            + "  WIL: "
            + str(measures["wil"])
        )
        errors.append(measures)

    num_errors = len(errors)
    average_wer = sum([x["wer"] for x in errors]) / num_errors
    average_mer = sum([x["mer"] for x in errors]) / num_errors
    average_wil = sum([x["wil"] for x in errors]) / num_errors

    logger.info("Average WER: " + str(average_wer))
    logger.info("Average MER: " + str(average_mer))
    logger.info("Average WIL: " + str(average_wil))
