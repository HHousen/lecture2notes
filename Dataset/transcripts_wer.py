import os
import sys
import shutil
import logging
import argparse
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
    choices=["transcribe", "calc_wer"],
    help="`transcribe` each video and create a transcript using ML models or use `calc_wer` to compute the WER for the created transcripts",
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
    OUTPUT_DIR_YT_FORMAT = str(OUTPUT_DIR_YT) + "/%\(id\)s/%\(id\)s.%\(ext\)s"
    YT_FORMAT_STRING = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"

    # Remove the transcripts created by the ML model
    transcripts = [x for x in transcripts if ARGS.suffix not in x]

    # Create DeepSpeech model
    ds_model = transcribe.load_deepspeech_model(ARGS.deepspeech_dir)

    for transcript in tqdm(transcripts, desc="Transcribing"):
        video_id = transcript.split(".")[0]
        transcript_ml_path = TRANSCRIPTS_DIR / (transcript[:-4] + ARGS.suffix + ".txt")
        process_folder = OUTPUT_DIR_YT / video_id

        if not os.path.isfile(transcript_ml_path):
            # Download video
            start_time = timer()
            os.system(
                'youtube-dl -f "'
                + YT_FORMAT_STRING
                + '" -o '
                + OUTPUT_DIR_YT_FORMAT
                + " -- "
                + video_id
            )
            video_path = process_folder / (video_id + ".mp4")

            end_time = timer() - start_time
            logger.info("Stage 1 (Download Video) took %s" % end_time)

            # Extract audio
            start_time = timer()
            audio_path = process_folder / "audio.wav"
            transcribe.extract_audio(video_path, audio_path)

            end_time = timer() - start_time
            logger.info("Stage 2 (Extract Audio) took %s" % end_time)

            # Transcribe
            start_time = timer()
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
            logger.info("Stage 3 (Transcribe) took %s" % end_time)

            transcribe.write_to_file(transcript, transcript_ml_path)
            logger.info(
                "Wrote transcript to disk at location " + str(transcript_ml_path)
            )

            shutil.rmtree(process_folder)
            logger.info("Removed temp folder")

elif ARGS.mode == "calc_wer":
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(),
            jiwer.RemoveEmptyStrings(),
            jiwer.SentencesToListOfWords(word_delimiter=" "),
        ]
    )

    # Only select transcripts created by the ML model
    transcripts = [x for x in transcripts if ARGS.suffix in x]

    for transcript_ml_path in tqdm(transcripts, desc="Calculating WER"):
        transcript_ml_path = TRANSCRIPTS_DIR / transcript_ml_path

        transcript_path = str(transcript_ml_path)[: -(4 + len(ARGS.suffix))] + ".vtt"
        transcript_ground_truth = transcribe.caption_file_to_string(
            transcript_path, remove_speakers=True
        )

        with open(transcript_ml_path, "r") as file:
            transcript_prediction = file.read()

        error = jiwer.wer(
            transcript_ground_truth,
            transcript_prediction,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )

        print(error)
