import os
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
    OUTPUT_DIR_YT_FORMAT = str(OUTPUT_DIR_YT) + "/%(id)s/%(id)s.%(ext)s"
    ydl_opts = {
        "format": "bestaudio[ext=m4a]",
        "outtmpl": OUTPUT_DIR_YT_FORMAT,
    }
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # Remove the transcripts created by the ML model
    transcripts = [x for x in transcripts if ARGS.suffix not in x]

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


            video_audio_path = process_folder / (video_id + ".m4a")

            end_time = timer() - start_time
            logger.info("Stage 1 (Download Audio) took %s" % end_time)

            # Convert audio
            start_time = timer()
            audio_path = process_folder / "audio.wav"
            transcribe.extract_audio(video_audio_path, audio_path)

            end_time = timer() - start_time
            logger.info("Stage 2 (Convert Audio) took %s" % end_time)

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
