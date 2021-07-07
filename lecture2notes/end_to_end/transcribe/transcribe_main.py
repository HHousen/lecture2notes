import contextlib
import glob
import json
import logging
import os
import wave
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import sox
import spacy
import speech_recognition as sr
import torch
import webvtt
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from vosk import KaldiRecognizer, Model

from ..transcript_downloader import TranscriptDownloader
from . import webrtcvad_utils

logger = logging.getLogger(__name__)


def extract_audio(video_path, output_path):
    """Extracts audio from video at ``video_path`` and saves it to ``output_path``"""
    logger.info(
        "Extracting audio from "
        + str(video_path)
        + " and saving to "
        + str(output_path)
    )
    command = (
        "ffmpeg -i " + str(video_path) + " -f wav -ab 192000 -vn " + str(output_path)
    )
    os.system(command)
    return output_path


def transcribe_audio(audio_path, method="sphinx", **kwargs):
    """Transcribe audio using DeepSpeech, Vosk, or a method offered by
    :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.transcribe_audio_generic`.

    Args:
        audio_path (str): Path to the audio file to transcribe.
        method (str, optional): The method to use for transcription. Defaults to "sphinx".
        ``**kwargs``: Passed to the transcription function.

    Returns:
        tuple: (transcript_text, transcript_json)
    """
    if method == "vosk":
        return transcribe_audio_vosk(audio_path, **kwargs)
    if method == "deepspeech":
        return transcribe_audio_deepspeech(audio_path, **kwargs)
    if method == "wav2vec":
        return transcribe_audio_wav2vec(audio_path, **kwargs)
    return transcribe_audio_generic(audio_path, method, **kwargs), None


def transcribe_audio_generic(audio_path, method="sphinx", **kwargs):
    """Transcribe an audio file using CMU Sphinx or Google through the speech_recognition library

    Arguments:
        audio_path (str): audio file path
        method (str, optional): which service to use for transcription ("google" or "sphinx").
            Default is "sphinx".

    Returns:
        str: the transcript of the audio file
    """
    if method not in ["sphinx", "google"]:
        raise AssertionError
    transcript = None
    logger.debug("Initializing speech_recognition library")
    r = sr.Recognizer()

    with sr.AudioFile(str(audio_path)) as source:
        audio = r.record(source)

    try:
        logger.info("Transcribing file at " + str(audio_path))
        if method == "sphinx":
            transcript = r.recognize_sphinx(audio)
        elif method == "google":
            transcript = r.recognize_google(audio)
        else:
            logger.error("Incorrect method to transcribe audio")
            return -1
        return transcript
    except sr.UnknownValueError:
        logger.error("Could not understand audio")
    except sr.RequestError as e:
        logger.error("Error; {}".format(e))

    return transcript


def load_vosk_model(model_dir):
    if type(model_dir) is Model:
        return model_dir
    model = Model(model_dir)
    return model


def transcribe_audio_vosk(
    audio_path_or_chunks,
    model="../vosk_models",
    chunks=False,
    desired_sample_rate=16000,
    chunk_size=2000,
    **kwargs,
):
    """Transcribe audio using a ``vosk`` model.

    Args:
        audio_path_or_chunks (str or generator): Path to an audio file or a generator of chunks created by :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.chunk_by_speech`
        model (str or vosk.Model, optional): Path to the directory containing the ``vosk`` models or loaded ``vosk.Model``. Defaults to "../vosk_models".
        chunks (bool, optional): If the `audio_path_or_chunks` is chunks. Defaults to False.
        desired_sample_rate (int, optional): The sample rate that the model requires to convert audio to. Defaults to 16000.
        chunk_size (int, optional): The number of wave frames per loop. Amount of audio data transcribed at a time. Defaults to 2000.

    Returns:
        tuple: (text_transcript, results_json) The transcript as a string and as JSON.
    """
    if chunks:
        audio = audio_path_or_chunks
    else:
        pcm_data, sample_rate, duration = read_wave(
            audio_path_or_chunks, desired_sample_rate, force=False
        )
        if type(pcm_data) is bytes:
            pcm_data = np.frombuffer(pcm_data)
        pcm_data = np.array_split(pcm_data, pcm_data.shape[0] / chunk_size)
        audio = pcm_data

    model = load_vosk_model(model)
    rec = KaldiRecognizer(model, desired_sample_rate)

    results = []
    for data in tqdm(audio, desc="Vosk Transcribing"):
        # if data.size == 0:
        #     break
        if type(data) is np.ndarray:
            data = data.tobytes()
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result = json.loads(result)
            if result["text"] != "":
                # input(result["text"])
                results.extend(result["result"])
        # else:
        #     partial_result = rec.PartialResult()
        #     if partial_result is not None:
        #         partial_result = json.loads(partial_result)
        #         if partial_result["partial"] != "":
        #             input(partial_result["partial"])
        #             results.append(partial_result["partial"])

    final_result = rec.FinalResult()
    if final_result is not None:
        final_result = json.loads(final_result)
        if final_result["text"] != "":
            results.extend(final_result["result"])

    results_json = results
    results_text = [x["word"] if type(x) is dict else x for x in results]

    return " ".join(results_text), results_json


def load_wav2vec_model(
    model="facebook/wav2vec2-base-960h",
    tokenizer="facebook/wav2vec2-base-960h",
    **kwargs,
):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(model)
    return model, tokenizer


def transcribe_audio_wav2vec(
    audio_path_or_chunks, model=None, chunks=False, desired_sample_rate=16000
):
    if model is None:
        model = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-base-960h"]
    if isinstance(model, str):
        model = [model] * 2
    if isinstance(model[0], str):
        model, tokenizer = load_wav2vec_model(model[0], model[1])
    else:
        model, tokenizer = model[0], model[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if chunks:
        audio = audio_path_or_chunks
    else:
        pcm_data, sample_rate, duration = read_wave(
            audio_path_or_chunks, desired_sample_rate, force=False
        )
        if type(pcm_data) is bytes:
            pcm_data = np.frombuffer(pcm_data, dtype=np.float64)
        chunk_len = 15  # 15 seconds
        num_chunks = int(duration / chunk_len)
        pcm_data = np.array_split(pcm_data, num_chunks)
        audio = pcm_data

    # tokenize speech
    final_transcript = []
    for data in tqdm(audio, desc="Wav2Vec Transcribing"):
        data = data.astype("float64")
        input_values = tokenizer(
            data, return_tensors="pt", padding="longest", truncation="longest_first"
        ).input_values
        # input_values = input_values.type(torch.long)
        # retrieve logits
        input_values = input_values.to(device)
        logits = model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)

        combined_transcript = " ".join(transcription).strip()
        final_transcript.append(combined_transcript.lower())
    return " ".join(final_transcript).strip(), None


def read_wave(path, desired_sample_rate=None, force=False):
    """Reads a ".wav" file and converts to ``desired_sample_rate`` with one channel.

    Arguments:
        path (str): path to wave file to load
        desired_sample_rate (int, optional): resample the loaded pcm data from the wave file
            to this sample rate. Default is None, no resampling.
        force (bool, optional): Force the audio to be converted even if it is detected to meet
            the necessary criteria.

    Returns:
        tuple: (PCM audio data, sample rate, duration)
    """
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise AssertionError
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / sample_rate

        # if no `desired_sample_rate` then resample to the current `sample_rate` (no effect)
        if not desired_sample_rate:
            desired_sample_rate = sample_rate

        num_channels = wf.getnchannels()  # stereo or mono
        if num_channels == 1 and sample_rate == desired_sample_rate and not force:
            # no resampling is needed
            pcm_data = wf.readframes(frames)
        else:
            # different warning message depending on the problem (or both messages)
            if num_channels != 1:
                logger.warn(
                    "Resampling to one channel since {} channels were detected".format(
                        num_channels
                    )
                )
            if sample_rate != desired_sample_rate:
                logger.warn(
                    "Original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.".format(
                        sample_rate, desired_sample_rate
                    )
                )

            # run resampling (automatically converts to one channel)
            sample_rate, pcm_data = convert_samplerate(path, desired_sample_rate)

    return pcm_data, sample_rate, duration


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def segment_sentences(text, text_json=None, do_capitalization=True):
    """Detect sentence boundaries without punctuation or capitalization.

    Arguments:
        text (str): The string to segment by sentence.
        text_json (str or dict, optional): If the detected sentence boundaries should
            be applied to the JSON format of a transcript. Defaults to None.
        do_capitalization (bool, optiona): If the first letter of each detected sentence
            should be capitalized. Defaults to True.

    Returns:
        str: The punctuated (and optionally capitalized) string
    """
    from deepsegment import DeepSegment

    logger.info("Segmenting transcript using linguistic features...")
    inference_start = timer()

    segmenter = DeepSegment("en")
    segmented_text = segmenter.segment_long(text)

    inference_end = timer() - inference_start
    logger.info(
        "Segmentation Successful! It took "
        + str(inference_end)
        + " to split transcript into sentences."
    )

    if do_capitalization:
        segmented_text = [x.capitalize() for x in segmented_text]

    # add periods after each predicted sentence boundary
    final_text = ". ".join(segmented_text).strip()
    # add period to final sentence
    final_text += "."

    if text_json is not None:
        if type(text_json) is str:
            text_json = json.loads(text_json)

        boundaries = [len(sentence.split(" ")) for sentence in segmented_text]

        if do_capitalization:
            text_json[0]["word"] = text_json[0]["word"].title()

        for idx, boundary in enumerate(boundaries):
            if idx != 0:
                boundary += boundaries[idx - 1] + 1
                boundaries[idx] = boundary
            text_json.insert(boundary, {"start": 0, "end": 0, "word": "."})

            if do_capitalization:
                try:
                    text_json[boundary + 1]["word"] = text_json[boundary + 1][
                        "word"
                    ].title()
                except IndexError:
                    pass

        return final_text, json.dumps(text_json)

    return final_text, None


def metadata_to_string(metadata):
    """Helper function to convert metadata tokens from deepspeech to a string."""
    return "".join(token.text for token in metadata.tokens)


def metadata_to_json(candidate_transcript):
    """Helper function to convert metadata tokens from deepspeech to a dictionary."""
    json_result = {
        "confidence": candidate_transcript.confidence,
        "tokens": [
            {
                "start_time": token.start_time,
                "text": token.text,
                "timestep": token.timestep,
            }
            for token in candidate_transcript.tokens
        ],
    }
    return json_result


def metadata_to_list(candidate_transcript):
    json = metadata_to_json(candidate_transcript)
    return json["tokens"]


def convert_samplerate(audio_path, desired_sample_rate):
    """Use `SoX` to resample wave files to 16 bits, 1 channel, and ``desired_sample_rate`` sample rate.

    Arguments:
        audio_path (str): path to wave file to process
        desired_sample_rate (int): sample rate in hertz to convert the wave file to

    Returns:
        tuple: (desired_sample_rate, output) where ``desired_sample_rate`` is the new
            sample rate and ``output`` is the newly resampled pcm data
    """
    tfm = sox.Transformer()
    tfm.set_output_format(rate=desired_sample_rate, channels=1)
    output = tfm.build_array(input_filepath=str(audio_path))

    return desired_sample_rate, output


def resolve_deepspeech_models(dir_name):
    """Resolve directory path for deepspeech models and fetch each of them.

    Arguments:
        dir_name (str): Path to the directory containing pre-trained models

    Returns:
        tuple: a tuple containing each of the model files (pb, scorer)
    """
    pb = glob.glob(dir_name + "/*.pbmm")[0]
    logging.debug("Found model: %s", pb)

    scorer = glob.glob(dir_name + "/*.scorer")[0]
    logging.debug("Found scorer: %s", scorer)

    return pb, scorer


def load_deepspeech_model(model_dir, beam_width=500, lm_alpha=None, lm_beta=None):
    """Load the deepspeech model from ``model_dir``

    Arguments:
        model_dir (str): path to folder containing the ".pbmm" and optionally ".scorer" files
        beam_width (int, optional): beam width for decoding. Default is 500.
        lm_alpha (float, optional}: alpha parameter of language model. Default is None.
        lm_beta (float, optional): beta parameter of langage model. Default is None.

    Returns:
        deepspeech.Model: the loaded deepspeech model
    """
    from deepspeech import Model as ds_Model

    model, scorer = resolve_deepspeech_models(model_dir)
    logger.debug("Loading model...")
    model = ds_Model(model)
    model.setBeamWidth(beam_width)

    if scorer:
        logger.debug("Loading scorer from files {}".format(scorer))
        model.enableExternalScorer(scorer)

        if lm_alpha and lm_beta:
            model.setScorerAlphaBeta(lm_alpha, lm_beta)

    return model


def load_model(method, *args, **kwargs):
    if method == "deepspeech":
        return load_deepspeech_model(*args, **kwargs)
    if method == "vosk":
        return load_vosk_model(*args, **kwargs)
    if method == "wav2vec":
        return load_wav2vec_model(*args, **kwargs)
    logger.error("There is no method with name '%s'", method)


def transcribe_audio_deepspeech(
    audio_path_or_data, model, raw_audio_data=False, json_num_transcripts=None, **kwargs
):
    """Transcribe an audio file or pcm data with the deepspeech model

    Args:
        audio_path_or_data (str or byte string): a path to a wave file or a byte string
            containing pcm data from a wave file. set ``raw_audio_data`` to True if pcm data
            is used.
        model (deepspeech model or str): a deepspeech model object or a path to a folder
            containing the model files (see :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.load_deepspeech_model`)
        raw_audio_data (bool, optional): must be True if ``audio_path_or_data`` is
            raw pcm data. Defaults to False.
        json_num_transcripts (str, optional): Specify this value to generate multiple transcipts
            in json format.

    Returns:
        tuple: (transcript_text, transcript_json) the transcribed audio file in string format
        and the transcript in json
    """
    if isinstance(model, str):
        model = load_deepspeech_model(model)

    # load audio
    if raw_audio_data:
        audio = np.frombuffer(audio_path_or_data, np.int16)
    else:
        desired_sample_rate = model.sampleRate()
        pcm_data, sample_rate, duration = read_wave(
            audio_path_or_data, desired_sample_rate
        )

        audio = np.frombuffer(pcm_data, np.int16)

    logger.debug("Transcribing audio file...")
    inference_start = timer()
    if json_num_transcripts and json_num_transcripts > 1:
        model_output_metadata = model.sttWithMetadata(audio, json_num_transcripts)
        transcript_json = json.dumps(
            {
                "transcripts": [
                    metadata_to_list(candidate_transcript)
                    for candidate_transcript in model_output_metadata
                ]
            }
        )
        model_output_metadata = model_output_metadata.transcripts[0]
    else:
        model_output_metadata = model.sttWithMetadata(audio, 1).transcripts[0]
        transcript_json = metadata_to_list(model_output_metadata)
    transcript_text = metadata_to_string(model_output_metadata)

    # Convert deepspeech json from letter-by-letter to word-by-word
    transcript_json_converted = convert_deepspeech_json(transcript_json)

    inference_end = timer() - inference_start
    logger.debug("Inference (transcription) took %0.3fs.", inference_end)

    return transcript_text, transcript_json_converted


def convert_deepspeech_json(transcript_json):
    """Convert a deepspeech json transcript from a letter-by-letter format to word-by-word.

    Args:
        transcript_json (dict or str): The json format transcript as a dictionary or a json
            string, which will be loaded using ``json.loads()``.

    Returns:
        dict: The word-by-word transcript json.
    """
    if type(transcript_json) is str:
        transcript_json = json.loads(transcript_json)

    final_transcript_json = []
    current_word = ""
    start_time = 0
    looking_for_word_end = None
    for char_details in transcript_json:
        at_word_end = char_details["text"] == " " or char_details["text"] == "."
        if at_word_end and looking_for_word_end:
            end_time = char_details["start_time"]
            final_transcript_json.append(
                {"start": start_time, "end": end_time, "word": current_word}
            )
            current_word = ""
            looking_for_word_end = False

        elif char_details["text"] != " ":
            start_time = char_details["start_time"]
            looking_for_word_end = True
            current_word += char_details["text"]

        if char_details["text"] == ".":
            final_transcript_json.append(
                {
                    "start": char_details["start_time"],
                    "end": char_details["start_time"],
                    "word": char_details["text"],
                }
            )

    end_time = transcript_json[-1]["start_time"]
    final_transcript_json.append(
        {"start": start_time, "end": end_time, "word": current_word}
    )

    return final_transcript_json


def write_to_file(
    transcript,
    transcript_save_file,
    transcript_json=None,
    transcript_json_save_path=None,
):
    """Write ``transcript`` to ``transcript_save_file`` and ``transcript_json`` to ``transcript_json_save_path``."""
    with open(transcript_save_file, "w+") as file_results:
        logger.info("Writing text transcript to file " + str(transcript_save_file))
        file_results.write(transcript)

    if transcript_json and transcript_json_save_path:
        with open(transcript_json_save_path, "w+") as file_results:
            logger.info(
                "Writing JSON transcript to file " + str(transcript_json_save_path)
            )
            file_results.write(transcript_json)


def chunk_by_speech(
    audio_path, output_path=None, aggressiveness=1, desired_sample_rate=None
):
    """
    Uses the python interface to the WebRTC Voice Activity Detector (VAD) API to
    create chunks of audio that contain voice. The VAD that Google developed for
    the WebRTC project is reportedly one of the best available, being fast, modern
    and free.

    Args:
        audio_path (str): path to the audio file to process
        output_path (str, optional): path to save the chunk files. if not specified then no wave
            files will be written to disk and the raw pcm data will be returned. Defaults to None.
        aggressiveness (int, optional): determines how aggressive filtering out non-speech is. must
            be an interger between 0 and 3. Defaults to 1.
        desired_sample_rate (int, optional): the sample rate of the returned segments. the default is
            the same rate of the input audio file. Defaults to None.

    Returns:
        tuple: (segments, sample_rate, audio_length). See :meth:`~lecture2notes.end_to_end.transcribe.webrtcvad_utils.vad_segment_generator`.
    """
    if desired_sample_rate:
        if desired_sample_rate not in (
            8000,
            16000,
            32000,
            48000,
        ):
            raise AssertionError("The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.")

    segments, sample_rate, audio_length = webrtcvad_utils.vad_segment_generator(
        audio_path,
        aggressiveness=aggressiveness,
        desired_sample_rate=desired_sample_rate,
    )
    if output_path:
        for i, segment in tqdm(
            enumerate(segments), total=len(segments), desc="Writing Chunks"
        ):
            chunk_number = "chunk" + f"{i:05}" + ".wav"
            logger.debug("Exporting " + chunk_number)
            save_path = Path(output_path) / chunk_number
            write_wave(save_path, segment, sample_rate)

    return segments, sample_rate, audio_length


def process_segments(
    segments,
    model,
    audio_length="unknown",
    method="deepspeech",
    do_segment_sentences=True,
):
    """Transcribe a list of byte strings containing pcm data

    Args:
        segments (list): list of byte strings containing pcm data (generated by :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.chunk_by_speech`)
        model (deepspeech model): a deepspeech model object or a path to a folder
            containing the model files (see :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.load_deepspeech_model`).
        audio_length (str, optional): the length of the audio file if known (used for logging statements)
            Default is "unknown".
        method (str, optional): The model to use to perform speech-to-text. Supports 'deepspeech' and
            'vosk'. Defaults to "deepspeech".
        do_segment_sentences (bool, optional): Find sentence boundaries using
            :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.segment_sentences`. Defaults to True.

    Returns:
        tuple: (full_transcript, full_transcript_json) The combined transcript of all the items in
        ``segments`` as a string and as dictionary/json.
    """
    if method == "deepspeech" and isinstance(model, str):
        model = load_deepspeech_model(model)
    elif model == "vosk":
        model = load_vosk_model(model)
    elif model == "wav2vec":
        model = load_wav2vec_model()

    create_json = True
    full_transcript = ""
    full_transcript_json = []
    start_time = timer()

    # `segments` is a generator so total length is not known
    # also, this means that the audio file is split on voice activity as needed
    # (the entire file is not split and stored in memory at once)
    if method == "deepspeech":
        for i, segment in tqdm(enumerate(segments), desc="Processing Segments"):
            # Run deepspeech on each chunk which completed VAD
            audio = np.frombuffer(segment, dtype=np.int16)
            transcript, transcript_json = transcribe_audio_deepspeech(
                segment, model, raw_audio_data=True
            )

            logging.debug("Chunk Transcript: %s", transcript)

            full_transcript_json.extend(transcript_json)

            full_transcript += transcript + " "
    elif method == "vosk":
        full_transcript, full_transcript_json = transcribe_audio_vosk(
            segments, model, chunks=True
        )
    elif method == "wav2vec":
        create_json = False
        for i, segment in tqdm(enumerate(segments), desc="Processing Segments"):
            audio = np.frombuffer(segment)
            transcript = transcribe_audio_wav2vec(audio, model, chunks=True)

            full_transcript += transcript + " "

    if create_json:
        # Convert `full_transcript_json` to json string
        full_transcript_json = json.dumps(full_transcript_json)
    else:
        full_transcript_json = None

    total_time = timer() - start_time
    logger.info(
        "It took "
        + str(total_time)
        + " to transcribe an audio file with duration "
        + str(audio_length)
        + "."
    )
    logger.info("The above time includes time spent determining voice activity.")

    if do_segment_sentences:
        full_transcript, full_transcript_json = segment_sentences(
            full_transcript, full_transcript_json
        )

    return full_transcript, full_transcript_json


def chunk_by_silence(
    audio_path, output_path, silence_thresh_offset=5, min_silence_len=2000
):
    """Split an audio file into chunks on areas of silence

    Arguments:
        audio_path (str): path to a wave file
        output_path (str): path to a folder where wave file chunks will be saved
        silence_thresh_offset (int, optional): a value subtracted from the mean dB volume of
            the file. Default is 5.
        min_silence_len (int, optional): the length in milliseconds in which there must be no sound
            in order to be marked as a splitting point. Default is 2000.
    """
    logger.info("Loading audio")
    audio = AudioSegment.from_wav(audio_path)
    logger.info("Average loudness of audio track is " + str(audio.dBFS))
    silence_thresh = audio.dBFS - silence_thresh_offset
    logger.info("Silence Threshold of audio track is " + str(silence_thresh))
    logger.info(
        "Minimum silence length for audio track is " + str(min_silence_len) + " ms"
    )
    logger.info("Creating chunks")
    chunks = split_on_silence(
        # Use the loaded audio.
        audio,
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len=min_silence_len,
        # Consider a chunk silent if it's quieter than `silence_thresh` dBFS.
        silence_thresh=silence_thresh,
    )
    logger.info("Created " + str(len(chunks)) + " chunks")

    os.makedirs(output_path, exist_ok=True)

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Writing Chunks"):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)
        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Export the audio chunk with new bitrate.
        chunk_number = "chunk" + f"{i:05}" + ".wav"
        logger.debug("Exporting " + chunk_number)
        save_path = Path(output_path) / chunk_number
        audio_chunk.export(str(save_path.resolve()), bitrate="192k", format="wav")


def process_chunks(chunk_dir, method="sphinx", model_dir=None):
    """
    Performs transcription on every noise activity chunk (audio file) created by
    :meth:`~lecture2notes.end_to_end.transcribe.transcribe_main.chunk_by_silence` in a directory.
    """
    chunks = os.listdir(chunk_dir)
    chunks.sort()
    full_transcript = ""
    full_transcript_json = []

    for chunk in tqdm(chunks, desc="Processing Chunks"):
        if chunk.endswith(".wav"):
            chunk_path = Path(chunk_dir) / chunk
            if method == "deepspeech" or method == "vosk":
                if model_dir is None:
                    raise AssertionError
                model = load_model(method, model_dir)
                transcript, transcript_json = transcribe_audio(
                    chunk_path, method, model=model
                )
                full_transcript_json.extend(json.loads(transcript_json))
            else:
                transcript = transcribe_audio_generic(chunk_path, method)
            full_transcript += transcript + " "

    # Convert `full_transcript_json` to json string if it contains any items
    if full_transcript_json:
        full_transcript_json = json.dumps(full_transcript_json)
        return full_transcript, full_transcript_json

    return full_transcript, None


def caption_file_to_string(transcript_path, remove_speakers=False):
    """
    Converts a .srt, .vtt, or .sbv file saved at ``transcript_path`` to a python string.
    Optionally removes speaker entries by removing everything before ": " in each subtitle cell.
    """
    transcript_path = Path(transcript_path)
    if not transcript_path.is_file():
        raise AssertionError
    if transcript_path.suffix == ".srt":
        subtitles = webvtt.from_srt(transcript_path)
    elif transcript_path.suffix == ".sbv":
        subtitles = webvtt.from_sbv(transcript_path)
    elif transcript_path.suffix == ".vtt":
        subtitles = webvtt.read(transcript_path)
    else:
        return None, None

    transcript = ""
    transcript_json = []
    for subtitle in subtitles:
        content = subtitle.text.replace("\n", " ")  # replace newlines with space
        if remove_speakers:
            content = content.split(": ", 1)[-1]  # remove everything before ": "
        transcript += content + " "  # add space after each subtitle block in srt file
        transcript_json.append(
            {
                "end": subtitle.end_in_seconds,
                "start": subtitle.start_in_seconds,
                "word": content,
            }
        )
    return transcript, json.dumps(transcript_json)


def get_youtube_transcript(video_id, output_path, use_youtube_dl=True):
    """Downloads the transcript for ``video_id`` and saves it to ``output_path``"""
    downloader = TranscriptDownloader(ytdl=use_youtube_dl)
    transcript_path = downloader.download(video_id, output_path)
    return transcript_path


def check_transcript(generated_transcript, ground_truth_transcript):
    """Compares ``generated_transcript`` to ``ground_truth_transcript`` to check for accuracy using spacy similarity measurement. Requires the "en_vectors_web_lg" model to use "real" word vectors."""
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded Spacy `en_vectors_web_lg`")
    gen_doc = nlp(generated_transcript)
    logger.info("NLP done on generated_transcript")
    real_doc = nlp(ground_truth_transcript)
    logger.info("NLP done on ground_truth_transcript")
    similarity = gen_doc.similarity(real_doc)
    logger.info("Similarity Computed: " + str(similarity))
    return similarity


# extract_audio("nykOeWgQcHM.mp4", "process/audio.wav")
# create_chunks("process/audio-short.wav", "process/chunks", 5, 2000)
# process_chunks("process/chunks", "process/output.txt")

# transcript = caption_file_to_string(Path("test.srt"))
# print(transcript)

# print(get_youtube_transcript("TtaWB0bL3zQ", Path("subtitles.vtt")))

# generated_transcript = open(Path("process/audio.txt"), "r").read()
# ground_truth_transcript = transcript
# similarity = check_transcript(generated_transcript, ground_truth_transcript)
# print(similarity)

# transcript = transcribe_audio_deepspeech("process/audio.wav", "../deepspeech-models")
# print(transcript)

# result, _ = transcribe_audio_vosk("process/audio.wav", "../vosk-model-en-us-daanzu-20200905")
# print(result)
# load_vosk_model("../vosk-model-en-us-daanzu-20200905")

# segments, _, audio_length = chunk_by_speech(
#     "process/audio.wav", desired_sample_rate=16000
# )
# transcript, transcript_json = process_segments(
#     segments,
#     "../deepspeech-models",
#     method="deepspeech",
#     audio_length=audio_length,
#     do_segment_sentences=True,
# )
# print(transcript)
# print(" ".join([x["word"] for x in json.loads(transcript_json)]))
# write_to_file(transcript, "process/audio.txt", transcript_json, "process/audio.json")
