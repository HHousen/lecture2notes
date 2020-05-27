import os
import glob
import shlex
import wave
import contextlib
import logging
import subprocess
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
from timeit import default_timer as timer
from helpers import make_dir_if_not_exist
from transcript_downloader import TranscriptDownloader
import webvtt
import transcribe.webrtcvad_utils as webrtcvad_utils
import spacy
import numpy as np

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

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


def transcribe_audio(audio_path, method="sphinx"):
    """Transcribe an audio file using CMU Sphinx or Google through the speech_recognition library

    Arguments:
        audio_path (str): audio file path
        method (str, optional): which service to use for transcription ("google" or "sphinx").
            Default is "sphinx".

    Returns:
        [str]: the transcript of the audio file
    """
    assert method in ["sphinx", "google"]
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
        logger.error("Error; {0}".format(e))
    
    return transcript


def read_wave(path, desired_sample_rate=None):
    """Reads a ".wav" file and converts to `desired_sample_rate` with one channel.

    Arguments:
        path (str): path to wave file to load
        desired_sample_rate (int, optional): resample the loaded pcm data from the wave file
            to this sample rate Default is None, no resampling.

    Returns:
        [tuple]: (PCM audio data, sample rate, duration)
    """
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / sample_rate

        # if no `desired_sample_rate` then resample to the current `sample_rate` (no effect)
        if not desired_sample_rate:
            desired_sample_rate = sample_rate

        num_channels = wf.getnchannels()  # stereo or mono
        if num_channels == 1 and sample_rate == desired_sample_rate:
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


def segment_sentences(text):
    """Detect sentence boundaries without punctuation or capitalization.

    Arguments:
        text {str}: The string to segment by sentence

    Returns:
        [str]: The punctuated string
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

    # add periods after each predicted sentence boundary
    final_text = ". ".join(segmented_text)

    return final_text.strip()


def metadata_to_string(metadata):
    """ Helper function to handle metadata tokens from deepspeech """
    return "".join(token.text for token in metadata.tokens)


def metadata_json_output(metadata):
    """ Helper function to handle metadata json from deepspeech """
    json_result = dict()
    json_result["transcripts"] = [
        {
            "confidence": transcript.confidence,
            "words": words_from_candidate_transcript(transcript),
        }
        for transcript in metadata.transcripts
    ]
    return json.dumps(json_result, indent=2)


def convert_samplerate(audio_path, desired_sample_rate):
    """Use `SoX` to resample wave files to 16 bits, 1 channel, and `desired_sample_rate` sample rate.

    Arguments:
        audio_path (str): path to wave file to process
        desired_sample_rate (int): sample rate in hertz to convert the wave file to

    Raises:
        RuntimeError: if SoX returns a non-zero status
        OSError: if SoX is not found

    Returns:
        [tuple]: (desired_sample_rate, output) where ``desired_sample_rate`` is the new 
            sample rate and ``output`` is the newly resampled pcm data
    """
    sox_cmd = "sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - ".format(
        quote(str(audio_path)), desired_sample_rate
    )
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno,
            "SoX not found, use {}hz files or install it: {}".format(
                desired_sample_rate, e.strerror
            ),
        )

    return desired_sample_rate, output


def resolve_models(dir_name):
    """Resolve directory path for deepspeech models and fetch each of them.

    Arguments:
        dir_name (str): Path to the directory containing pre-trained models

    Returns:
        [tuple]: a tuple containing each of the model files (pb, scorer)
    """

    pb = glob.glob(dir_name + "/*.pbmm")[0]
    logging.debug("Found model: %s" % pb)

    scorer = glob.glob(dir_name + "/*.scorer")[0]
    logging.debug("Found scorer: %s" % scorer)

    return pb, scorer


def load_deepspeech_model(model_dir, beam_width=500, lm_alpha=None, lm_beta=None):
    """Load the deepspeech model from ``model_dir``

    Arguments:
        model_dir (str): path to folder containing the ".pbmm" and optionally ".scorer" files
        beam_width (int, optional): beam width for decoding. Default is 500.
        lm_alpha (float, optional}: alpha parameter of language model. Default is None.
        lm_beta (float, optional): beta parameter of langage model. Default is None.

    Returns:
        [deepspeech model]: the loaded deepspeech model
    """
    from deepspeech import Model

    model, scorer = resolve_models(model_dir)
    logger.debug("Loading model...")
    model = Model(model)
    model.setBeamWidth(beam_width)

    if scorer:
        logger.debug("Loading scorer from files {}".format(scorer))
        model.enableExternalScorer(scorer)

        if lm_alpha and lm_beta:
            model.setScorerAlphaBeta(lm_alpha, lm_beta)

    return model


def transcribe_audio_deepspeech(
    audio_path_or_data, model, raw_audio_data=False, method=None
):
    """Transcribe an audio file or pcm data with the deepspeech model

    Args:
        audio_path_or_data (str or byte string): a path to a wave file or a byte string
            containing pcm data from a wave file. set ``raw_audio_data`` to True if pcm data
            is used.
        model (deepspeech model or str): a deepspeech model object or a path to a folder
            containing the model files (see :meth:`~transcribe.transcribe_main.load_deepspeech_model`)
        raw_audio_data (bool, optional): must be True if ``audio_path_or_data`` is 
            raw pcm data. Defaults to False.
        method (str, optional): which method of the deepspeech model to use for speech-to-text.
            the default is ``model.stt()``. Defaults to None.

    Returns:
        [str]: the transcribed audio file in string format
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
    if method == "extended":
        transcript = metadata_to_string(model.sttWithMetadata(audio, 1).transcripts[0])
    elif method == "json":
        transcript = metadata_json_output(model.sttWithMetadata(audio, 3))
    else:
        transcript = model.stt(audio)
    inference_end = timer() - inference_start
    logger.debug("Inference (transcription) took %0.3fs." % inference_end)

    return transcript


def write_to_file(results, save_file):
    """ Write ``results`` to ``save_file`` in append mode. """
    file_results = open(save_file, "a+")
    logger.info("Writing results to file " + str(save_file))
    file_results.write(results)
    file_results.close()


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
        [tuple]: segments, sample_rate, audio_length. See :meth:`~transcribe.webrtcvad_utils.vad_segment_generator`.
    """
    if desired_sample_rate:
        assert desired_sample_rate in (
            8000,
            16000,
            32000,
            48000,
        ), "The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz."

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


def process_segments(segments, ds_model, audio_length="unknown"):
    """Transcribe a list of byte strings containing pcm data

    Args:
        segments (list): list of byte strings containing pcm data (generated by :meth:`~transcribe.transcribe_main.chunk_by_speech`)
        ds_model (deepspeech model): a deepspeech model object or a path to a folder
            containing the model files (see :meth:`~transcribe.transcribe_main.load_deepspeech_model`).
        audio_length (str, optional): the length of the audio file if known (used for logging statements)
            Default is "unknown".

    Returns:
        [str]: the combined transcript of all the items in `segments`.
    """
    if isinstance(ds_model, str):
        ds_model = load_deepspeech_model(ds_model)

    full_transcript = ""
    start_time = timer()

    # `segments` is a generator so total length is not known
    # also, this means that the audio file is split on voice activity as needed
    # (the entire file is not split and stored in memory at once)
    for i, segment in tqdm(enumerate(segments), desc="Processing Segments"):
        # Run deepspeech on each chunk which completed VAD
        audio = np.frombuffer(segment, dtype=np.int16)
        transcript = transcribe_audio_deepspeech(segment, ds_model, raw_audio_data=True)
        logging.debug("Chunk Transcript: %s" % transcript)

        full_transcript += transcript + " "

    total_time = timer() - start_time
    logger.info(
        "It took "
        + str(total_time)
        + " to transcribe an audio file with duration "
        + str(audio_length)
        + "."
    )
    logger.info("The above time includes time spent determining voice activity.")

    full_transcript = segment_sentences(full_transcript)

    return full_transcript


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

    make_dir_if_not_exist(output_path)

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
    """Runs transcription on every chunk (audio file) in a directory"""
    chunks = os.listdir(chunk_dir)
    chunks.sort()
    full_transcript = ""

    for chunk in tqdm(chunks, desc="Processing Chunks"):
        if chunk.endswith(".wav"):
            chunk_path = Path(chunk_dir) / chunk
            if method == "deepspeech":
                assert model_dir is not None
                ds_model = load_deepspeech_model(model_dir)
                transcript = transcribe_audio_deepspeech(chunk_path, ds_model)
            else:
                transcript = transcribe_audio(chunk_path, method)
            full_transcript += transcript + " "

    return full_transcript


def caption_file_to_string(transcript_path, remove_speakers=False):
    """
    Converts a .srt or .vtt file saved at ``transcript_path`` to a python string. 
    Optionally removes speaker entries by removing everything before ": " in each subtitle cell.
    """
    assert transcript_path.is_file()
    if transcript_path.suffix == ".srt":
        subtitles = webvtt.from_srt(transcript_path)
    else:
        subtitles = webvtt.read(transcript_path)

    transcript = ""
    for subtitle in subtitles:
        content = subtitle.text.replace("\n", " ")  # replace newlines with space
        if remove_speakers:
            content = content.split(": ", 1)[-1]  # remove everything before ": "
        transcript += content + " "  # add space after each subtitle block in srt file
    return transcript


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

# print(segment_sentences("the following content is provided under a creative commons license your support will help him it open course where continued to offer high quality educational resources for free to make a donation or view additional materials from hundreds of mit courses visit it open course were at octavie's begin sosimenes before this artistic will be recorded fridays again in future lectures if you don't want to have the back of your head show up just don't sit in this this friend area here so first of all low what a crowd you guys were finally and twenty six one hundred six your bull one made it big huh so good afternoon and welcome to the very first class of six or belindas six hundred the semester so my name is annabel for senna langbein a lecture and the easiest apartment and i'll be giving some of the lectures for to day along with later on in the in the term professor crimson who sitting right down there we'll be giving some of the lectures well okay so to day are going to go over some just basic administering level what computer do just to make sure we're all in the same page and then we're going to die right into a pythonic rental a little bit about mathematical operations you can do with pity and then we're going to talk about pineries and types so as i mentioned in my introductory email all the slides in code that all talk about during electrolier lecture so i highly encourage you to download them and to have them open we're going to go through some class exercises which will be available theatre a friend it is true in this is a really fast pace course and we ramp up really quickly so we do want to position you to succeed in this course so as i was writing this estranging about when i was first starting to program now what help me get through my first merry first programming course and this is really a like a good list so the first thing was i just read the piece seasons they came out made sure that you know the terminology kind of just sunk in and then during lectures you know if the lecture was talking about something that i suddenly remembered all i saw that word in the piece i didn't know what it was will he now i know what it is right so just give it a reed he don't need to start it if your new operating i think the caristie so it's like math or reading the more you practice the better you get at it if you're not going to sort programming but watching me right for gramps cassidy no how to program right you say to practice so down become before life farfallo along whatever i type by can tie and i think also one of the big things is if your atargatis your kind of afraid that you're going to break your computer right and you can't really do that just by running anaconda and and typing in some commands so don't be afraid to just type sayson and see what it does worse case you just restarted the computer right so ye ye so that's probably the big thing right there i should probably highlighted it but don't be afraid great so this is pretty much a romp of olive six triple one or six hundred as i've just explained it to their three make things we want you to get out of this course bikers thing is the knowledge of concepts which is pretty much true of any class that i'll take right plastileather something through lectures example test how much you know this is a class and programming breathe other thing we need you we want you to get out of it is programming skills and lasting and i think this is what makes his plateaus we teach you how to solve problems and we do that through the pieces so that's really of yolara of this course looks like an underlying all of these is just practice so you have to just type some stuff away and coat blot and you'll succeed in this course i think kate so what are the things we're going to learn in this class i feel like but things were now learn this class can be divided into basically three different sections so the first one is related to these first two first items here so it's really about learning how to program saleroom is part of it is figuring out what um what objects to create you'll learn about these later how do you represent knowledge with the destructors that sort of the broad term for that and then as your writing for grannie to programs are just the near sometimes programs jump around they make decisions there some control float to programs so that's what the second line is going to be about the second big part of the score is a little bit more abstract and deals with how do you write good coat good style cobered able so when you write god you want to ride it such that no you're mad company other people will read it other people would use it so as to be readable and understandable by others so to that end unit write code that well organized lodger is to understand it and not only that not only will your code be read by other people but next year maybe you'll take another course and you want to look back at some of the promises you wrote in this class you kind of want to be better weary your coat right if it's a big mess you might not be able to understand or rounders and what you what you were doing so writing read preble coating organizing cups also big part and the last section is going to deal with us the first two are actually part of the programming in introduction to programming a computer science and python and the last one deals aloides with the computer science part in introduction to programming a computer sciences yon storial about once you florodora programs in pitanatae programs and pipe to how do you know that one program is better than the other right how do you know that one program is more efficient than the other how do you know that one albertists better than the other so that's where we're going to talk about in the last partook so that's all for it at the servant minister to part of the course selects let's start by talking and let a high level what is a computer do so fund"))
