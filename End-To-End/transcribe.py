import os
import logging
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
from helpers import make_dir_if_not_exist
from transcript_downloader import TranscriptDownloader
import webvtt
import spacy
import numpy as np

logger = logging.getLogger(__name__)

def extract_audio(video_path, output_path):
    """Extracts audio from video at `video_path` and saves it to `output_path`"""
    logger.info("Extracting audio from " + str(video_path) + " and saving to " + str(output_path))
    command = 'ffmpeg -i ' + str(video_path) + ' -f wav -ab 192000 -vn ' + str(output_path)
    os.system(command)
    return output_path

def transcribe_audio(audio_path, method="sphinx"):
    assert method in ["sphinx", "google"]
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

def transcribe_audio_deepspeech(audio_path, model_dir, model='output_graph.pb', beam_width=500, lm="lm.binary", trie="trie", lm_alpha=0.75, lm_beta=1.85):
    import deepspeech
    import scipy.io.wavfile as wav
    from scipy import signal
    
    # load model
    model = os.path.join(model_dir, model)
    lm = os.path.join(model_dir, lm)
    trie = os.path.join(model_dir, trie)
    model = deepspeech.Model(model, beam_width)
    model.enableDecoderWithLM(lm, trie, lm_alpha, lm_beta)
    
    # load audio
    sampling_rate, audio = wav.read(audio_path)

    if audio.ndim == 2:
        logger.warning("[WARNING] Your audio has 2 channels. The second one will automatically be removed. While this will work, it is better to resample to 1 channel before running this function.")
        audio = audio[:, 0]
    elif audio.ndim > 2:
        raise Exception("> Transcriber: Your audio has " + str(audio.ndim) + " dimensions/channels. The maximum is two. Please resample to 1 channel.")

    # resample to 16000
    if sampling_rate != 16000:
        resample_size = int(len(audio) / sampling_rate * 16000)
        resample = signal.resample(audio, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        transcript = model.stt(resample16)
    else:
        transcript = model.stt(audio)
    return transcript

def write_to_file(results, save_file):
    file_results = open(save_file, "a+")
    logger.info("Writing results to file " + str(save_file))
    file_results.write(results + " ")
    file_results.close()

def create_chunks(audio_path, output_path, silence_thresh_offset, min_silence_len):
    logger.info("Loading audio")
    audio = AudioSegment.from_wav(audio_path)
    logger.info("Average loudness of audio track is " + str(audio.dBFS))
    silence_thresh = audio.dBFS - silence_thresh_offset
    logger.info("Silence Threshold of audio track is " + str(silence_thresh))
    logger.info("Minimum silence length for audio track is " + str(min_silence_len) + " ms")
    logger.info("Creating chunks")
    chunks = split_on_silence(
        # Use the loaded audio.
        audio, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = min_silence_len,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        silence_thresh = silence_thresh
    )
    logger.info("Created " + str(len(chunks)) + " chunks")

    make_dir_if_not_exist(output_path)

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Writing Chunks"):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)
        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Export the audio chunk with new bitrate.
        chunk_number = "chunk" + f'{i:05}' + ".wav"
        logger.debug("Exporting " + chunk_number)
        save_path = Path(output_path) / chunk_number
        audio_chunk.export(
            str(save_path.resolve()),
            bitrate = "192k",
            format = "wav"
        )

def process_chunks(chunk_dir, save_file, method="sphinx", model_dir=None):
    """Runs transcription on every chunk (audio file) in a directory"""
    chunks = os.listdir(chunk_dir)
    chunks.sort()
    for chunk in chunks:
        if chunk.endswith(".wav"):
            chunk_path = Path(chunk_dir) / chunk
            if method == "deepspeech":
                assert model_dir is not None
                transcript = transcribe_audio_deepspeech(chunk_path, model_dir)
            else:
                transcript = transcribe_audio(chunk_path, method)
            write_to_file(transcript, save_file)

def caption_file_to_string(transcript_path, remove_speakers=False):
    """
    Converts a .srt or .vtt file saved at `transcript_path` to a python string. 
    Optionally removes speaker entries by removing everything before ": " in each subtitle cell.
    """
    assert transcript_path.is_file()
    if transcript_path.suffix == ".srt":
        subtitles = webvtt.from_srt(transcript_path)
    else:
        subtitles = webvtt.read(transcript_path)

    transcript = ""
    for subtitle in subtitles:
        content = subtitle.text.replace('\n', ' ') # replace newlines with space
        if remove_speakers:
            content = content.split(': ', 1)[-1] # remove everything before ": "
        transcript += (content+" ") # add space after each subtitle block in srt file
    return transcript

def get_youtube_transcript(video_id, output_path, use_youtube_dl=True):
    """Downloads the transcript for `video_id` and saves it to `output_path`"""
    downloader = TranscriptDownloader(ytdl=use_youtube_dl)
    transcript_path = downloader.download(video_id, output_path)
    return transcript_path

def check_transcript(generated_transcript, ground_truth_transcript):
    """Compares `generated_transcript` to `ground_truth_transcript` to check for accuracy using spacy similarity measurement."""
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