import os
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
from helpers import make_dir_if_not_exist
from transcript_downloader import TranscriptDownloader
import srt
import spacy

def extract_audio(video_path, output_path):
    print("> Transcriber: Extracting audio from " + str(video_path) + " and saving to " + str(output_path))
    command = 'ffmpeg -i ' + str(video_path) + ' -f wav -ab 192000 -vn ' + str(output_path)
    os.system(command)
    return output_path

def transcribe_audio(audio_path, method="sphinx"):
    assert method in ["sphinx", "google"]
    print("> Transcriber: Initializing speech_recognition library")
    r = sr.Recognizer()
    with sr.AudioFile(str(audio_path)) as source:
        audio = r.record(source)

    try:
        print("> Transcriber: Transcribing file at " + str(audio_path))
        if method == "sphinx":
            transcript = r.recognize_sphinx(audio)
        elif method == "google":
            transcript = r.recognize_google(audio)
        else:
            print("> Transcriber: Incorrect method to transcribe audio")
            return -1
        return transcript
    except sr.UnknownValueError:
        print("> Transcriber: Could not understand audio")
    except sr.RequestError as e:
        print("> Transcriber: Error; {0}".format(e))

def write_to_file(results, save_file):
    file_results = open(save_file, "a+")
    print("> Transcriber: Writing results to file " + str(save_file))
    file_results.write(results + " ")
    file_results.close()

def create_chunks(audio_path, output_path, silence_thresh_offset, min_silence_len):
    print("> Transcriber: Loading audio")
    audio = AudioSegment.from_wav(audio_path)
    print("> Transcriber: Average loudness of audio track is " + str(audio.dBFS))
    silence_thresh = audio.dBFS - silence_thresh_offset
    print("> Transcriber: Silence Threshold of audio track is " + str(silence_thresh))
    print("> Transcriber: Minimum silence length for audio track is " + str(min_silence_len) + " ms")
    print("> Transcriber: Creating chunks")
    chunks = split_on_silence(
        # Use the loaded audio.
        audio, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = min_silence_len,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        silence_thresh = silence_thresh
    )
    print("> Transcriber: Created " + str(len(chunks)) + " chunks")

    make_dir_if_not_exist(output_path)

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Writing Chunks"):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)
        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Export the audio chunk with new bitrate.
        chunk_number = "chunk" + f'{i:05}' + ".wav"
        print("Exporting " + chunk_number)
        save_path = Path(output_path) / chunk_number
        audio_chunk.export(
            str(save_path.resolve()),
            bitrate = "192k",
            format = "wav"
        )

def process_chunks(chunk_dir, save_file, method="sphinx"):
    chunks = os.listdir(chunk_dir)
    chunks.sort()
    for chunk in chunks:
        if chunk.endswith(".wav"):
            chunk_path = Path(chunk_dir) / chunk
            transcript = transcribe_audio(chunk_path, method)
            write_to_file(transcript, save_file)

def srt_to_string(transcript_path, remove_speakers=False):
    assert transcript_path.is_file()
    transcript_srt_string = open(transcript_path, "r")
    subtitle_generator = srt.parse(transcript_srt_string)
    subtitles = list(subtitle_generator)
    transcript = ""
    for subtitle in subtitles:
        content = subtitle.content.replace('\n',' ') # replace newlines with space
        if remove_speakers:
            content = content.split(': ', 1)[-1] # remove everything before ": "
        transcript += (content+" ") # add space after each subtitle block in srt file
    return transcript

def get_youtube_transcript(video_id, output_path):
    downloader = TranscriptDownloader()
    transcript_path = downloader.download(video_id, output_path)
    return transcript_path

def check_transcript(generated_transcript, ground_truth_transcript):
    """Compares `generated_transcript` to `ground_truth_transcript` to check for accuracy."""
    nlp = spacy.load("en_vectors_web_lg")
    print("> Transcriber: Loaded Spacy `en_vectors_web_lg`")
    gen_doc = nlp(generated_transcript)
    print("> Transcriber: NLP done on generated_transcript")
    real_doc = nlp(ground_truth_transcript)
    print("> Transcriber: NLP done on ground_truth_transcript")
    similarity = gen_doc.similarity(real_doc)
    print("> Transcriber: Similarity Computed: " + str(similarity))
    return similarity

# extract_audio("nykOeWgQcHM.mp4", "process/audio.wav")
# create_chunks("process/audio-short.wav", "process/chunks", 5, 2000)
# process_chunks("process/chunks", "process/output.txt")

# transcript = srt_to_string(Path("test.srt"))
# print(transcript)

# generated_transcript = open(Path("process/audio.txt"), "r").read()
# ground_truth_transcript = transcript
# similarity = check_transcript(generated_transcript, ground_truth_transcript)
# print(similarity)