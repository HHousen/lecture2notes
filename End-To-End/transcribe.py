import os
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def extract_audio(video_path, output_path):
    print("> Transcriber: Extracting audio from " + str(video_path) + " and saving to " + str(output_path))
    command = 'ffmpeg -i ' + str(video_path) + ' -f wav -ab 192000 -vn ' + str(output_path)
    os.system(command)
    return output_path

def transcribe_audio(audio_path):
    print("> Transcriber: Initializing speech_recognition library")
    r = sr.Recognizer()
    with sr.AudioFile(str(audio_path)) as source:
        audio = r.record(source)

    try:
        print("> Transcriber: Transcribing file at " + str(audio_path))
        transcript = r.recognize_sphinx(audio)
        return transcript
    except sr.UnknownValueError:
        print("> Transcriber: Sphinx could not understand audio")
    except sr.RequestError as e:
        print("> Transcriber: Sphinx error; {0}".format(e))

def write_to_file(results, save_file):
    file_results = open(save_file, "a+")
    print("> Transcriber: Writing results to file " + str(save_file))
    file_results.write(results)
    file_results.close()

def create_chunks(audio_path, output_path):
    print("> Transcriber: Loading audio")
    audio = AudioSegment.from_wav(audio_path)
    print("> Transcriber: Creating chunks")
    chunks = split_on_silence(
        # Use the loaded audio.
        audio, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = 2000,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        silence_thresh = -16
    )
    print("> Transcriber: Created " + len(chunks) + " chunks")
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Writing Chunks"):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)
        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Export the audio chunk with new bitrate.
        print("Exporting chunk{0}.mp3.".format(i))
        save_path = output_path / "chunk{0}.wav".format(i)
        audio_chunk.export(
            save_path,
            bitrate = "192k",
            format = "wav"
        )

def process_chunks(chunk_dir, save_file):
    for chunk in os.listdir(chunk_dir):
        if chunk.endswith(".wav"):
            chunk_path = chunk_dir / chunk
            transcript = transcribe_audio(chunk_path)
            write_to_file(transcript, save_file)

extract_audio("nykOeWgQcHM.mp4", "d55d9dd25523/audio.wav")
create_chunks("d55d9dd25523/audio.wav", "d55d9dd25523/chunks")
# test = transcribe_audio('short.wav')
# write_to_file(test, "./test-del.txt")