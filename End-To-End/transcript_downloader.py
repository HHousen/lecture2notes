import io
import os
from youtube_api import init_youtube
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path

class TranscriptDownloader:
    def __init__(self, youtube=None, ytdl=True):
        self.ytdl = ytdl
        if youtube is None and not ytdl:
            self.youtube = init_youtube(oauth=True)
        else:
            self.youtube = youtube

    def check_suffix(self, output_path):
        sub_format = output_path.suffix[1:]
        if output_path.suffix == "":
            output_path = output_path.with_suffix('.vtt')
            sub_format = "vtt"
        elif output_path.suffix != ".srt" and output_path.suffix != ".vtt":
            raise Exception("Only .srt and .vtt files are supported. You tried to create a " + output_path.suffix + " file.")
        return output_path, sub_format

    def get_transcript_ytdl(self, video_id, output_path):
        output_path, sub_format = self.check_suffix(output_path)
        output_path_no_extension = os.path.splitext(output_path)[0]

        os.system('youtube-dl --sub-lang en --sub-format ' + sub_format + ' --write-sub --skip-download -o ' + str(output_path_no_extension) + ' ' + video_id)

        # remove the ".en" that youtube-dl adds
        os.rename((output_path_no_extension + '.en.' + sub_format), output_path)
        return output_path

    def get_transcript_api(self, caption_id, output_path):
        output_path, sub_format = self.check_suffix(output_path)

        request = self.youtube.captions().download(
            id=caption_id,
            tfmt=sub_format
        )
        fh = io.FileIO(output_path, "wb")

        download = MediaIoBaseDownload(fh, request)
        complete = False
        while not complete:
            status, complete = download.next_chunk()
        
        self.transcript_path = output_path
        return output_path

    def get_caption_id(self, video_id, lang="en"):
        
        request = self.youtube.captions().list(
            part="snippet",
            videoId=video_id
        )
        response = request.execute()
        for caption in response["items"]:
            trackKind = caption["snippet"]["trackKind"]
            language = caption["snippet"]["language"]
            caption_id = caption["id"]
            if trackKind == "standard" and language == lang:
                self.caption_id = caption_id
                return caption_id
        raise Exception("No caption track exists for language '" + lang + "'.")
    
    def download(self, video_id, output_path):
        """
        Convenience function to download transcript with one call
        Calls `get_caption_id` and passes result to `get_transcript`
        """
        if self.ytdl:
            output_path = self.get_transcript_ytdl(video_id, output_path)
        else:
            caption_id = self.get_caption_id(video_id)
            output_path = self.get_transcript_api(caption_id, output_path)
        return output_path

# downloader = TranscriptDownloader()
# output_path = Path("test.srt")
# transcript_path = downloader.download("Vss3nofHpZI", output_path)