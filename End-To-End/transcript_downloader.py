import io
from youtube_api import init_youtube
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path

class TranscriptDownloader:
    def __init__(self, youtube=None):
        if youtube is None:
            self.youtube = init_youtube(oauth=True)
        else:
            self.youtube = youtube

    def get_transcript(self, caption_id, output_path):
        if output_path.suffix == "":
            output_path.with_suffix('.srt')
        elif output_path.suffix != ".srt":
            raise Exception("Only .srt files are supported. You tried to create a " + output_path.suffix + " file.")

        request = self.youtube.captions().download(
            id=caption_id,
            tfmt="srt"
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
        caption_id = self.get_caption_id(video_id)
        output_path = self.get_transcript(caption_id, output_path)
        return output_path

# downloader = TranscriptDownloader()
# output_path = Path("test.srt")
# transcript_path = downloader.download("Vss3nofHpZI", output_path)