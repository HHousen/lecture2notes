import io
import logging
import os
import glob
import subprocess

from googleapiclient.http import MediaIoBaseDownload

from .youtube_api import init_youtube

logger = logging.getLogger(__name__)


class TranscriptDownloader:
    """Download transcripts from YouTube using the YouTube API or ``youtube-dl``."""

    def __init__(self, youtube=None, ytdl=True):
        self.ytdl = ytdl
        if youtube is None and not ytdl:
            self.youtube = init_youtube(oauth=True)
        else:
            self.youtube = youtube

    @staticmethod
    def check_suffix(output_path):
        """
        Gets the file extension from ``output_path`` and verifies it is either ".srt", ".vtt", or it is not present in ``output_path``.
        The default is ".vtt".
        """
        sub_format = output_path.suffix[1:]
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".vtt")
            sub_format = "vtt"
        elif output_path.suffix != ".srt" and output_path.suffix != ".vtt":
            raise Exception(
                "Only .srt and .vtt files are supported. You tried to create a "
                + output_path.suffix
                + " file."
            )
        return output_path, sub_format

    def get_transcript_ytdl(self, video_id, output_path):
        """
        Gets the transcript for ``video_id`` using ``youtube-dl`` and saves it to ``output_path``.
        The extension from ``output_path`` will be the ``--sub-format`` that is passed to the ``youtube-dl`` command.
        """

        def run_command(command_array):
            completed_command = subprocess.run(
                command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output = completed_command.stdout.decode("utf-8")
            errors = completed_command.stderr.decode("utf-8")
            return output, errors

        output_path, sub_format = self.check_suffix(output_path)
        output_path_no_extension = os.path.splitext(output_path)[0]

        command_array = [
            "yt-dlp",
            "--sub-langs",
            "en.*",
            "--sub-format",
            sub_format,
            "--write-sub",
            "--skip-download",
            "-o",
            str(output_path_no_extension),
            video_id,
        ]

        output, errors = run_command(command_array)
        tries = 1

        while (
            "video is unavailable" in errors or "Unable to download webpage" in errors
        ) and tries < 3:
            output, errors = run_command(command_array)
            tries += 1

        if tries == 3:
            logger.warn("YouTube timed out while getting " + video_id)
            return None

        if "WARNING: video doesn't have subtitles" in errors or "There are no subtitles for the requested languages" in output:
            logger.warn(
                video_id
                + " does not contain a subtitle file for the specified language and format."
            )
            return None
        if " " in errors or "ERROR" in errors or "WARNING" in errors:
            logger.info("The youtube-dl command returned the following error message:")
            logger.error(errors)
            return None

        # remove the ".en" that youtube-dl adds
        os.rename(glob.glob(output_path_no_extension + ".en*." + sub_format)[0], output_path)
        return output_path

    def get_transcript_api(self, caption_id, output_path):
        """Downloads a caption track by id directly from the YouTube API.

        Args:
            caption_id (str): the id of the caption track to download
            output_path (str): path to save the captions. file extensions are parsed by :meth:`~lecture2notes.end_to_end.transcript_downloader.check_suffix`

        Returns:
            [str]: the path where the transcript was saved (may not be the same as the ``output_path`` parameter)
        """
        output_path, sub_format = self.check_suffix(output_path)

        request = self.youtube.captions().download(id=caption_id, tfmt=sub_format)
        fh = io.FileIO(output_path, "wb")

        download = MediaIoBaseDownload(fh, request)
        complete = False
        while not complete:
            status, complete = download.next_chunk()

        self.transcript_path = output_path
        return output_path

    def get_caption_id(self, video_id, lang="en"):
        """Gets the caption id with language ``land`` for a video on YouTube with id ``video_id``."""
        request = self.youtube.captions().list(part="snippet", videoId=video_id)
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
        Convenience function to download transcript with one call.
        If ``self.ytdl`` is False, calls :meth:`~lecture2notes.end_to_end.transcript_downloader.TranscriptDownloader.get_caption_id` and passes result to :meth:`~lecture2notes.end_to_end.transcript_downloader.TranscriptDownloader.get_transcript`.
        If ``self.ytdl`` is True, calls :meth:`~lecture2notes.end_to_end.transcript_downloader.TranscriptDownloader.get_transcript_ytdl`.
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
