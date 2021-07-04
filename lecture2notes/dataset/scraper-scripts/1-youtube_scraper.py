import argparse
import json
import os
import sys
from pathlib import Path

import isodate
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], "../../End-To-End"))
from transcript_downloader import TranscriptDownloader  # noqa: E402
from youtube_api import init_youtube  # noqa: E402

PARSER = argparse.ArgumentParser(description="YouTube Scraper")
PARSER.add_argument(
    "mode",
    choices=["video", "channel", "transcript"],
    help="Get metadata for a video or a certain number of videos from a channel. Transcript mode downloads the transcript for a video_id.",
)
PARSER.add_argument(
    "id", type=str, metavar="STR", help="Channel or video id depending on mode"
)
PARSER.add_argument(
    "-n",
    "--num_pages",
    default=10,
    type=int,
    metavar="N",
    help="Number of pages of videos to scape if mode is `channel`. 50 videos per page.",
)
PARSER.add_argument(
    "-t",
    "--transcript",
    dest="get_transcript",
    action="store_true",
    help="Download transcript for each video scraped.",
)
PARSER.add_argument(
    "--transcript-use-yt-api",
    dest="transcript_use_api",
    action="store_true",
    help="Use the YouTube API instead of youtube-dl to download transcripts. `--transcript` must be specified for this option to take effect.",
)
PARSER.add_argument(
    "-l",
    "--min_length_check",
    default=None,
    type=int,
    metavar="N",
    help='Minimum video length in minutes to be scraped. Only works when `mode` is "channel"',
)
PARSER.add_argument(
    "-f",
    "--file",
    metavar="PATH",
    default="../videos-dataset.csv",
    help="File to add scraped results to.",
)
PARSER.add_argument(
    "-o",
    "--search_order",
    type=str,
    default="date",
    help="The order to list videos from a channel when `mode` is 'channel'. Acceptable values are in the YouTube API Documentation: https://developers.google.com/youtube/v3/docs/search/list",
)
PARSER.add_argument(
    "-p",
    "--params",
    type=str,
    default=None,
    help="A string dictionary of parameters to pass to the call to the YouTube API. If mode=video then the `videos.list` api is used. If mode=channel then the `search.list` api is used.",
)
ARGS = PARSER.parse_args()

csv_path = Path(ARGS.file)
# If CSV file exists then load it otherwise create a new dataframe that will be saved once scraping is complete
if csv_path.is_file():
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.DataFrame(
        columns=[
            "date",
            "provider",
            "video_id",
            "page_link",
            "download_link",
            "title",
            "description",
            "thumbnail_default",
            "thumbnail_medium",
            "thumbnail_high",
            "downloaded",
        ]
    )


def get_youtube_results(
    youtube, page="", channel=None, video_id=None, parts="snippet", order="date"
):
    """Retrieves results from YouTube Data API"""
    if video_id is not None:
        request = youtube.videos().list(part=parts, id=video_id)
    else:
        params = {
            "part": parts,
            "channelId": channel,
            "maxResults": 50,
            "pageToken": page,
            "type": "video",
            "videoDefinition": "high",
            "order": order,
        }

        if ARGS.params is not None:
            extra_params = json.loads(ARGS.params)
            params = {**params, **extra_params}

        request = youtube.search().list(**params)

    response = request.execute()
    return response


if ARGS.mode == "transcript" or ARGS.get_transcript:
    transcript_dir = Path("../transcripts")
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)

if ARGS.transcript_use_api:
    youtube = init_youtube(oauth=True)
else:
    youtube = init_youtube(oauth=False)


if ARGS.mode == "video":
    response = get_youtube_results(youtube, video_id=ARGS.id)
    items = response["items"]

    for item in items:
        date = item["snippet"]["publishedAt"]
        video_id = item["id"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"].splitlines()[0]
        thumbnail_default = item["snippet"]["thumbnails"]["default"]["url"]
        thumbnail_medium = item["snippet"]["thumbnails"]["medium"]["url"]
        thumbnail_high = item["snippet"]["thumbnails"]["high"]["url"]
        channel = item["snippet"]["channelId"]
        df.loc[len(df.index)] = [
            date,
            "youtube",
            video_id,
            0,
            0,
            title,
            description,
            thumbnail_default,
            thumbnail_medium,
            thumbnail_high,
            False,
        ]

        if ARGS.get_transcript:
            output_path = transcript_dir / (video_id + ".vtt")
            if not output_path.is_file():
                downloader = TranscriptDownloader(
                    youtube, ytdl=(not ARGS.transcript_use_api)
                )
                downloader.download(video_id, output_path)

    df.to_csv(csv_path)
elif ARGS.mode == "channel":
    if ARGS.get_transcript and ARGS.transcript_use_api:
        api_cost = (
            100 * int(ARGS.num_pages) + 50 * int(ARGS.num_pages) * 250
        )  # 100 tokens * num_pages + 50 videos per page * num_pages * 250 tokens per video (50 for caption search and 200 for transcript)
        print(
            "> YouTube Scraper: [WARNING] This operation is expensive. You have specified to download transcripts for "
            + str(ARGS.num_pages)
            + " pages from channel "
            + str(ARGS.id)
            + ". This will cost 100 api tokens per page and 250 api tokens per video transcript totaling "
            + api_cost
            + " tokens."
        )
        input(
            "> YouTube Scraper: Press enter to confirm execution or Ctrl+C to cancel."
        )
    next_page = ""
    for i in range(int(ARGS.num_pages)):
        response = get_youtube_results(
            youtube, next_page, channel=ARGS.id, order=ARGS.search_order
        )
        items = response["items"]
        if "nextPageToken" in response:
            next_page = response["nextPageToken"]

        for item in items:
            date = item["snippet"]["publishedAt"]
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            thumbnail_default = item["snippet"]["thumbnails"]["default"]["url"]
            thumbnail_medium = item["snippet"]["thumbnails"]["medium"]["url"]
            thumbnail_high = item["snippet"]["thumbnails"]["high"]["url"]
            channel = item["snippet"]["channelId"]
            if ARGS.min_length_check:
                response = get_youtube_results(
                    youtube, video_id=video_id, parts="contentDetails"
                )
                video_item = response["items"][0]
                duration = video_item["contentDetails"]["duration"]
                duration_timedelta = isodate.parse_duration(duration)
                duration_mins = int((duration_timedelta.seconds // 60) % 60)
                if duration_mins > ARGS.min_length_check:
                    df.loc[len(df.index)] = [
                        date,
                        "youtube",
                        video_id,
                        0,
                        0,
                        title,
                        description,
                        thumbnail_default,
                        thumbnail_medium,
                        thumbnail_high,
                        False,
                    ]
            else:
                df.loc[len(df.index)] = [
                    date,
                    "youtube",
                    video_id,
                    0,
                    0,
                    title,
                    description,
                    thumbnail_default,
                    thumbnail_medium,
                    thumbnail_high,
                    False,
                ]

            if ARGS.get_transcript:
                output_path = transcript_dir / (video_id + ".vtt")
                if not output_path.is_file():
                    downloader = TranscriptDownloader(
                        youtube, ytdl=(not ARGS.transcript_use_api)
                    )
                    downloader.download(video_id, output_path)

        df.to_csv(csv_path)
elif ARGS.mode == "transcript":
    output_path = transcript_dir / (ARGS.id + ".vtt")
    if not output_path.is_file():
        downloader = TranscriptDownloader(youtube, ytdl=(not ARGS.transcript_use_api))
        downloader.download(ARGS.id, output_path)
else:
    raise Exception("Invalid `mode` specified.")
