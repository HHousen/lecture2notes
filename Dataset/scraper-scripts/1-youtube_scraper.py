import os, sys
from pathlib import Path
import pandas as pd
import argparse
import isodate

sys.path.insert(1, os.path.join(sys.path[0], '../../End-To-End'))
from youtube_api import init_youtube
from transcript_downloader import TranscriptDownloader

parser = argparse.ArgumentParser(description='YouTube Scraper')
parser.add_argument('mode', choices=["video", "channel", "transcript"],
                    help='Get metadata for a video or a certain number of videos from a channel. Transcript mode downloads the transcript for a video_id.')
parser.add_argument('id', type=str, metavar='STR',
                    help='Channel or video id depending on mode')
parser.add_argument('-n', '--num_pages', default=10, type=int, metavar='N',
                    help='Number of pages of videos to scape if mode is `channel`')
parser.add_argument('-t', '--transcript', dest='get_transcript', action='store_true',
                    help='Download transcript for each video scraped.')
parser.add_argument('-l', '--min_length_check', default=None, type=int, metavar='N',
                    help='Minimum video length to be scraped. Only works when `mode` is "channel"')

args = parser.parse_args()

csv_path = Path("../mass-download-list.csv")
# If CSV file exists then load it otherwise create a new dataframe that will be saved once scraping is complete
if csv_path.is_file():
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.DataFrame(columns=["date","provider","video_id","page_link","download_link","title","description","thumbnail_default","thumbnail_medium","thumbnail_high","downloaded"])

def get_youtube_results(youtube, page="", channel=None, video_id=None, parts="snippet"):
    if video_id != None:
        request = youtube.videos().list(
            part=parts,
            id=video_id
        )
    else:
        request = youtube.search().list(
            part=parts,
            channelId=channel,
            maxResults=50,
            order="date",
            pageToken=page,
            type="video",
            videoDefinition="high"
        )
    
    response = request.execute()
    return response


if args.mode == "transcript" or args.get_transcript:
    youtube = init_youtube(oauth=True)
    transcript_dir = Path("../transcripts")
    if not os.path.exists(transcript_dir):
        os.makedirs(transcript_dir)
else:
    youtube = init_youtube(oauth=False)

if args.mode == "video":
    response = get_youtube_results(youtube, video_id=args.id)
    items = response['items']

    for item in items:
        date = item['snippet']['publishedAt']
        video_id = item['id']
        title = item['snippet']['title']
        description = item['snippet']['description'].splitlines()[0]
        thumbnail_default = item['snippet']['thumbnails']['default']['url']
        thumbnail_medium = item['snippet']['thumbnails']['medium']['url']
        thumbnail_high = item['snippet']['thumbnails']['high']['url']
        channel = item['snippet']['channelId']
        df.loc[len(df.index)]=[date,"youtube",video_id,0,0,title,description,thumbnail_default,thumbnail_medium,thumbnail_high,"false"]

        if args.get_transcript:
            output_path = transcript_dir / video_id + ".srt"
            if not output_path.is_file():
                downloader = TranscriptDownloader(youtube)
                downloader.download(video_id, output_path)
    
    df.to_csv(csv_path)
elif args.mode == "channel":
    if args.get_transcript:
        api_cost = 100 * int(args.num_pages) + 50 * int(args.num_pages) * 250 # 100 tokens * num_pages + 50 videos per page * num_pages * 250 tokens per video (50 for caption search and 200 for transcript)
        print("> YouTube Scraper: [WARNING] This operation is expensive. You have specified to download transcripts for " + str(args.num_pages) + " pages from channel " + str(args.id) + ". This will cost 100 api tokens per page and 250 api tokens per video transcript totaling " + api_cost + " tokens.")
    next_page=""
    for i in range(int(args.num_pages)):
        response = get_youtube_results(youtube, next_page, channel=args.id)
        items = response['items']
        next_page=response['nextPageToken']

        for item in items:
            date = item['snippet']['publishedAt']
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            thumbnail_default = item['snippet']['thumbnails']['default']['url']
            thumbnail_medium = item['snippet']['thumbnails']['medium']['url']
            thumbnail_high = item['snippet']['thumbnails']['high']['url']
            channel = item['snippet']['channelId']
            if args.min_length_check:
                response = get_youtube_results(youtube, video_id=video_id, parts="contentDetails")
                video_item = response['items'][0]
                duration = video_item['contentDetails']['duration']
                duration_timedelta = isodate.parse_duration(duration)
                duration_mins = int((duration_timedelta.seconds//60)%60)
                if duration_mins > args.min_length_check:
                    df.loc[len(df.index)]=[date,"youtube",video_id,0,0,title,description,thumbnail_default,thumbnail_medium,thumbnail_high,"false"]
            else:
                df.loc[len(df.index)]=[date,"youtube",video_id,0,0,title,description,thumbnail_default,thumbnail_medium,thumbnail_high,"false"]

            if args.get_transcript:
                output_path = transcript_dir / video_id + ".srt"
                if not output_path.is_file():
                    downloader = TranscriptDownloader(youtube)
                    downloader.download(video_id, output_path)

        df.to_csv(csv_path)
elif args.mode == "transcript":
    output_path = transcript_dir / (args.id + ".srt")
    if not output_path.is_file():
        downloader = TranscriptDownloader(youtube)
        downloader.download(args.id, output_path)
else:
    raise Exception("Invalid `mode` specified.")
