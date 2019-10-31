# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os, sys
import googleapiclient.discovery
from pathlib import Path

import pandas as pd

def main():
    csv_path = Path("../videos-dataset.csv")
    df = pd.read_csv(csv_path, index_col=0)
    # df = pd.DataFrame(columns=["date","video_id","title","description","thumbnail_default","thumbnail_medium","thumbnail_high","downloaded"])
    
    if sys.argv[1] == "channel":
        # python youtube_scraper.py channel UCEBb1b_L6zDS3xTUrIALZOw
        next_page=""
        for i in range(50):
            if sys.argv[2]:
                response = get_youtube_results(next_page, sys.argv[2])
            else:
                response = get_youtube_results(next_page)
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
                if channel != sys.argv[2] or channel != 'UCEBb1b_L6zDS3xTUrIALZOw':
                    print("Video From Wrong Channel\n-------------------\nVideo ID: " + video_id + "\nTitle: " + title)
                df.loc[len(df.index)]=[date,video_id,title,description,thumbnail_default,thumbnail_medium,thumbnail_high,"false"]
            df.to_csv(csv_path)
    else:
        # python youtube_scraper.py video 1Qws70XGSq4
        response = get_youtube_results(video_id=sys.argv[2])
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
            df.loc[len(df.index)]=[date,video_id,title,description,thumbnail_default,thumbnail_medium,thumbnail_high,"false"]
        df.to_csv(csv_path)

def get_youtube_results(page="",channel="UCEBb1b_L6zDS3xTUrIALZOw",video_id="VT2o4KCEbes"):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBEjf7WknjMp6wmzhnpeJIsWDhGk3Uq-MM"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    if video_id != "":
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
    else:
        request = youtube.search().list(
            part="snippet",
            channelId=channel,
            maxResults=50,
            order="date",
            pageToken=page,
            type="video",
            videoDefinition="high"
        )
    
    response = request.execute()
    return response

if __name__ == "__main__":
    main()