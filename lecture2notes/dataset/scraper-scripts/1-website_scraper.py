import hashlib
import sys
import urllib.request as urllib
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

date = sys.argv[1]
page_link = sys.argv[2]
video_download_link = sys.argv[3]


if len(sys.argv) > 4:  # if sys.argv[4] is supplied
    description = sys.argv[4]
else:
    description = 0

csv_path = Path("../videos-dataset.csv")

# If CSV file exists then load it otherwise create a new dataframe that will be saved once scraping is complete
if csv_path.is_file():
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.DataFrame(
        columns=[
            "date",
            "provider",
            "video_id",
            "download_link",
            "title",
            "description",
            "thumbnail_default",
            "thumbnail_medium",
            "thumbnail_high",
            "downloaded",
        ]
    )
datetime_object = datetime.strptime(date, "%m-%d-%Y")
date = datetime_object.isoformat("T") + "Z"

if page_link.lower().startswith("http"):
    soup = BeautifulSoup(urllib.urlopen(page_link), features="html.parser")
else:
    raise ValueError from None

page_title = soup.title.string

video_id = hashlib.sha1(page_title.encode("UTF-8")).hexdigest()[:11]
print("Video ID: " + video_id)

df.loc[len(df.index)] = [
    date,
    "website",
    video_id,
    page_link,
    video_download_link,
    page_title,
    description,
    0,
    0,
    0,
    "false",
]
df.to_csv(csv_path)
