import os
import sys
import pandas as pd

method = sys.argv[1]
csv_path = "../slides-dataset.csv"
output_dir="../slides/pdfs"

if method == "csv":
    # python slides_downloader.py csv
    df = pd.read_csv(csv_path, index_col=0)
    for index, row in df.iterrows():
        if row['downloaded'] == False:
            link = row['pdf_link']
            os.system('wget ' + link + ' -P ' + output_dir)
            row['downloaded'] = True # NOT WORKING
else:
    # python slides_downloader.py https://ocw.mit.edu/courses/history/21h-343j-making-books-the-renaissance-and-today-spring-2016/lecture-slides/MIT21H_343JS16_Print.pdf
    os.system('wget ' + method + ' -P ' + output_dir)

df.to_csv(csv_path)