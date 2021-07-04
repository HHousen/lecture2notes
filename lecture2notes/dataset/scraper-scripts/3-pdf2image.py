import os
from pathlib import Path

from tqdm import tqdm

pdf_dir = Path("../slides/pdfs/")
output_path = Path("../slides/images/")

pdfs = os.listdir(pdf_dir)
for item in tqdm(pdfs, total=len(pdfs), desc="Converting PDFs to Images"):
    name = item[:-4]
    item_path = os.path.join(pdf_dir, item)
    output_dir_final = output_path / name

    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final)

    os.system(
        "pdftoppm -png " + str(item_path) + " " + str(output_dir_final) + "/" + name
    )
