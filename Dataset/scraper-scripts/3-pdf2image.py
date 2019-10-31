import sys, os
from pathlib import Path

pdf_dir = Path("../slides/pdfs/")
output_path = Path("../slides/images/")

for item in os.listdir(pdf_dir):
    name = item[:-4]
    item_path = os.path.join(pdf_dir, item)
    output_dir_final = output_path / name

    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final)
    
    os.system('pdftoppm -png ' + str(item_path) + ' ' + str(output_dir_final) + '/' + name)
