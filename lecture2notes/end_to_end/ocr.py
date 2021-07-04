try:
    from PIL import Image
except ImportError:
    import Image

import logging
import os

import pytesseract
from tqdm import tqdm

logger = logging.getLogger(__name__)


def all_in_folder(path):
    """Perform OCR using ``pytesseract`` on every file in folder and return results"""
    results = []
    images = os.listdir(path)
    images.sort()
    for item in tqdm(images, total=len(images), desc="> OCR: Progress"):
        # logger.info("> OCR: Processing file " + item)
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path):
            ocr_result = pytesseract.image_to_string(Image.open(current_path))
            results.append(ocr_result)
    logger.debug("Returning results")
    return results


def write_to_file(results, save_file):
    """Write everything stored in `results` to file at path `save_file`. Used to write results from `all_in_folder()` to `save_file`."""
    file_results = open(save_file, "a+")
    logger.info("Writing results to file " + str(save_file))
    for item in tqdm(
        results, total=len(results), desc="> OCR: Writing To File Progress"
    ):
        file_results.write(item + "\r\n")
    file_results.close()
    logger.debug("Results written to " + str(save_file))
