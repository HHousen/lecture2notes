try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract, os

def all_in_folder(path):
    """Perform OCR on every file in folder and return results"""
    results = []
    images = os.listdir(path)
    images.sort()
    for item in images:
        print("> OCR: Processing file " + item)
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path):
            ocr_result = pytesseract.image_to_string(Image.open(current_path))
            results.append(ocr_result)
    print("> OCR: Returning results")
    return results

def write_to_file(results, save_file):
    """Write everything stored in `results` to file at path `save_file`. Used to write results from `all_in_folder()` to `save_file`."""
    file_results = open(save_file, "a+")
    print("> OCR: Writing results to file " + str(save_file))
    for item in results:
        file_results.write(item + "\r\n")
    file_results.close()
    print("> OCR: Results written to " + str(save_file))
