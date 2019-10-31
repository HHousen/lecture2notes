try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract, os

def all_in_folder(path):
    results = []
    for item in os.listdir(path):
        print("> OCR: Processing file " + item)
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path):
            ocr_result = pytesseract.image_to_string(Image.open(current_path))
            results.append(ocr_result)
    print("> OCR: Returning results")
    return results

def write_to_file(results, save_file):
    file_results = open(save_file, "a+")
    print("> OCR: Writing results to file " + str(save_file))
    for item in results:
        file_results.write(item + "\r\n")
    file_results.close()
    print("> OCR: Results written to " + str(save_file))
