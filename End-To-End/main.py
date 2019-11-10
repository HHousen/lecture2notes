# Main process to convert video to notes (end-to-end)
# 1. Extract frames
# 2. Classify slides
# 3. OCR slides

import sys, os
from pathlib import Path
from helpers import *

sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from custom_nnmodules import *

root_process_folder = Path("./process/")
if len(sys.argv) > 2:
    skip_to = int(sys.argv[2])
else:
    skip_to = 0

# 1. Extract frames
if skip_to <= 1: 
    from frames_extractor import extract_frames
    input_video_path = sys.argv[1]
    quality = 5
    output_path = root_process_folder / "frames"
    extract_every_x_seconds = 1
    extract_frames(input_video_path, quality, output_path, extract_every_x_seconds)

# 2. Classify slides
if skip_to <= 2:
    from slide_classifier import classify_frames
    frames_dir = root_process_folder / "frames"
    frames_sorted_dir = classify_frames(frames_dir)

# 3. Cluster slides
if skip_to <= 3: 
    if skip_to   >= 3: # if step 2 (classify slides) was skipped
        frames_sorted_dir = root_process_folder / "frames_sorted"
    slides_dir = frames_sorted_dir / "slide"
    from cluster import make_clusters
    cluster_dir = make_clusters(slides_dir)

# 4. OCR slides
if skip_to <= 4: 
    if skip_to >= 4: # if step 3 (cluster slides) was skipped
        frames_sorted_dir = root_process_folder / "frames_sorted"
    import ocr
    slides_folder = frames_sorted_dir / "slide"
    save_file = root_process_folder / "ocr.txt"
    results = ocr.all_in_folder(slides_folder)
    ocr.write_to_file(results, save_file)