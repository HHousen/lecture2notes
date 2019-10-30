# Main process to convert video to notes (end-to-end)
# 1. Extract frames
# 2. Classify slides
# 3. OCR slides

import sys, os


# 1. Extract frames

root_process_folder = "./process/"
input_video_path = sys.argv[1]
extract_every_x_seconds = str(20)
quality = str(5)
output_path = root_process_folder + "frames"
command = 'ffmpeg -i ' + input_video_path + ' -vf "fps=1/' + extract_every_x_seconds + '" -q:v ' + quality + ' ' + output_path + '/img_%03d.jpg'

print("Running Command: " + command)
#os.system(command)


# 2. Classify slides
import os, operator
from fastai.vision import *

models_path = "./Models/slide-classifier/saved-models/"
learn = load_learner(models_path)

frames_dir = root_process_folder + "frames"
frames_sorted_dir = root_process_folder + "frames_sorted"
def model_predict(img_path):
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    model_results = outputs.numpy().tolist()
    model_results_percent = [i * 100 for i in model_results]
    classes = learn.data.classes
    final = dict(zip(classes, model_results_percent))
    return final

frames = os.listdir(frames_dir)
num_frames = len(frames)
for idx, frame in enumerate(frames):
    print("Progress: " + str(idx) + "/" + str(num_frames))
    current_frame_path = os.path.join(frames_dir, frame)
    # run classification
    predictions = model_predict(current_frame_path)
    # get key with maximum value from the returned `predictions` dictionary
    best_guess = max(predictions.items(), key=operator.itemgetter(1))[0]
    print("AI Predicts: " + best_guess)

    classified_image_dir = frames_sorted_dir + '/' + best_guess
    if not os.path.exists(classified_image_dir):
        os.makedirs(classified_image_dir)
    os.system('mv ' + current_frame_path + ' ' + classified_image_dir)

# 3. OCR slides
slides_folder = frames_sorted_dir + "/slide"
save_file = root_process_folder + "ocr.txt"
import OCR.ocr as ocr
results = ocr.all_in_folder(slides_folder)
ocr.write_to_file(results, save_file)