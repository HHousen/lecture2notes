import os, operator
from pathlib import Path
from helpers import make_dir_if_not_exist
from fastai.vision import *

def model_predict(learn, img_path):
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    model_results = outputs.numpy().tolist()
    model_results_percent = [i * 100 for i in model_results]
    classes = learn.data.classes
    final = dict(zip(classes, model_results_percent))
    print("> AI Prediction Engine: Model prediction successful")
    return final

def classify_frames(frames_dir, models_path):
    learn = load_learner(models_path)
    frames_sorted_dir = frames_dir.parents[0] / "frames_sorted"
    print("> AI Prediction Engine: Received inputs:\nmodels_path=" + str(models_path) + "\nframes_dir=" + str(frames_dir))

    frames = os.listdir(frames_dir)
    num_frames = len(frames)
    print("> AI Prediction Engine: Ready to classify " + str(num_frames) + " frames")
    for idx, frame in enumerate(frames):
        print("> AI Prediction Engine: Progress: " + str(idx+1) + "/" + str(num_frames))
        current_frame_path = os.path.join(frames_dir, frame)
        # run classification
        predictions = model_predict(learn, current_frame_path)
        # get key with maximum value from the returned `predictions` dictionary
        best_guess = max(predictions.items(), key=operator.itemgetter(1))[0]
        print(" > AI Prediction Engine: Prediction is " + best_guess)

        classified_image_dir = frames_sorted_dir / best_guess
        make_dir_if_not_exist(classified_image_dir)
        os.system('mv ' + str(current_frame_path) + ' ' + str(classified_image_dir))
    print("> AI Prediction Engine: Returning frames_sorted_dir=" + str(frames_sorted_dir))
    return frames_sorted_dir