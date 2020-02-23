import os, sys, shutil
from helpers import make_dir_if_not_exist
from termcolor import colored
from PIL import Image

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from custom_nnmodules import * #pylint: disable=import-error,wrong-import-position,wildcard-import
import inference #pylint: disable=wrong-import-position

def classify_frames(frames_dir, do_move=True, incorrect_treshold=0.60):
    certainties = []
    frames_sorted_dir = frames_dir.parents[0] / "frames_sorted"
    print("> AI Prediction Engine: Received inputs:\nframes_dir=" + str(frames_dir))

    frames = os.listdir(frames_dir)
    num_frames = len(frames)
    num_incorrect = 0
    print("> AI Prediction Engine: Ready to classify " + str(num_frames) + " frames")
    for idx, frame in enumerate(frames):
        print("> AI Prediction Engine: Progress: " + str(idx+1) + "/" + str(num_frames))
        current_frame_path = os.path.join(frames_dir, frame)
        # run classification
        best_guess, best_guess_idx, probs, _ = inference.get_prediction(Image.open(current_frame_path), extract_features=False) #pylint: disable=no-member
        prob_max_correct = list(probs.values())[best_guess_idx]
        certainties.append(prob_max_correct)
        print("> AI Prediction Engine: Prediction is " + best_guess)
        print("> AI Prediction Engine: Probabilities are " + str(probs))
        if prob_max_correct < incorrect_treshold:
            num_incorrect = num_incorrect + 1
            print(colored(str(prob_max_correct) + " Likely Incorrect", 'red'))
        else:
            print(colored(str(prob_max_correct) + " Likely Correct", 'green'))
        
        if do_move:
            classified_image_dir = frames_sorted_dir / best_guess
            make_dir_if_not_exist(classified_image_dir)
            shutil.move(str(current_frame_path), str(classified_image_dir))
    if num_incorrect == 0:
        percent_wrong = 0
    else:
        percent_wrong = (num_incorrect / num_frames) * 100
    print("> AI Prediction Engine: Percent frames classified incorrectly: " + str(percent_wrong))
    print("> AI Prediction Engine: Returning frames_sorted_dir=" + str(frames_sorted_dir))
    return frames_sorted_dir, certainties, percent_wrong