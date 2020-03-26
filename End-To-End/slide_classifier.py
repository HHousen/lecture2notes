import os, sys, shutil
import loggin
from helpers import make_dir_if_not_exist
from termcolor import colored
from PIL import Image

logger = logging.getLogger(__name__)

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from custom_nnmodules import * #pylint: disable=import-error,wrong-import-position,wildcard-import
import inference #pylint: disable=wrong-import-position

def classify_frames(frames_dir, do_move=True, incorrect_treshold=0.60):
    certainties = []
    frames_sorted_dir = frames_dir.parents[0] / "frames_sorted"
    logger.debug("Received inputs:\nframes_dir=" + str(frames_dir))

    frames = os.listdir(frames_dir)
    num_frames = len(frames)
    num_incorrect = 0
    logger.info("Ready to classify " + str(num_frames) + " frames")
    for idx, frame in enumerate(frames):
        logger.info("Progress: " + str(idx+1) + "/" + str(num_frames))
        current_frame_path = os.path.join(frames_dir, frame)
        # run classification
        best_guess, best_guess_idx, probs, _ = inference.get_prediction(Image.open(current_frame_path), extract_features=False) #pylint: disable=no-member
        prob_max_correct = list(probs.values())[best_guess_idx]
        certainties.append(prob_max_correct)
        logger.info("Prediction is " + best_guess)
        logger.debug("Probabilities are " + str(probs))
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
    logger.info("Percent frames classified incorrectly: " + str(percent_wrong))
    logger.debug("Returning frames_sorted_dir=" + str(frames_sorted_dir))
    return frames_sorted_dir, certainties, percent_wrong