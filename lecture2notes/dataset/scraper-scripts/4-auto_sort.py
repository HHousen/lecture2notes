import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from termcolor import colored

videos_dir = Path("../videos")
sorted_videos_list = []
csv_path = Path("../to-be-sorted.csv")

# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], "../../models/slide_classifier"))
from custom_nnmodules import *  # noqa: F403,E402,F401
from inference import get_prediction, load_model  # noqa: E402

MODEL = load_model()

if csv_path.is_file():
    df = pd.read_csv(csv_path, index_col=0)
else:
    df = pd.DataFrame(columns=["video_id", "frame", "best_guess", "probability"])

for item in os.listdir(videos_dir):
    current_dir = videos_dir / item
    frames_dir = current_dir / "frames"
    frames_sorted_dir = current_dir / "frames_sorted"
    if (
        os.path.isdir(current_dir)
        and os.path.exists(frames_dir)
        and not os.path.exists(frames_sorted_dir)
    ):
        print("Video Folder " + item + " with Frames Directory Found!")
        num_incorrect = 0
        sorted_videos_list.append(item)
        frames = os.listdir(frames_dir)
        num_frames = len(frames)
        for idx, frame in enumerate(frames):
            print("Progress: " + str(idx + 1) + "/" + str(num_frames))
            current_frame_path = os.path.join(frames_dir, frame)
            # run classification
            best_guess, best_guess_idx, probs, _ = get_prediction(
                MODEL, Image.open(current_frame_path), extract_features=False
            )
            prob_max_correct = list(probs.values())[best_guess_idx]
            print("AI Predicts: " + best_guess)
            print("Probabilities: " + str(probs))
            if prob_max_correct < 0.60:
                num_incorrect = num_incorrect + 1
                print(colored(str(prob_max_correct) + " Likely Incorrect", "red"))
                df.loc[len(df.index)] = [item, frame, best_guess, prob_max_correct]
            else:
                print(colored(str(prob_max_correct) + " Likely Correct", "green"))

            classified_image_dir = frames_sorted_dir / best_guess
            if not os.path.exists(classified_image_dir):
                os.makedirs(classified_image_dir)
            shutil.move(str(current_frame_path), str(classified_image_dir))
        if num_incorrect == 0:
            # df.loc[len(df.index)]=[item,frame,best_guess,prob_max_correct]
            percent_wrong = 0
        else:
            percent_wrong = (num_incorrect / num_frames) * 100
        print(
            "> AI Prediction Engine: Percent frames classified incorrectly: "
            + str(percent_wrong)
        )
        df.to_csv(csv_path)
print("The Following Videos Need Manual Sorting:\n" + str(sorted_videos_list))
