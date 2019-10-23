import os, operator
from fastai.vision import *

videos_dir = '../videos'
sorted_videos_list = []

models_path = "../../Models/slide-classifier/saved-models/"
learn = load_learner(models_path)

def model_predict(img_path):
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    model_results = outputs.numpy().tolist()
    model_results_percent = [i * 100 for i in model_results]
    classes = learn.data.classes
    final = dict(zip(classes, model_results_percent))
    return final

for item in os.listdir(videos_dir):
    current_dir = os.path.join(videos_dir, item)
    frames_dir = current_dir + "/frames"
    frames_sorted_dir = current_dir + "/frames_sorted"
    if os.path.isdir(current_dir) and os.path.exists(frames_dir) and not os.path.exists(frames_sorted_dir):
        print("Video Folder " + item + " with Frames Directory Found!")
        sorted_videos_list.append(item)
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

print("The Following Videos Need Manual Sorting:\n" + str(sorted_videos_list))
