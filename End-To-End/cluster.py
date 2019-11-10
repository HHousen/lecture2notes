import sys, os, shutil
from tqdm import tqdm
from helpers import make_dir_if_not_exist
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from class_cluster import Cluster
from custom_nnmodules import *
from inference import *

def make_clusters(slides_dir, copy=True):
    slides = os.listdir(slides_dir)
    num_slides = len(slides)
    cluster = Cluster(num_centroids=20)

    print("> AI Clustering Engine: Ready to cluster " + str(num_slides) + " slides")
    for idx, slide in tqdm(enumerate(slides), total=num_slides, desc="> AI Clustering Engine: Feature extraction"):
        current_slide_path = os.path.join(slides_dir, slide)
        _, _, _, extracted_features = get_prediction(Image.open(current_slide_path))
        cluster.add(extracted_features, slide)

    move_list = cluster.get_move_list()
    cluster_dir = slides_dir.parents[0] / "slide_clusters" # cluster_dir = up one directory from slides_dir then into "slide_clusters"
    for filename in tqdm(move_list, desc="> AI Clustering Engine: Move into cluster folders"):
        cluster_number = move_list[filename]
        current_slide_path = os.path.join(slides_dir, filename)
        current_cluster_path = cluster_dir / str(cluster_number)
        make_dir_if_not_exist(current_cluster_path)
        if copy:
            shutil.copy(str(current_slide_path), str(current_cluster_path))
        else:
            shutil.move(str(current_slide_path), str(current_cluster_path))
    
    return cluster_dir