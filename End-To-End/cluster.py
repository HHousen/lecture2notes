import sys, os, shutil
from tqdm import tqdm
from helpers import make_dir_if_not_exist
# Hack to import modules from different parent directory
sys.path.insert(1, os.path.join(sys.path[0], '../Models/slide-classifier'))
from class_cluster_scikit import Cluster
from custom_nnmodules import *
from inference import *

def make_clusters(slides_dir, copy=True):
    """Clusters all images in directory `slides_dir` and saves each cluster to a subfolder in `cluster_dir` (directory in parent of `slides_dir`)"""
    slides = os.listdir(slides_dir)
    num_slides = len(slides)
    cluster = Cluster(algorithm_name="affinity_propagation", preference=-6, damping=0.7)

    print("> AI Clustering Engine: Ready to cluster " + str(num_slides) + " slides")
    for idx, slide in tqdm(enumerate(slides), total=num_slides, desc="> AI Clustering Engine: Feature extraction"):
        current_slide_path = os.path.join(slides_dir, slide)
        _, _, _, extracted_features = get_prediction(Image.open(current_slide_path))
        cluster.add(extracted_features, slide)

    #cluster.calculate_best_k()
    move_list = cluster.create_move_list()

    num_clusters = cluster.get_num_clusters()
    print("Predicted Number of Clusters: " + str(num_clusters))

    cluster_dir = slides_dir.parents[0] / "slide_clusters" # cluster_dir = up one directory from slides_dir then into "slide_clusters"
    for filename in tqdm(move_list, desc="> AI Clustering Engine: Move/copy into cluster folders"):
        cluster_number = move_list[filename]
        current_slide_path = os.path.join(slides_dir, filename)
        current_cluster_path = cluster_dir / str(cluster_number)
        make_dir_if_not_exist(current_cluster_path)
        if copy:
            shutil.copy(str(current_slide_path), str(current_cluster_path))
        else:
            shutil.move(str(current_slide_path), str(current_cluster_path))
    
    return cluster_dir