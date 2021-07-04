import logging
import os
import shutil

from PIL import Image
from tqdm import tqdm

from ..models.slide_classifier import inference
from ..models.slide_classifier.class_cluster_scikit import (  # pylint: disable=import-error,wrong-import-position
    Cluster,
)
from ..models.slide_classifier.custom_nnmodules import *  # noqa: F403,F401
from .helpers import make_dir_if_not_exist

logger = logging.getLogger(__name__)


class ClusterFilesystem(Cluster):
    """Clusters images from a directory and saves them to disk in folders corresponding to each centroid."""

    def __init__(
        self,
        slides_dir,
        algorithm_name="kmeans",
        num_centroids=20,
        preference=None,
        damping=0.5,
        max_iter=200,
        model_path="model_best.ckpt",
    ):
        self.slides_dir = slides_dir
        self.model = inference.load_model(model_path)
        super().__init__(
            algorithm_name=algorithm_name,
            preference=preference,
            damping=damping,
            max_iter=max_iter,
        )

    def extract_and_add_features(self, copy=True):
        """Extracts features from the images in `slides_dir` and saves feature vectors with super().add()"""
        slides = os.listdir(self.slides_dir)
        num_slides = len(slides)

        logger.info("Extracting features from " + str(num_slides) + " slides")
        for idx, slide in tqdm(
            enumerate(slides),
            total=num_slides,
            desc="> AI Clustering Engine: Feature extraction",
        ):
            current_slide_path = os.path.join(self.slides_dir, slide)
            _, _, _, extracted_features = inference.get_prediction(
                self.model, Image.open(current_slide_path)
            )
            super().add(extracted_features, slide)
        super().create_algorithm_if_none()

    def transfer_to_filesystem(self, copy=True, create_best_samples_folder=True):
        """Uses `move_list` from super() to take all images in directory `slides_dir` and save each cluster to a subfolder in `cluster_dir` (directory in parent of `slides_dir`)"""
        cluster_dir = (
            self.slides_dir.parents[0] / "slide_clusters"
        )  # cluster_dir = up one directory from slides_dir then into "slide_clusters"
        move_list = super().get_move_list()

        best_samples_path = None
        if create_best_samples_folder:
            closest_filenames = super().get_closest_sample_filenames_to_centroids()
            best_samples_path = cluster_dir / "best_samples"
            make_dir_if_not_exist(best_samples_path)
            for filename in closest_filenames:
                slide_path = self.slides_dir / filename
                shutil.copy(str(slide_path), str(best_samples_path))

        for filename in tqdm(
            move_list, desc="> AI Clustering Engine: Move/copy into cluster folders"
        ):
            cluster_number = move_list[filename]
            current_slide_path = os.path.join(self.slides_dir, filename)
            current_cluster_path = cluster_dir / str(cluster_number)
            make_dir_if_not_exist(current_cluster_path)
            if copy:
                shutil.copy(str(current_slide_path), str(current_cluster_path))
            else:
                shutil.move(str(current_slide_path), str(current_cluster_path))
        return cluster_dir, best_samples_path


# def extract_features(slides_dir, copy=True):
#     """Clusters all images in directory `slides_dir` and saves each cluster to a subfolder in `cluster_dir` (directory in parent of `slides_dir`)"""
#     slides = os.listdir(slides_dir)
#     num_slides = len(slides)
#     cluster = Cluster(algorithm_name="affinity_propagation", preference=-8, damping=0.72)

#     print("> AI Clustering Engine: Extracting features from " + str(num_slides) + " slides")
#     for idx, slide in tqdm(enumerate(slides), total=num_slides, desc="> AI Clustering Engine: Feature extraction"):
#         current_slide_path = os.path.join(slides_dir, slide)
#         _, _, _, extracted_features = get_prediction(model, Image.open(current_slide_path))
#         cluster.add(extracted_features, slide)

#     #cluster.calculate_best_k()
#     move_list = cluster.create_move_list()

#     num_clusters = cluster.get_num_clusters()
#     print("> AI Clustering Engine: Predicted Number of Clusters: " + str(num_clusters))

#     return move_list

# def transfer_to_filesystem(slides_dir, move_list, create_best_samples_folder=True):
#     cluster_dir = slides_dir.parents[0] / "slide_clusters" # cluster_dir = up one directory from slides_dir then into "slide_clusters"
#     if create_best_samples_folder:

#     for filename in tqdm(move_list, desc="> AI Clustering Engine: Move/copy into cluster folders"):
#         cluster_number = move_list[filename]
#         current_slide_path = os.path.join(slides_dir, filename)
#         current_cluster_path = cluster_dir / str(cluster_number)
#         make_dir_if_not_exist(current_cluster_path)
#         if copy:
#             shutil.copy(str(current_slide_path), str(current_cluster_path))
#         else:
#             shutil.move(str(current_slide_path), str(current_cluster_path))
#     return cluster_dir
