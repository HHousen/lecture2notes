import logging
import os
import shutil
from pathlib import Path

from PIL import Image

# import statistics
from scipy import spatial
from tqdm import tqdm

from ..models.slide_classifier import inference
from .helpers import make_dir_if_not_exist

logger = logging.getLogger(__name__)


class SegmentCluster:
    """Iterates through frames in order and splits based on large visual differences
    (measured by the cosine difference between the feature vectors from the slide classifier)"""

    def __init__(self, slides_dir, model_path="model_best.ckpt"):
        self.slides_dir = Path(slides_dir)
        self.slides_list = sorted(os.listdir(self.slides_dir))
        self.model = inference.load_model(model_path)
        self.change_indexes = None

    def extract_and_add_features(self, gamma=1.08):
        """Extracts features from the images in ``slides_dir`` and saves feature vectors"""
        num_slides = len(self.slides_list)

        all_features = []
        logger.info("Extracting features from " + str(num_slides) + " slides")
        for idx, slide in tqdm(
            enumerate(self.slides_list),
            total=num_slides,
            desc="> AI Segmenting Engine: Feature extraction",
        ):
            current_slide_path = os.path.join(self.slides_dir, slide)
            _, _, _, extracted_features = inference.get_prediction(
                self.model,
                Image.open(current_slide_path),
            )
            all_features.append(extracted_features)

        # spatial.distance.cosine computes the cosine difference, a larger value means more different
        # and a smaller value means more similar
        similarities = [
            spatial.distance.cosine(all_features[i - 1], all_features[i])
            for i in range(1, len(all_features))
        ]
        mean_similarities = sum(similarities) / len(similarities)
        # std_similarities = statistics.pstdev(similarities)

        # A larger `sim_compare_value` = less segments
        # A larger divisor = more segments
        sim_compare_value = mean_similarities * gamma

        change_indexes = [
            i
            for i in range(0, len(similarities))
            if similarities[i] > sim_compare_value
        ]

        self.change_indexes = change_indexes
        return change_indexes

    def transfer_to_filesystem(self, copy=True, create_best_samples_folder=True):
        """Takes all images in directory ``slides_dir`` and saves each cluster to a subfolder in ``cluster_dir`` (directory in parent of ``slides_dir``)"""
        cluster_dir = (
            self.slides_dir.parents[0] / "slide_clusters"
        )  # cluster_dir = up one directory from slides_dir then into "slide_clusters"

        if create_best_samples_folder:
            best_samples_path = cluster_dir / "best_samples"
            make_dir_if_not_exist(best_samples_path)

        current_folder = 0
        current_cluster_path = cluster_dir / str(current_folder)
        for idx, slide in enumerate(self.slides_list):
            current_slide_path = os.path.join(self.slides_dir, slide)
            if idx in self.change_indexes:
                current_folder += 1
                if create_best_samples_folder:
                    if copy:
                        shutil.copy(str(current_slide_path), str(best_samples_path))
                    else:
                        shutil.move(str(current_slide_path), str(best_samples_path))
            current_cluster_path = cluster_dir / str(current_folder)

            make_dir_if_not_exist(current_cluster_path)

            if copy:
                shutil.copy(str(current_slide_path), str(current_cluster_path))
            else:
                shutil.move(str(current_slide_path), str(current_cluster_path))

        return cluster_dir, best_samples_path


# seg = SegmentCluster("slide")
# seg.extract_and_add_features()
# seg.transfer_to_filesystem()

# TODO
# 1. Matrix of similarity/difference scores to remove the most similar images (in the case that the presenter switches back and forth between two things, thus producing duplicates)
# 2. Test if model features for same slide as screen capture vs camera are the same
# 3. If in the above step the model features are difference then change the dataset so that all screen captures are in the slides class and all cameras are in the presenter_slide class
