import logging
import os
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from . import inference

logger = logging.getLogger(__name__)

if os.environ.get("DISPLAY", "") == "":
    logger.debug("No display found. Using non-interactive Agg backend")
    mpl.use("Agg")


class Cluster:
    def __init__(
        self,
        algorithm_name="kmeans",
        num_centroids=20,
        preference=None,
        damping=0.5,
        max_iter=200,
    ):
        """Set up cluster object by defining necessary variables and asserting that user provided algorithm is supported"""
        self.vectors = OrderedDict()

        algorithms = ["kmeans", "affinity_propagation"]
        if algorithm_name not in algorithms:
            raise AssertionError

        self.algorithm_name = algorithm_name
        self.centroids = None
        self.algorithm = None
        self.cost = None
        self.labels = None
        self.closest = None
        self.move_list = None
        self.closest_filenames = None
        self.num_centroids = num_centroids
        self.preference = preference
        self.damping = damping
        self.max_iter = max_iter

    def add(self, vector, filename):
        """Adds a filename and its coresponding feature vector to the cluster object"""
        self.vectors[filename] = vector

    def get_vectors(self):
        return self.vectors

    def get_labels(self):
        return self.labels

    def create_algorithm_if_none(self):
        """Creates algorithm if it has not been created (if it equals None) based on algorithm_name set in __init__"""
        if self.algorithm is None:
            if self.algorithm_name == "kmeans":
                self.create_kmeans(self.num_centroids)
            elif self.algorithm_name == "affinity_propagation":
                self.create_affinity_propagation(
                    self.preference, self.damping, self.max_iter
                )

    def predict(self, array):
        """Wrapper function for algorithm.predict. Creates algorithm if it has not been created."""
        self.create_algorithm_if_none()
        return self.algorithm.predict(array)

    def get_vector_array(self):
        """Return a numpy array of the list of vectors stored in self.vectors"""
        vector_list = list(self.vectors.values())
        vector_array = np.stack(vector_list)
        return vector_array

    def create_affinity_propagation(self, preference, damping, max_iter, store=True):
        """Create and fit an affinity propagation cluster"""
        logger.info("Creating and fitting affinity propagation cluster")
        vector_array = self.get_vector_array()

        affinity_propagation = AffinityPropagation(
            preference=preference, damping=damping, max_iter=max_iter
        )
        affinity_propagation.fit(vector_array)

        centroids = affinity_propagation.cluster_centers_
        labels = affinity_propagation.labels_

        if store:
            self.centroids = centroids
            self.algorithm = affinity_propagation
            self.labels = labels

        return affinity_propagation, centroids, labels

    def create_kmeans(self, num_centroids, store=True):
        """Create and fit a kmeans cluster"""
        logger.info("Creating and fitting kmeans cluster")
        vector_array = self.get_vector_array()

        kmeans = KMeans(n_clusters=num_centroids, random_state=0)
        kmeans.fit(vector_array)

        cost = str(kmeans.inertia_)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        if store:
            self.centroids = centroids
            self.algorithm = kmeans
            self.cost = cost
            self.labels = labels

        return kmeans, centroids, cost, labels

    def get_move_list(self):
        """Creates a dictionary of file names and their coresponding centroid numbers"""
        if self.move_list is not None:
            return self.move_list

        self.create_algorithm_if_none()

        move_list = {}
        for idx, filename in enumerate(self.vectors):
            move_list[filename] = self.labels[idx]
        self.move_list = move_list
        return move_list

    def get_num_clusters(self):
        if self.algorithm_name == "affinity_propagation":
            cluster_centers_indices = self.algorithm.cluster_centers_indices_
            n_clusters_ = len(cluster_centers_indices)
            return n_clusters_
        return self.num_centroids

    def get_closest_sample_filenames_to_centroids(self):
        """
        Return the sample indexes that are closest to each centroid.
        Ex: If [0,8] is returned then X[0] (X is training data/vectors) is the closest
        point in X to centroid 0 and X[8] is the closest to centroid 1
        """
        if self.closest_filenames is not None:
            return self.closest_filenames

        vector_array = self.get_vector_array()
        closest, _ = pairwise_distances_argmin_min(self.centroids, vector_array)
        self.closest = closest
        closest_filenames = []
        for centroid_number, sample_index in enumerate(closest):
            # vector = vector_array[sample_index]
            vector_filenames = list(self.vectors.keys())
            filename = vector_filenames[sample_index]
            # filename = list(self.vectors.keys())[list(self.vectors.values()).index(vector.all())] # get key by value in self.vectors
            closest_filenames.append(filename)
        self.closest_filenames = closest_filenames
        return closest_filenames

    def visualize(self, tensorboard_dir):
        """Creates tensorboard projection of cluster for simplified viewing and understanding"""
        logger.info("Visualizing cluster")
        import torch
        from torch.utils.tensorboard import SummaryWriter

        images = []
        for current_image, _ in self.vectors.items():
            img = Image.open(self.slides_dir / current_image)
            images.append(inference.transform_image(img).float())

        feature_vectors = self.get_vector_array()

        writer = SummaryWriter(tensorboard_dir)
        writer.add_embedding(
            feature_vectors, metadata=self.labels, label_img=torch.cat(images, 0)
        )
        writer.close()

    def calculate_best_k(self, max_k=50):
        """
        Implements elbow method to graph the cost (squared error) as a function of the number of centroids (value of k)
        The point at which the graph becomes essentially linear is the optimal value of k.
        Only works if `algorithm` is "kmeans".
        """
        # Elbow method: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        # Other methods: https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
        if self.algorithm_name != "kmeans":
            raise AssertionError
        costs = []
        for i in range(1, max_k):
            kmeans, _, cost, _ = self.create_kmeans(num_centroids=i, store=False)
            costs.append(cost)
            logger.info("Iteration " + str(i) + ": " + cost)

        costs = [int(float(cost)) for cost in costs]

        # plot the cost against K values
        plt.plot(range(1, max_k), costs)
        plt.xlabel("Value of K")
        plt.ylabel("Sqaured Error (Cost)")
        if mpl.backends.backend == "agg":
            plt.savefig("best_k_value.png")
        else:
            plt.show()
