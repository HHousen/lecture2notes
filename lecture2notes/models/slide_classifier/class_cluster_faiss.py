import os

import faiss
import matplotlib as mpl
import numpy

if os.environ.get("DISPLAY", "") == "":
    print("=> Class Cluster: No display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self, num_centroids=20):
        self.vectors = {}
        self.centroids = None
        self.kmeans = None
        self.cost = None
        self.nearest_centroids = None
        self.num_centroids = num_centroids

    def add(self, vector, filename):
        self.vectors[filename] = vector

    def create_kmeans(self, num_centroids):
        vector_list = list(self.vectors.values())
        vectors_to_cluster = numpy.stack(vector_list)

        niter = 50
        verbose = False
        d = vectors_to_cluster.shape[1]
        kmeans = faiss.Kmeans(d, num_centroids, niter=niter, verbose=verbose)
        kmeans.train(vectors_to_cluster)

        cost = str(kmeans.obj[niter - 1])
        centroids = kmeans.centroids

        return kmeans, centroids, cost

    def compute_nearest_centroids(self):
        if self.kmeans is None:
            kmeans, centroids, cost = self.create_kmeans(self.num_centroids)
            self.centroids = centroids
            self.kmeans = kmeans
            self.cost = cost

        vector_array = numpy.stack(list(self.vectors.values()))
        _, nearest_centroids = self.kmeans.index.search(vector_array, 1)
        nearest_centroids = nearest_centroids.flatten()
        return nearest_centroids

    def create_move_list(self):
        if self.nearest_centroids is None:
            self.nearest_centroids = self.compute_nearest_centroids()

        move_list = {}
        for idx, filename in enumerate(self.vectors):
            move_list[filename] = self.nearest_centroids[idx]
        return move_list

    def calculate_best_k(self, max_k=100):
        # Implements elbow method: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        # Other methods: https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
        costs = []
        for i in range(1, max_k):
            kmeans, _, cost = self.create_kmeans(num_centroids=i)
            costs.append(cost)
            print(cost)

        costs = [int(float(cost)) for cost in costs]

        # plot the cost against K values
        plt.plot(range(1, max_k), costs)
        plt.xlabel("Value of K")
        plt.ylabel("Sqaured Error (Cost)")
        if mpl.backends.backend == "agg":
            plt.savefig("best_k_value.png")
        else:
            plt.show()
