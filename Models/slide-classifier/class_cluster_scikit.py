import os, numpy
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('=> Class Cluster: No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, algorithm_name="kmeans", num_centroids=20, preference=None, damping=0.5):
        """Set up cluster object by defining necessary variables and asserting that user provided algorithm is supported"""
        self.vectors = dict()

        algorithms = ["kmeans","affinity_propagation"]
        assert algorithm_name in algorithms

        self.algorithm_name = algorithm_name
        self.centroids = None
        self.algorithm = None
        self.cost = None
        self.labels = None
        self.num_centroids = num_centroids
        self.preference = preference
        self.damping = damping
    
    def add(self, vector, filename):
        """Adds a filename and its coresponding feature vector to the cluster object"""
        self.vectors[filename] = vector

    def create_algorithm_if_none(self):
        """Creates algorithm if it has not been created (if it equals None) based on algorithm_name set in __init__"""
        if self.algorithm is None:
            if self.algorithm_name == "kmeans":
                self.create_kmeans(self.num_centroids)
            elif self.algorithm_name == "affinity_propagation":
                self.create_affinity_propagation(self.preference, self.damping)

    def predict(self, array):
        """Wrapper function for algorithm.predict. Creates algorithm if it has not been created."""
        self.create_algorithm_if_none()
        return self.algorithm.predict(array)

    def get_vector_array(self):
        """Return a numpy array of the list of vectors stored in self.vectors"""
        vector_list = list(self.vectors.values())
        vector_array = numpy.stack(vector_list)
        return vector_array

    def create_affinity_propagation(self, preference, damping, store=True):
        """Create and fit an affinity propagation cluster"""
        vector_array = self.get_vector_array()

        affinity_propagation = AffinityPropagation(preference=preference, damping=damping)
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
    
    def create_move_list(self):
        """Creates a dictionary of file names and their coresponding centroid numbers"""
        self.create_algorithm_if_none()
            
        move_list = dict()
        for idx, filename in enumerate(self.vectors):
            move_list[filename] = self.labels[idx]
        return move_list

    def get_num_clusters(self):
        if self.algorithm_name == "affinity_propagation":
            cluster_centers_indices = self.algorithm.cluster_centers_indices_
            n_clusters_ = len(cluster_centers_indices)
            return n_clusters_
        else:
            return self.num_centroids

    def calculate_best_k(self, max_k=50):
        """
        Implements elbow method to graph the cost (squared error) as a function of the number of centroids (value of k)
        The point at which the graph becomes essentially linear is the optimal value of k.
        Only works if `algorithm` is "kmeans".
        """
        # Elbow method: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        # Other methods: https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
        assert self.algorithm_name == "kmeans"
        costs = []
        for i in range(1, max_k):
            kmeans, _, cost, _ = self.create_kmeans(num_centroids=i, store=False)
            costs.append(cost)
            print("Iteration " + str(i) + ": " + cost)
        
        costs = [int(float(cost)) for cost in costs]

        # plot the cost against K values
        plt.plot(range(1, max_k), costs)
        plt.xlabel("Value of K")
        plt.ylabel("Sqaured Error (Cost)")
        if mpl.backends.backend == "agg":
            plt.savefig("best_k_value.png")
        else:
            plt.show()