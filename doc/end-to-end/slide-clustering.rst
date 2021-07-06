.. _slide_clustering:

Slide Clustering
================

Clustering is used to group slides that contain the same content. We implement two main methods of clustering: ``normal`` algorithms (Affinity Propagation and KMeans) and our novel ``segment`` approach, which is the default. By grouping slides, the system is capable of choosing the frame that best represents each group.

Normal Algorithms
-----------------

When the ``normal`` mode is selected, if the number of slides is specified by the user KMeans will be used, otherwise Affinity Propagation will be used. Features are extracted from the layer before pooling for all model architectures. KMeans and Affinity Propagation select the frame closest to the centroid.

Relevant Classes:

1. :class:`lecture2notes.end_to_end.cluster.ClusterFilesystem`
2. :class:`lecture2notes.models.slide_classifier.class_cluster_scikit.Cluster`

Segment Method
--------------

The ``segment`` clustering method iterates chronologically through extracted frames that have been filtered and transformed by the perspective cropping (see Section \ref{Perspective Cropping}) algorithm. The algorithm marks splitting points based on large visual differences, which are measured by the cosine similarity between the feature vectors extracted from the slide classifier. A split is marked when the difference between two frames is greater than the mean of the differences across all frames.

Relevant Classes:

1. :class:`lecture2notes.end_to_end.segment_cluster.SegmentCluster`
