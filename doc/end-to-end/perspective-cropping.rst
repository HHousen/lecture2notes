.. _perspective_cropping:

Perspective Cropping
====================

To improve the SSA (see :ref:`slide_structure_analysis`) and the set of slides shown to the user, the frames classified as ``presenter_slide`` need to be cropped to only contain the slide. Two main algorithms were created to accomplish this: feature matching and corner crop transform.

.. _feature_matching:

SIFT Matcher & Perspective Cropping
-----------------------------------

The feature matching algorithm iterates through the ``slide`` and ``presenter_slide`` frames in chronological order. When the class switches, the algorithm begins matching using Oriented FAST and Rotated BRIEF (ORB) (which performs the same task as Scale Invariant Feature Transform (SIFT) but at two orders of magnitude the speed) for feature detection/description and Fast Library for Approximate Nearest Neighbors (FLANN) for matching. If the number of matched features is above a threshold then the images are considered to contain the same slide. The algorithm then continues detecting duplicates based on the number of feature matches until another ``slide`` frame is encountered. The ``slide`` or ``presenter_slide`` frame with the most content is kept. If the best frame is a ``presenter_slide`` then the RANSAC transformation will be used to crop the image to only contain the slide.

.. note:: The main function is :meth:`lecture2notes.end_to_end.sift_matcher.match_features`

Content Detector
^^^^^^^^^^^^^^^^

The content detector determines if and how much content is added between two images. The algorithm dilates both images and finds contours. It then computes the total area of those contours. If :math:`\gamma\%` more than the area of the first image's contours is greater than the area of the second image's contours then it is assumed more content is added. The difference in area is the amount of content added.

.. note:: The main function is :meth:`lecture2notes.end_to_end.sift_matcher.is_content_added`

Camera Motion Detection
^^^^^^^^^^^^^^^^^^^^^^^

The camera motion detection algorithm detects camera movement between two frames by tracking features along the borders of the image. Only the borders are used because the center of the image will contain a slide. Tracking features of the slide is not robust since those features will disappear when the slide changes. Furthermore, features are not found in the bottom border because ``presenter_slide`` images may have the peoples' heads at the bottom, which will move and do not represent camera motion. Features are identified using ShiTomasi Corner Detection. The Lucas Kanade optical flow is calculated between consecutive frames and the average distance of all features is the total camera movement. If the camera moves more than 10 pixels, then there is assumed to be camera movement. If the camera doesn't move then the feature matching algorithm will automatically crop each ``presenter_slide`` frame even if it does not have a matching ``slide`` frame.

.. note:: The main function is :meth:`lecture2notes.end_to_end.sift_matcher.does_camera_move`


Corner Crop Transform
---------------------

Located on its own page here: :ref:`corner_crop_transform`.

.. note:: The main function is :meth:`lecture2notes.end_to_end.corner_crop_transform.crop`
