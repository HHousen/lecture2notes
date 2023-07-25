import itertools
import logging
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import auto_canny
from tqdm import tqdm

from .corner_crop_transform import persp_transform
from .figure_detection import area_of_overlapping_rectangles, convert_coords_to_corners
from .helpers import frame_number_from_filename

logger = logging.getLogger(__name__)

OUTPUT_PATH_MODIFIER = "_cropped"


def sift_flann_match(query_image, train_image, algorithm="orb", num_features=1000):
    """Locate ``query_image`` within ``train_image`` using ``algorithm`` for feature
    detection/description and `FLANN (Fast Library for Approximate Nearest Neighbors) <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#flann-based-matcher>`_
    for matching. You can read more about matching in the
    `OpenCV "Feature Matching" documentation <https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html>`_
    or about homography on the `OpenCV Python Tutorial "Feature Matching + Homography to find Objects" <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html>`_

    Args:
        query_image (np.array): Image to find. Loading using ``cv2.imread()``.
        train_image (np.array): Image to search. Loading using ``cv2.imread()``.
        algorithm (str, optional): The feature detection/description algorithm. Can be one
            of `ORB <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html>`_,
            (`ORB Class Reference <https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html>`_)
            `SIFT <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html>`_,
            (`SIFT Class Reference <https://docs.opencv.org/3.4.9/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html>`_)
            or `FAST <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html>`_.
            (`FAST Class Reference <https://docs.opencv.org/3.4/df/d74/classcv_1_1FastFeatureDetector.html>`_)
            Defaults to "orb".
        num_features (int, optional): The maximum number of features to retain when using `ORB`
            and `SIFT`. Does not take effect when using the `FAST` detection  algorithm. Setting
            to 0 for `SIFT` is a good starting point. The default for `ORB` is 500, but it was
            increased to 1000 to improve accuracy. Defaults to 1000.

    Returns:
        tuple: (good, kp1, kp2, img1, img2) The good matches as per Lowe's ratio test, the
        key points from image 1, the key points from image 2, modified image 1, and modified
        image 2.
    """
    algorithm = algorithm.lower()

    img1 = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    # Initiate detector and find the keypoints and descriptors
    if algorithm == "orb":
        orb = cv2.ORB_create(nfeatures=num_features)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
    elif algorithm == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    elif algorithm == "fast":
        fast = cv2.FastFeatureDetector_create(threshold=25)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)
        kp1 = fast.detect(img1, None)
        _, des1 = sift.compute(img1, kp1)
        kp2 = fast.detect(img2, None)
        _, des2 = sift.compute(img2, kp2)

    # FLANN parameters
    if algorithm == "orb":
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1,  # 2
        )
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=75)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    # Source: https://stackoverflow.com/a/32354540
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        (m, n) = m_n

        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good, kp1, kp2, img1, img2


def ransac_transform(sift_matches, kp1, kp2, img1, img2, draw_matches=False):
    """Use data from :meth:`~lecture2notes.end_to_end.sift_matcher.sift_flann_match` to find the coordinates
    of ``img1`` in ``img2``. ``sift_matches``, ``kp1``, ``kp2``, ``img1``, and ``img2``
    are all the outputs of meth:`~sift_matcher.sift_flann_match`. If ``draw_matches``
    is enabled then the features matches will be drawn and shown on the screen.

    Returns:
        np.array: The corner coordinates of the quadrilateral representing ``img1``
        within ``img2``.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in sift_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in sift_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # found_image_highlighted = cv2.polylines(img2, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
    # cv2.imwrite("found_image_highlighted.png", found_image_highlighted)

    if draw_matches:
        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, sift_matches, None, **draw_params)
        plt.imshow(img3, "gray"), plt.show()

    return dst


def does_camera_move(old_frame, frame, gamma=10, border_ratios=(10, 19), bottom=False):
    """Detects camera movement between two frames by tracking features in the borders
    of the image. Only the borders are used because the center of the image probably
    contains a slide. Thus, tracking features of the slide is not robust since those
    features will disappear when the slide changes.

    Args:
        old_frame (np.array): First frame/image as loaded with ``cv2.imread()``
        frame (np.array): Second frame/image as loaded with ``cv2.imread()``
        gamma (int, optional): The threshold pixel movement value. If the camera moves
            more than this value, then there is assumed to be camera movement between
            the two frames. Defaults to 10.
        border_ratios (tuple, optional): The ratios of the height and width respectively
            of the first frame to be searched for features. Only the borders are searched
            for features. these values specify how much of the image should be counted as
            a border. Defaults to (10, 19).
        bottom (bool, optional): Whether to find features in the bottom border. This is not
            recommended because 'presenter_slide' images may have the peoples' heads at the
            bottom, which will move and do not represent camera motion. Defaults to False.

    Returns:
        tuple: (total_movement > gamma, total_movement) If there is camera movement between
        the two frames and the total movement between the frames.
    """
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    height = old_gray.shape[0]
    width = old_gray.shape[1]
    feature_mask = np.ones_like(old_gray)

    height_bounds = height // border_ratios[0]
    width_bounds = width // border_ratios[1]

    if bottom:
        feature_mask[
            height_bounds : height - height_bounds, width_bounds : width - width_bounds
        ] = 0
    else:
        feature_mask[height_bounds:height, width_bounds : width - width_bounds] = 0
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=feature_mask, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)  # noqa: F841

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow. Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # The mean of a set of vectors is calculated component-wise. In other words,
    # for 2D vectors simply find the mean of the first coordinates and the mean
    # of the second coordinates, and those will be the coordinates of the mean vector.
    # Source: https://math.stackexchange.com/a/80925
    mean_new = np.mean(good_new, axis=0)
    mean_old = np.mean(good_old, axis=0)
    # Distance calculation. Source: https://stackoverflow.com/a/1401828
    total_movement = np.linalg.norm(mean_new - mean_old)

    # Draw the tracks
    # Create some random colors
    # color = np.random.randint(0, 255, (100, 3))
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    # img = cv2.add(frame, mask)
    # cv2.imwrite("frame.png", img)

    # If movement amount is greater than gamma then movement is detected
    return total_movement > gamma, total_movement


def does_camera_move_all_in_folder(folder_path):
    """Runs :meth:`~lecture2notes.end_to_end.sift_matcher.does_camera_move` on all the files in a folder and
    calculates statistics about camera movement within those files.

    Args:
        folder_path (str): Directory containing the files to be processed.

    Returns:
        tuple: (movement_detection_percentage, average_move_value, max_move_value)
        A float representing the precentage of frames where movement was detected from
        the previous frame. The average of the ``total_movement`` values returned from
        :meth:`~lecture2notes.end_to_end.sift_matcher.does_camera_move`. The maximum of the the ``total_movement``
        values returned from :meth:`~lecture2notes.end_to_end.sift_matcher.does_camera_move`.
    """
    images = [
        os.path.join(folder_path, image_path) for image_path in os.listdir(folder_path)
    ]
    images.sort()

    if len(images) < 2:
        return 0, 0, 0

    move_values = []
    movement_detection_values = []
    previous_image = cv2.imread(images[0])
    for current_image_path in tqdm(images[1:], desc="Detecting Camera Movement"):
        current_image = cv2.imread(current_image_path)

        movement_detection, move_value = does_camera_move(previous_image, current_image)
        move_values.append(move_value)
        movement_detection_values.append(movement_detection)
        previous_image = current_image

    average_move_value = sum(move_values) / len(move_values)
    movement_detection_percentage = sum(movement_detection_values) / len(
        movement_detection_values
    )
    max_move_value = max(move_values)
    return movement_detection_percentage, average_move_value, max_move_value


def is_content_added(
    first,
    second,
    first_area_modifier=0.70,
    second_area_modifier=0.40,
    gamma=0.09,
    dilation_amount=22,
):
    r"""Detect if ``second`` contains more content than ``first`` and how much more content
    it adds. This algorithm dilates both images and finds contours. It then computes the total
    area of those contours. If ``gamma``\% more than the area of the first image's contours is
    greater than the area of the second image's contours then it is assumed more content is
    added.

    Args:
        first (np.array): Image loaded using ``cv2.imread()`` belonging to the 'slide' class
        second (np.array): Image loaded using ``cv2.imread()`` belonging to the
            'presenter_slide' class
        first_area_modifier (float, optional): The maximum percent area of the ``first`` image
            that a contour can take up before it is excluded. Defaults to 0.70.
        second_area_modifier (float, optional): The maximum percent area of the ``second`` image
            that a contour can take up before it is excluded. Images belonging to the
            'presenter_slide' class are more likely to have mistaken large contours. Defaults
            to 0.40.
        gamma (float, optional): The percentage increase in content area necessary for
            `second`` to be classified as having more content than ``first``. Defaults to 0.09.
        dilation_amount (int, optional): How much the canny edge maps of each both images
            ``first`` and ``second`` should be dilated. This helps to combine multiple
            components of one object into a single contour. Defaults to 22.

    Returns:
        tuple: (content_is_added, amount_of_added_content) Boolean if ``second`` contains more
        content than ``first`` and float describing the difference in content from ``first`` to
        ``second``. ``amount_of_added_content`` can be negative.
    """
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    first_area = first.shape[0] * first.shape[1]
    second_area = second.shape[0] * second.shape[1]

    first_thresh = cv2.threshold(
        first, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    # Need to use canny in addition to threshold in case the threshold is inverted.
    # Difference between edges and contours: https://stackoverflow.com/a/17104541
    first_canny = auto_canny(first_thresh)
    first_canny_dilated = cv2.dilate(
        first_canny, np.ones((dilation_amount, dilation_amount), dtype=np.uint8)
    )
    first_contours = cv2.findContours(
        first_canny_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    first_contour_area = sum(
        cv2.contourArea(x)
        for x in first_contours
        if cv2.contourArea(x) < first_area_modifier * first_area
    )
    # first_contour_image = cv2.drawContours(first.copy(), first_contours, -1, (0, 255, 0), 3)
    # plt.imshow(first_contour_image, "gray"), plt.show()

    # first_bounding_boxes = np.array(
    #     [cv2.boundingRect(contour) for contour in first_contours]
    # )
    # first_bounding_boxes_image = first.copy()
    # for x, y, w, h in first_bounding_boxes:
    #     cv2.rectangle(first_bounding_boxes_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # plt.imshow(first_bounding_boxes_image, "gray"), plt.show()

    second_thresh = cv2.threshold(
        second, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    second_canny = auto_canny(second_thresh)
    second_canny_dilated = cv2.dilate(
        second_canny, np.ones((dilation_amount, dilation_amount), dtype=np.uint8)
    )
    second_contours = cv2.findContours(
        second_canny_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    second_contour_area = sum(
        cv2.contourArea(x)
        for x in second_contours
        if cv2.contourArea(x) < second_area_modifier * second_area
    )
    # second_contour_image = cv2.drawContours(
    #     second.copy(), second_contours, -1, (0, 255, 0), 3
    # )
    # plt.imshow(second_contour_image, "gray"), plt.show()

    # Show the two thresholds
    # scale_to = first.shape[:2]
    # scale_to = (scale_to[1], scale_to[0])
    # second_thresh = cv2.resize(second_thresh, scale_to)
    # thresh_combined = np.hstack((first_thresh, second_thresh))
    # plt.imshow(thresh_combined, "gray"), plt.show()

    logger.debug("First Image Contour Area: %.1f", first_contour_area)
    logger.debug("Second Image Contour Area: %.1f", second_contour_area)

    content_is_added = first_contour_area * (1 + gamma) < second_contour_area
    amount_of_added_content = second_contour_area - first_contour_area
    return content_is_added, amount_of_added_content


def match_features(
    slide_path,
    presenter_slide_path,
    min_match_count=33,
    min_area_percent=0.37,
    do_motion_detection=True,
):
    """Match features between images in `slide_path` and `presenter_slide_path`.
    The images in `slide_path` are the queries to the matching algorithm and the
    images in `presenter_slide_path` are the train/searched images.

    Args:
        slide_path (str): Path to the images classified as "slide" or any directory
            containing query images.
        presenter_slide_path (str): Path to the images classified as "presenter_slide"
            or any directory containing train images.
        min_match_count (int, optional): The minimum number of matches returned by
            :meth:`~lecture2notes.end_to_end.sift_matcher.sift_flann_match` required for the image pair to
            be considered as containing the same slide. Defaults to 33.
        min_area_percent (float, optional): Percentage of the area of the train image (images
            belonging to the 'presenter_slide' category) that a matched slide must take up to
            be counted as a legitimate duplicate slide. This removes incorrect matches that
            can result in crops to small portions of the train image. Defaults to 0.37.
        do_motion_detection (bool, optional): Whether motion detection using
            :meth:`~lecture2notes.end_to_end.sift_matcher.does_camera_move_all_in_folder` should be performed.
            If set to False then it is assumed that there is movement since assuming no
            movement leaves room for a lot of false positives. If no camera motion is detected
            and this option is enabled then all slides that are unique to the "presenter_slide"
            category (they have no matches in the "slide" category) will automatically be
            cropped to contain just the slide. They will be saved to the originating folder but
            with the string defined by the variable ``OUTPUT_PATH_MODIFIER`` in their filename.
            Even if :meth:`~lecture2notes.end_to_end.sift_matcher.does_camera_move_all_in_folder` detects no movement it is
            still possible that movement is detected while running this function since a check is
            performed to make sure all slide bounding boxes found contain 80% overlapping area with
            all previously found bounding boxes. Defaults to True.

    Returns:
        tuple: (non_unique_presenter_slides, transformed_image_paths)
        ``non_unique_presenter_slides``: The images in the  "presenter_slide" category that are
        not unique and should be deleted
        ``transformed_image_paths``: The paths to the cropped images if `do_motion_detection`
        was enabled and no motion was detected.
    """
    if do_motion_detection:
        movement_detected = does_camera_move_all_in_folder(presenter_slide_path)[0] > 0.05
    else:
        movement_detected = True

    slide_images = [
        os.path.join(slide_path, image_path) for image_path in os.listdir(slide_path)
    ]
    presenter_slide_images = [
        os.path.join(presenter_slide_path, image_path)
        for image_path in os.listdir(presenter_slide_path)
    ]

    slide_images_dict = list(zip(slide_images, itertools.repeat("slide")))
    presenter_slide_images_dict = list(
        zip(presenter_slide_images, itertools.repeat("presenter_slide"))
    )
    images = slide_images_dict + presenter_slide_images_dict
    # Sort according to the image filename, which will organize into
    # chronological order due to the frame number in the filename.
    # The regex selects the number between an underscore and an underscore
    # or a period. This will match filenames like "IJquEYhiq_U-img_050_noborder.jpg"
    # and "IJquEYhiq_U-img_140.jpg".
    regex_sort = lambda o: frame_number_from_filename(o[0])  # noqa: E731
    images = sorted(images, key=regex_sort)

    non_unique_presenter_slides = []
    current_batch_non_unique_presenter_slides = []
    all_dst_corners = []
    match_successful = False
    frame_with_most_content = None
    dst_coords = None
    previous_category = images[0][1]

    for idx, (filename, category) in tqdm(
        enumerate(images[1:]), desc="Feature Matching", total=len(images[1:])
    ):
        previous_category = images[idx][1]
        if category != previous_category:
            logger.info("Switching from '%s' to '%s'", previous_category, category)
            # If a 'presenter_slide' was detected as having more content than
            # the `query_image` (aka the previous 'slide'), then add the `query_image`
            # to the `non_unique_presenter_slides` and remove `frame_with_most_content`
            # from `current_batch_non_unique_presenter_slides`.
            if frame_with_most_content is not None:
                logger.info(
                    "Removing %s from and adding %s to the current batch of non unique presenter slides",
                    frame_with_most_content,
                    query_image_path,  # noqa: F821
                )
                non_unique_presenter_slides.append(query_image_path)  # noqa: F821
                current_batch_non_unique_presenter_slides.remove(
                    frame_with_most_content
                )

            # Append the current batch of non unique presenter slides if there are
            # any items in `current_batch_non_unique_presenter_slides`
            if current_batch_non_unique_presenter_slides:
                non_unique_presenter_slides += current_batch_non_unique_presenter_slides

            # Reset variables needed for the detection of multiple 'presenter_slide'
            # images in a row.
            match_successful = False
            max_content = 0
            frame_with_most_content = None
            current_batch_non_unique_presenter_slides.clear()

            previous_filename = images[idx][0]  # 0 selects filename
            # If the most recently processed 'presenter_slide' was added to
            # `non_unique_presenter_slides` then it matches with the last detected
            # 'slide' and does not need to be compared to the 'slide' that is going to
            # be tested in this iteration.
            if (
                len(non_unique_presenter_slides) > 1
                and previous_filename == non_unique_presenter_slides[-1]
            ):
                logger.info(
                    "Skipping iteration since last slide in segment of 'presenter_slide' matched with the previously detected slide."
                )
                continue

            if category == "presenter_slide":  # previous_category is "slide"
                # previous_filename is a "slide"
                query_image_path = previous_filename
                # filename is a "presenter_slide"
                train_image_path = filename
            elif category == "slide":  # previous_category is "presenter_slide"
                # filename is a "slide"
                query_image_path = filename
                # previous_filename is a "presenter_slide"
                train_image_path = previous_filename
            else:
                logger.error("Category not found")
                sys.exist(1)

            query_image = cv2.imread(query_image_path)
            train_image = cv2.imread(train_image_path)

            train_image_area = train_image.shape[0] * train_image.shape[1]

            sift_outputs = sift_flann_match(query_image, train_image)

            sift_matches = sift_outputs[0]
            print("Matches " + str(len(sift_matches)))
            # ransac_transform(*sift_outputs, draw_matches=True)
            if len(sift_matches) > min_match_count:  # Images contain the same slide
                logger.info(
                    "%s contains the 'slide' at %s according to the number of matches",
                    train_image_path,
                    query_image_path,
                )
                match_successful = True

                dst_coords = ransac_transform(*sift_outputs)
                dst_coords_area = cv2.contourArea(dst_coords)

                # If the area of the detected slide in the 'presenter_slide' image is less than
                # `min_area_percent`% of the area of the entire 'presenter_slide' image then the
                # detection is a false-positive (it is not a slide) and we should move to the
                # next iteration.
                max_area = min_area_percent * train_image_area
                if dst_coords_area < max_area:
                    logger.info(
                        "Previously detected match is incorrect because the detected slide is too small. It had an area of %i, while the minimum is %i",
                        dst_coords_area,
                        max_area,
                    )
                    continue

                content_is_added, amount_of_added_content = is_content_added(
                    query_image, persp_transform(train_image, dst_coords)
                )
                if content_is_added and amount_of_added_content > max_content:
                    logger.info(
                        "%i pixels of content added to %s by %s",
                        amount_of_added_content,
                        query_image_path,
                        train_image_path,
                    )
                    max_content = amount_of_added_content
                    frame_with_most_content = train_image_path

                if not movement_detected:
                    dst_corners = convert_coords_to_corners(
                        cv2.boundingRect(dst_coords)
                    )
                    for prev_dst_corners in all_dst_corners:
                        if (
                            area_of_overlapping_rectangles(
                                dst_corners, prev_dst_corners
                            )
                            < 0.80 * dst_coords_area
                        ):
                            movement_detected = True
                    all_dst_corners.append(dst_corners)

                current_batch_non_unique_presenter_slides.append(train_image_path)

            continue

        if category == "presenter_slide" and match_successful:
            train_image_path = filename
            logger.info(
                "In a section of 'presenter_slide' and match was successful when entering this section. Testing %s...",
                filename,
            )
            train_image = cv2.imread(train_image_path)
            sift_outputs = sift_flann_match(query_image, train_image)

            if len(sift_matches) > min_match_count:  # Images contain the same slide
                content_is_added, amount_of_added_content = is_content_added(
                    query_image, persp_transform(train_image, dst_coords)
                )
                if content_is_added and amount_of_added_content > max_content:
                    logger.info(
                        "%i pixels of content added to %s by %s",
                        amount_of_added_content,
                        query_image_path,
                        train_image_path,
                    )
                    max_content = amount_of_added_content
                    frame_with_most_content = train_image_path
                logger.info(
                    "%s marked for removal because it contains enough matches with the last 'slide' at %s",
                    filename,
                    query_image_path,
                )
                current_batch_non_unique_presenter_slides.append(train_image_path)
            else:
                # If the 'presenter_slide' is not a duplicate of the last 'slide' then
                # it is unlikely that any future 'presenter_slide' images will match
                # since presentations don't frequently go back to previous slides.
                match_successful = False

    transformed_image_paths = []
    if not movement_detected and dst_coords is not None:
        presenter_slide_images = [
            x for x in presenter_slide_images if x not in non_unique_presenter_slides
        ]
        for image in presenter_slide_images:
            file_parse = os.path.splitext(str(image))
            filename = file_parse[0]
            ext = file_parse[1]
            output_path = filename + OUTPUT_PATH_MODIFIER + ext
            transformed_image_paths.append(output_path)

            transformed_image = persp_transform(cv2.imread(image), dst_coords)
            cv2.imwrite(output_path, transformed_image)

    return non_unique_presenter_slides, transformed_image_paths


logging.basicConfig(
    format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
    level=logging.getLevelName("INFO"),
)
# does_camera_move_all_in_folder("test_data/presenter_slide")
# input(
#     match_features(
#         "test_data/slide_new",
#         "test_data/presenter_slide_new",
#         do_motion_detection=False,
#     )
# )

# first = cv2.imread("test_data/IJquEYhiq_U-img_038.jpg")
# second = cv2.imread("test_data/IJquEYhiq_U-img_043.jpg")
# input(is_content_added(first, second))
