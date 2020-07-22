import os
import re
import sys
import itertools
import logging
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from imutils import auto_canny
from corner_crop_transform import persp_transform
from figure_detection import convert_coords_to_corners, area_of_overlapping_rectangles

logger = logging.getLogger(__name__)

OUTPUT_PATH_MODIFIER = "_cropped"


def sift_flann_match(query_image, train_image, algorithm="orb"):
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
        `SIFT <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html>`_, 
        or `FAST <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html>`_. 
        Defaults to "orb".

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
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
    elif algorithm == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    elif algorithm == "fast":
        fast = cv2.FastFeatureDetector_create(threshold=25)
        sift = cv2.xfeatures2d.SIFT_create()
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
    """Use data from :meth:`~sift_matcher.sift_flann_match` to find the coordinates
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
    mask = np.zeros_like(old_frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow. Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

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

    # If movement amount is greater than gamma then movement is detected
    return total_movement > gamma, total_movement

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


def does_camera_move_all_in_folder(folder_path):
    """Runs :meth:`~sift_matcher.does_camera_move` on all the files in a folder and 
    calculates statistics about camera movement within those files.

    Args:
        folder_path (str): Directory containing the files to be processed.

    Returns:
        tuple: (movement_detection_percentage, average_move_value, max_move_value)
        A float representing the precentage of frames where movement was detected from
        the previous frame. The average of the ``total_movement`` values returned from
        :meth:`~sift_matcher.does_camera_move`. The maximum of the the ``total_movement`` 
        values returned from :meth:`~sift_matcher.does_camera_move`.
    """
    images = [
        os.path.join(folder_path, image_path) for image_path in os.listdir(folder_path)
    ]
    images.sort()

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
    """Detect if ``second`` contains more content than ``first`` and how much more content
    it adds. This algorithm dilates both images and finds contours. It then computes the total
    area of those contours. If ``gamma``% more than the area of the first image's contours is 
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

    dst_coords = np.array(
        [
            [[80.396248, 66.36963]],
            [[80.84033, 940.3144]],
            [[1620.6646, 922.0189]],
            [[1638.6852, 80.77042]],
        ]
    )
    second = persp_transform(second, dst_coords)

    first_area = first.shape[0] * first.shape[1]
    second_area = second.shape[0] * second.shape[1]

    first_thresh = cv2.threshold(
        first, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    first_canny = auto_canny(first_thresh)
    first_canny_dilated = cv2.dilate(
        first_canny, np.ones((dilation_amount, dilation_amount), dtype=np.uint8)
    )
    first_contours = cv2.findContours(
        first_canny_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    first_contours = (
        first_contours[0] if len(first_contours) == 2 else first_contours[1]
    )
    first_contour_area = sum(
        [
            cv2.contourArea(x)
            for x in first_contours
            if cv2.contourArea(x) < first_area_modifier * first_area
        ]
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
    )
    second_contours = (
        second_contours[0] if len(second_contours) == 2 else second_contours[1]
    )
    second_contour_area = sum(
        [
            cv2.contourArea(x)
            for x in second_contours
            if cv2.contourArea(x) < second_area_modifier * second_area
        ]
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


# min_match_count = 120 for SIFT
def match_features(
    slide_path, presenter_slide_path, min_match_count=46, do_motion_detection=True
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
            :meth:`~sift_matcher.sift_flann_match` required for the image pair to
            be considered as containing the same slide. Defaults to 46.
        do_motion_detection (bool, optional): Whether motion detection using 
            :meth:`~sift_matcher.does_camera_move_all_in_folder` should be performed. 
            If set to False then it is assumed that there is movement since assuming no 
            movement leaves room for a lot of false positives. If no camera motion is detected 
            and this option is enabled then all slides that are unique to the "presenter_slide" 
            category (they have no matches in the "slide" category) will automatically be 
            cropped to contain just the slide. They will be saved to the originating folder but 
            with the string defined by the variable ``OUTPUT_PATH_MODIFIER`` in their filename. 
            Even if :meth:`~sift_matcher.does_camera_move_all_in_folder` detects no movement it is 
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
        movement_detected = does_camera_move_all_in_folder(presenter_slide_path)
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
    regex_sort = lambda o: re.search("(?<=\_)[0-9]+(?=\_|.)", o[0]).group(0)
    images = sorted(images, key=regex_sort)

    non_unique_presenter_slides = []
    current_batch_non_unique_presenter_slides = []
    all_dst_coords = []
    match_successful = False
    frame_with_most_content = None
    previous_category = images[0][1]

    for idx, (filename, category) in tqdm(
        enumerate(images[1:]), desc="Feature Matching", total=len(images[1:])
    ):
        previous_category = images[idx][1]
        if category != previous_category:
            logger.debug("Switching from '%s' to '%s'", previous_category, category)
            # If a 'presenter_slide' was detected as having more content than
            # the `query_image` (aka the previous 'slide'), then add the `query_image`
            # to the `non_unique_presenter_slides` and remove `frame_with_most_content`
            # from `current_batch_non_unique_presenter_slides`.
            if frame_with_most_content is not None:
                logger.debug(
                    "Removing %s from and adding %s to the current batch of non unique presenter slides",
                    frame_with_most_content,
                    query_image_path,
                )
                non_unique_presenter_slides.append(query_image_path)
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

            previous_filename = images[idx][0]  # 0 selects filename
            # If the most recently processed 'presenter_slide' was added to
            # `non_unique_presenter_slides` then it matches with the last detected
            # 'slide' and does not need to be compared to the 'slide' that is going to
            # be tested in this iteration.
            if (
                len(non_unique_presenter_slides) > 1
                and previous_filename == non_unique_presenter_slides[-1]
            ):
                logger.debug(
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

            sift_outputs = sift_flann_match(query_image, train_image)

            sift_matches = sift_outputs[0]
            if len(sift_matches) > min_match_count:  # Images contain the same slide
                logger.debug(
                    "%s contains the 'slide' at %s", train_image_path, query_image_path
                )
                match_successful = True
                dst_coords = ransac_transform(*sift_outputs)

                content_is_added, amount_of_added_content = is_content_added(
                    query_image, persp_transform(train_image, dst_coords)
                )
                if content_is_added and amount_of_added_content > max_content:
                    logger.debug(
                        "%i pixels of content added to %s by %s",
                        amount_of_added_content,
                        query_image_path,
                        train_image_path,
                    )
                    max_content = amount_of_added_content
                    frame_with_most_content = train_image_path

                if not movement_detected:
                    dst_corners = convert_coords_to_corners(dst_coords)
                    dst_corners_area = area_of_corner_box(dst_corners)
                    for prev_dst_corners in all_dst_coords:
                        if (
                            area_of_overlapping_rectangles(
                                dst_corners, prev_dst_corners
                            )
                            < 0.80 * dst_corners_area
                        ):
                            movement_detected = True
                    all_dst_coords.append(dst_coords)

                current_batch_non_unique_presenter_slides.append(train_image_path)

            continue

        if category == "presenter_slide" and match_successful:
            train_image_path = filename
            logger.debug(
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
                    logger.debug(
                        "%i pixels of content added to %s by %s",
                        amount_of_added_content,
                        query_image_path,
                        train_image_path,
                    )
                    max_content = amount_of_added_content
                    frame_with_most_content = train_image_path
                logger.debug(
                    "%s marked for removal because it contains enough matches with the last 'slide' at %s",
                    filename,
                    query_image_path,
                )
                current_batch_non_unique_presenter_slides.append(train_image_path)

    transformed_image_paths = []
    if not movement_detected and dst_coords:
        presenter_slide_images = [
            x for x in presenter_slide_images if x not in non_unique_presenter_slides
        ]
        for image in presenter_slide_images:
            file_parse = os.path.splitext(str(image_path))
            filename = file_parse[0]
            ext = file_parse[1]
            output_path = filename + OUTPUT_PATH_MODIFIER + ext
            transformed_image_paths.append(output_path)

            transformed_image = persp_transform(cv2.readim(image), dst_coords)
            cv2.imwrite(output_path, transformed_image)

    return non_unique_presenter_slides, transformed_image_paths


# logging.basicConfig(
#     format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
#     level=logging.getLevelName("INFO"),
# )
# does_camera_move_all_in_folder("test_data/presenter_slide")
# input(
#     match_features(
#         "test_data/slide", "test_data/presenter_slide", do_motion_detection=False
#     )
# )

# first = cv2.imread("test_data/IJquEYhiq_U-img_038.jpg")
# second = cv2.imread("test_data/IJquEYhiq_U-img_043.jpg")
# input(is_content_added(first, second))
