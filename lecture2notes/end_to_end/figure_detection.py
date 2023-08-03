import logging
import math
import os

import cv2
import numpy as np
from imutils import auto_canny
from PIL import Image, ImageStat
from .rlsa import rlsa
from skimage.measure.entropy import shannon_entropy
from tqdm import tqdm

from .helpers import frame_number_filename_mapping
from .text_detection import get_text_bounding_boxes, load_east

logger = logging.getLogger(__name__)

OUTPUT_PATH_MODIFIER = "_figure_"


def area_of_overlapping_rectangles(a, b):
    """
    Find the overlapping area of two rectangles ``a`` and ``b``.
    Inspired by https://stackoverflow.com/a/27162334.
    """
    dx = min(a[0], b[0]) - max(a[2], b[2])  # xmax, xmax, xmin, xmin
    dy = min(a[1], b[1]) - max(a[3], b[3])  # ymax, ymax, ymin, ymin
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def detect_color_image(image, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    """Detect if an image contains color, is black and white, or is grayscale.
    Based on `this StackOverflow answer <https://stackoverflow.com/a/23035464>`__.

    Args:
        image (np.array): Input image
        thumb_size (int, optional): Resize image to this size to speed up calculation.
            Defaults to 40.
        MSE_cutoff (int, optional): A larger value requires more color for an image to be
            labeled as "color". Defaults to 22.
        adjust_color_bias (bool, optional): Mean color bias adjustment, which improves the
            prediction. Defaults to True.

    Returns:
        str: Either "grayscale", "color", "b&w" (black and white), or "unknown".
    """
    pil_img = Image.fromarray(image)
    bands = pil_img.getbands()
    if bands == ("R", "G", "B") or bands == ("R", "G", "B", "A"):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum(
                (pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2]
            )
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= MSE_cutoff:
            return "grayscale"
        return "color"
    if len(bands) == 1:
        return "b&w"
    return "unknown"


def convert_coords_to_corners(box):
    x, y, w, h = box
    x_values = (x + w, x)
    y_values = (y + h, y)
    rectangle = (max(x_values), max(y_values), min(x_values), min(y_values))
    return rectangle


def area_of_corner_box(box):
    return (box[0] - box[2]) * (box[1] - box[3])


def detect_figures(
    image_path,
    output_path=None,
    east="frozen_east_text_detection.pb",
    text_area_overlap_threshold=0.32,  # 0.15
    figure_max_area_percentage=0.60,
    text_max_area_percentage=0.30,
    large_box_detection=True,
    do_color_check=True,
    do_text_check=True,
    entropy_check=2.5,
    do_remove_subfigures=True,
    do_rlsa=False,
):
    """Detect figures located in a slide.

    Args:
        image_path (str): Path to the image to process.
        output_path (str, optional): Path to save the figures. Defaults to
            ``[filename]_figure_[index].[ext]``.
        east (str or cv2.dnn_Net, optional): Path to the EAST model file or the pre-trained
            EAST model loaded with :meth:`~lecture2notes.end_to_end.text_detection.load_east`. ``do_text_check`` must
            be true for this option to take effect. Defaults to "frozen_east_text_detection.pb".
        text_area_overlap_threshold (float, optional): The percentage of the figure that
            can contain text. If the area of the text in the figure is greater than this
            value, the figure is discarded. ``do_text_check`` must be true for this option
            to take effect. Defaults to 0.10.
        figure_max_area_percentage (float, optional): The maximum percentage of the area of the
            original image that a figure can take up. If the figure uses more area than
            ``original_image_area*figure_max_area_percentage`` then the figure will be discarded.
            Defaults to 0.70.
        text_max_area_percentage (float, optional): The maximum percentage of the area of the
            original image that a block of text (as identified by the EAST model) can take up.
            If the text block uses more area than ``original_image_area*text_max_area_percentage``
            then that text block will be ignored. ``do_text_check`` must be true for this option
            to take effect. Defaults to 0.30.
        large_box_detection (bool, optional): Detect edges and classify large rectangles as
            figures. This will ignore `do_color_check` and
            `do_text_check`. This is useful for finding tables for example. Defaults to True.
        do_color_check (bool, optional): Check that potential figures contain color. This
            helps to remove large quantities of black and white text form the potential
            figure list. Defaults to True.
        do_text_check (bool, optional): Check that only `text_area_overlap_threshold` of
            potential figures contains text. This is useful to remove blocks of text that
            are mistakenly classified as figures. Checking for text increases processing
            time so be careful if processing a large number of files. Defaults to True.
        entropy_check (float, optional): Check that the entropy of all potential figures is above
            this value. Figures with a ``shannon_entropy`` lower than this value will be removed.
            Set to ``False`` to disable this check. The ``shannon_entropy`` implementation is from
            ``skimage.measure.entropy``. IMPORTANT: This check applies to both the regular tests
            *and* ``large_box_detection``, which most check do not apply to. Defaults to 3.5.
        do_remove_subfigures (bool, optional): Check that there are no overlapping figures.
            If an overlapping figure is detected, the smaller figure will be deleted. This
            is useful to have enabled when using `large_box_detection` since
            `large_box_detection` will commonly mistakenly detect subfigures. Defaults to True.
        do_rlsa (bool, optional): Use RLSA (Run Length Smoothing Algorithm) instead of dilation.
            Does not apply to `large_box_detection`. Defaults to False.

    Returns:
        tuple: (figures, output_paths) A list of figures extracted from the input slide image
        and a list of paths to those figures on disk.
    """
    image = cv2.imread(image_path)

    # image = cv2.copyMakeBorder(
    #     image,
    #     20,
    #     20,
    #     20,
    #     20,
    #     cv2.BORDER_CONSTANT,
    #     value=[0, 0, 0],
    # )

    image_height = image.shape[0]
    image_width = image.shape[1]
    image_area = image_height * image_width

    if not output_path:
        file_parse = os.path.splitext(str(image_path))
        filename = file_parse[0]
        ext = file_parse[1]
        start_output_path = filename + OUTPUT_PATH_MODIFIER

    if do_text_check:
        text_bounding_boxes = get_text_bounding_boxes(image, east)
        # Remove boxes that are too large
        text_bounding_boxes = [
            box
            for box in text_bounding_boxes
            if area_of_corner_box(box)  # area of box
            < text_max_area_percentage * image_area
        ]

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blurred = cv2.GaussianBlur(gray_thresh, (3, 3), 0)

    # Need to use canny in addition to threshold in case the threshold is inverted.
    # Difference between edges and contours: https://stackoverflow.com/a/17104541
    canny = auto_canny(blurred)

    # "large" pertains to components that are used to find figures not surrounded by a border
    # "small" is used to find rectangles on the slide, which are likely figures
    canny_dilated_large = cv2.dilate(canny, np.ones((22, 22), dtype=np.uint8))
    canny_dilated_small = cv2.dilate(canny, np.ones((3, 3), dtype=np.uint8))

    # cv2.imwrite("canny_dilated_large.png", canny_dilated_large)
    # cv2.imwrite("canny_dilated_small.png", canny_dilated_small)

    if do_rlsa:
        x, y = canny.shape
        value = max(math.ceil(x / 70), math.ceil(y / 70)) + 20  # heuristic
        rlsa_result = ~rlsa.rlsa(~canny, True, True, value)  # rlsa application
        canny_dilated_large = rlsa_result
        # cv2.imwrite('rlsah.png', rlsa_result)

    contours_large = cv2.findContours(
        canny_dilated_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_large = (
        contours_large[0] if len(contours_large) == 2 else contours_large[1]
    )

    # large_contour_img = image.copy()
    # large_contour_img = cv2.drawContours(
    #     large_contour_img, contours_large, -1, (0, 255, 0), 3
    # )
    # cv2.imwrite("large_contour_img.png", large_contour_img)

    bounding_boxes_large = np.array(
        [cv2.boundingRect(contour) for contour in contours_large]
    )

    if large_box_detection:
        contours_small = cv2.findContours(
            canny_dilated_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_small = (
            contours_small[0] if len(contours_small) == 2 else contours_small[1]
        )

        # small_contour_img = image.copy()
        # small_contour_img = cv2.drawContours(
        #     small_contour_img, contours_small, -1, (0, 255, 0), 3
        # )
        # cv2.imwrite("small_contour_img.png", small_contour_img)

    max_area = int(figure_max_area_percentage * image_area)
    min_area = (image_height // 3) * (image_width // 6)
    min_area_small = min_area

    padding = image_height // 70

    figures = []
    all_figure_boxes = []
    output_paths = []

    if large_box_detection:
        # none_tested = True
        for contour in contours_small:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)

            # Figure has 4 corners and it is convex
            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and min_area_small < cv2.contourArea(approx) < max_area
            ):
                # none_tested = False
                # min_area_small = cv2.contourArea(approx)
                figure_contour = approx[:, 0]

                # if not none_tested:
                bounding_box = cv2.boundingRect(figure_contour)
                x, y, w, h = bounding_box
                figure = original[
                    y - padding : y + h + padding, x - padding : x + w + padding
                ]
                figures.append(figure)
                all_figure_boxes.append(convert_coords_to_corners(bounding_box))

    for box in bounding_boxes_large:
        x, y, w, h = box
        area = w * h
        aspect_ratio = w / h
        if min_area < area < max_area and 0.2 < aspect_ratio < 6:
            # Draw bounding box rectangle, crop using numpy slicing
            roi_rectangle = convert_coords_to_corners(box)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.imwrite("rect.png", image)
            if (
                y + h >= image_height
                or x + w >= image_width
                or y <= image_height
                or x <= image_width
            ):
                potential_figure = original[y : y + h, x : x + w]
            else:
                potential_figure = original[
                    y - padding : y + h + padding, x - padding : x + w + padding
                ]
            # cv2.imwrite("potential_figure.png", potential_figure)

            # Go to next figure if the `potential_figure` is empty
            if potential_figure.size == 0:
                continue

            # Start all checks as passed (aka True). These lines ensure that if
            # the checks are intentionally disabled then the potential figure is
            # always added because `checks_passed` will be true.
            text_overlap_under_threshold = True
            roi_is_color = True

            if do_text_check:
                total_area_overlapped = sum(
                    area_of_overlapping_rectangles(roi_rectangle, text_rectangle)
                    for text_rectangle in text_bounding_boxes
                )
                logger.debug("Total area overlapped by text: %i", total_area_overlapped)
                text_overlap_under_threshold = (
                    total_area_overlapped < text_area_overlap_threshold * area
                )

            if do_color_check:
                roi_is_color = detect_color_image(potential_figure) == "color"

            checks_passed = roi_is_color and text_overlap_under_threshold

            if checks_passed:
                figures.append(potential_figure)
                all_figure_boxes.append(roi_rectangle)

    if do_remove_subfigures:
        remove_idxs = []
        for idx, figure in enumerate(all_figure_boxes):
            for compare_idx, figure_to_compare in enumerate(
                all_figure_boxes[idx + 1 :]
            ):
                overlapping_area = area_of_overlapping_rectangles(
                    figure, figure_to_compare
                )
                if overlapping_area > 0:
                    figure_area = area_of_corner_box(figure)
                    figure_to_compare_area = area_of_corner_box(figure_to_compare)
                    if figure_area > figure_to_compare_area:
                        remove_idxs.append(compare_idx)
                    else:
                        remove_idxs.append(idx)

        figures = [
            figure for idx, figure in enumerate(figures) if idx not in remove_idxs
        ]

    for idx, figure in enumerate(figures):
        if entropy_check:
            # If `entropy_check` is a boolean, then set it to the default
            if type(entropy_check) is bool and entropy_check:
                entropy_check = 2.5
            try:
                gray = cv2.cvtColor(figure, cv2.COLOR_BGR2GRAY)
            except:
                continue
            high_entropy = shannon_entropy(gray) > entropy_check
            if not high_entropy:
                continue

        full_output_path = start_output_path + str(idx) + ext
        output_paths.append(full_output_path)
        cv2.imwrite(full_output_path, figure)

    logger.debug("Number of Figures Detected: %i", len(figures))
    return figures, output_paths


def all_in_folder(
    path,
    remove_original=False,
    east="frozen_east_text_detection.pb",
    do_text_check=True,
    **kwargs
):
    """
    Perform figure detection on every file in folder and return new paths.
    ``**kwargs`` is passed to :meth:`~lecture2notes.end_to_end.figure_detection.detect_figures`.
    """
    figure_paths = []
    images = os.listdir(path)
    images.sort()

    if do_text_check:
        east = load_east(east)

    for item in tqdm(images, total=len(images), desc="> Figure Detection: Progress"):
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path) and OUTPUT_PATH_MODIFIER not in str(
            current_path
        ):
            # Above checks that file exists and does not contain `OUTPUT_PATH_MODIFIER` because that would
            # indicate that the file has already been processed.
            _, output_paths = detect_figures(
                current_path, east=east, do_text_check=do_text_check, **kwargs
            )

            figure_paths.extend(output_paths)
            if remove_original:
                os.remove(current_path)
    logger.debug("> Figure Detection: Returning figure paths")
    return figure_paths


def add_figures_to_ssa(ssa, figures_path):
    # If the SSA contains frame numbers
    if ssa and "frame_number" in ssa[0].keys():
        mapping = frame_number_filename_mapping(figures_path)

        for idx, slide in enumerate(ssa):
            current_slide_idx = slide["frame_number"]
            try:
                ssa[idx]["figure_paths"] = mapping[current_slide_idx]
            except KeyError:  # Ignore frames that have no figures
                pass

    return ssa


# import matplotlib.pyplot as plt
# all_in_folder("delete/")
# detect_figures("delete/img_01054_noborder.jpg")
# detect_figures("g-yPqNmrgYw-img_146.jpg", east="lecture2notes/end_to_end/frozen_east_text_detection.pb")
