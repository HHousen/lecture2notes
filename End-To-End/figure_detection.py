import os
from tqdm import tqdm
import logging
import numpy as np
import cv2
from PIL import Image, ImageStat
from text_detection import get_text_bounding_boxes, load_east

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
        else:
            return "color"
    elif len(bands) == 1:
        return "b&w"
    else:
        return "unknown"


def detect_figures(
    image_path,
    output_path=None,
    east="frozen_east_text_detection.pb",
    text_area_overlap_threshold=0.20,
    figure_max_area_percentage=0.70,
    text_max_area_percentage=0.30,
    do_color_check=True,
    do_text_check=True,
):
    """Detect figures located in a slide.

    Args:
        image_path (str): Path to the image to process.
        output_path (str, optional): Path to save the figures. Defaults to 
            ``[filename]_figure_[index].[ext]``.
        east (str or cv2.dnn_Net, optional): Path to the EAST model file or the pre-trained 
            EAST model loaded with :meth:`~text_detection.load_east`. ``do_text_check`` must 
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
        do_color_check (bool, optional): Check that potential figures contain color. This 
            helps to remove large quantities of black and white text form the potential 
            figure list. Defaults to True.
        do_text_check (bool, optional): Check that only `text_area_overlap_threshold` of 
            potential figures contains text. This is useful to remove blocks of text that 
            are mistakenly classified as figures. Checking for text increases processing 
            time so be careful if processing a large number of files. Defaults to True.

    Returns:
        tuple: (figures, output_paths) A list of figures extracted from the input slide image 
        and a list of paths to those figures on disk.
    """
    image = cv2.imread(image_path)

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
            if (box[0] - box[2]) * (box[1] - box[3])  # area of box
            < text_max_area_percentage * image_area
        ]

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    canny = cv2.Canny(blurred, 120, 255, 1)
    canny_dilated = cv2.dilate(canny, np.ones((25, 25), dtype=np.uint8))

    cnts = cv2.findContours(canny_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    bounding_boxes = np.array([cv2.boundingRect(contour) for contour in cnts])

    max_area = int(figure_max_area_percentage * image_area)

    padding = image_height // 70

    figures = []
    output_paths = []
    for box in bounding_boxes:
        x, y, w, h = box
        area = w * h
        if (image_height // 3 * image_width // 6) < area < max_area:
            # Draw bounding box rectangle, crop using numpy slicing
            x_values = (x + w, x)
            y_values = (y + h, y)
            roi_rectangle = (max(x_values), max(y_values), min(x_values), min(y_values))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            potential_figure = original[
                y - padding : y + h + padding, x - padding : x + w + padding
            ]

            # Start all checks as passed (aka True). These lines ensure that if
            # the checks are intentionally disabled then the potential figure is
            # always added because `checks_passed` will be true.
            text_overlap_under_threshold = True
            roi_is_color = True

            if do_text_check:
                total_area_overlapped = sum(
                    [
                        area_of_overlapping_rectangles(roi_rectangle, text_rectangle)
                        for text_rectangle in text_bounding_boxes
                    ]
                )
                logger.debug("Total area overlapped by text: %i", total_area_overlapped)
                text_overlap_under_threshold = (
                    total_area_overlapped < text_area_overlap_threshold * area
                )

            if do_color_check:
                roi_is_color = detect_color_image(potential_figure) == "color"

            checks_passed = roi_is_color and text_overlap_under_threshold

            if checks_passed:
                full_output_path = start_output_path + "{}.{}".format(len(figures), ext)
                output_paths.append(full_output_path)
                cv2.imwrite(full_output_path, potential_figure)
                figures.append(potential_figure)

    logger.info("Number of Figures Detected: {}", len(figures))
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
    ``**kwargs`` is passed to :meth:`~figure_detection.detect_figures`.
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


# all_in_folder("delete/")
# detect_figures("delete/o7h_sYMk_oc-img_143.jpg")
