import logging
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

OUTPUT_PATH_MODIFIER = "_noborder"


def does_image_have_border(image, gamma=5):
    """Detect if an image has a solid black border.

    Args:
        image (np.array): An image loaded using ``cv2.imread()``.
        gamma (int, optional): How far the pixel values can vary before they are
            considered not black. This is useful if the image contains
            noise. Defaults to 5.

    Returns:
        bool: If all pixel values on any or all sides of the image are
        black +/- ``gamma``.
    """
    if gamma == 0:
        gamma = 1

    left_edge = (image[:, 0] < gamma).all()
    right_edge = (image[:, -1] < gamma).all()
    top_edge = (image[0, :] < gamma).all()
    bottom_edge = (image[-1, :] < gamma).all()

    # If any of the edges contain all black pixels then the image has a border
    # on at least one side.
    if left_edge or right_edge or top_edge or bottom_edge:
        return True
    return False


def detect_solid_color(image, gamma=5):
    """Detect if an image contains a solid color.

    Args:
        image (np.array): An image loaded using ``cv2.imread()``.
        gamma (int, optional): How far the pixel values can vary before they are
            considered a different color. This is useful if the image contains
            noise. Defaults to 5.

    Returns:
        bool: If all pixel values are the same +/- ``gamma``.
    """
    flattened_image = np.ravel(image)
    # Check if all value in the 2D array are equal
    first_value = flattened_image[0]
    all_values_equal = ((first_value - gamma) < flattened_image).all() and (
        flattened_image < (first_value + gamma)
    ).all()
    # If all the values are equal (plus or minus gamma) then the image
    # contains a solid color
    return all_values_equal


def remove_border(image_path, output_path=None):
    """Remove a black border that may exist on any or all sides of an image.
    Inspired by https://stackoverflow.com/a/13539194.

    Args:
        image_path (str): Path to the image to have its borders removed.
        output_path (str, optional): Path to save the borderless image. Defaults to
            ``[filename]_noborder.[ext]``

    Returns:
        str: The path to the new borderless image.
    """
    image = cv2.imread(image_path)

    # Convert image to grayscale and make binary image for threshold value of 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Return none if the image is a solid color
    solid_color = detect_solid_color(gray)
    if solid_color:
        return None

    if not output_path:
        file_parse = os.path.splitext(str(image_path))
        filename = file_parse[0]
        ext = file_parse[1]
        output_path = filename + OUTPUT_PATH_MODIFIER + ext

    image_has_border = does_image_have_border(gray)
    if not image_has_border:
        shutil.copyfile(image_path, output_path)
        return output_path

    # border_amount = 10
    # image_border = cv2.copyMakeBorder(
    #     image,
    #     border_amount,
    #     border_amount,
    #     border_amount,
    #     border_amount,
    #     cv2.BORDER_CONSTANT,
    #     value=[0, 0, 0],
    # )

    # cv2.imwrite("gray.png", gray)
    blur_amount = 3
    blurred = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0)
    _, thresh = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)

    # cv2.imwrite("thresh.png", thresh)

    # Now find contours in it. There will be only one object, so find bounding rectangle for it.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # large_contour_img = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("large_contour_img.png", large_contour_img)

    height = image.shape[0]
    width = image.shape[1]
    min_area = height * width * 0.7
    # Maximum area is 15 less than the original image dimensions
    max_area = (width - 15) * (height - 15)
    final_contour = None
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)

        if (
            len(approx) == 4  # four corners
            and cv2.isContourConvex(approx)  # convex and not concave
            and min_area < cv2.contourArea(approx) < max_area  # get the largest contour
        ):
            min_area = cv2.contourArea(approx)
            final_contour = approx[:, 0]

    # If no `final_contour` is found then assume no border or that detection failed
    # and return the path of the original image
    if final_contour is None:
        return image_path

    x, y, w, h = cv2.boundingRect(final_contour)

    crop = image[
        y + blur_amount : y + h - blur_amount, x + blur_amount : x + w - blur_amount
    ]
    cv2.imwrite(output_path, crop)

    return output_path


def all_in_folder(path, remove_original=False, **kwargs):
    """
    Perform border removal on every file in folder and return new paths.
    ``**kwargs`` is passed to :meth:`~lecture2notes.end_to_end.border_removal.remove_border`.
    """
    removed_border_paths = []
    images = os.listdir(path)
    images.sort()

    for item in tqdm(images, total=len(images), desc="> Border Removal: Progress"):
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path) and OUTPUT_PATH_MODIFIER not in str(
            current_path
        ):
            # Above checks that file exists and does not contain `OUTPUT_PATH_MODIFIER` because that would
            # indicate that the file has already been processed.
            output_path = remove_border(current_path, **kwargs)
            # Skip to next image if output is None.
            if output_path is None:
                continue

            removed_border_paths.append(output_path)
            if remove_original:
                os.remove(current_path)
    logger.debug("> Border Removal: Returning removed border paths")
    return removed_border_paths


# remove_border("delete/img_00601.jpg")
