# Based on:
# 1. https://github.com/Breta01/handwriting-ocr/blob/master/notebooks/page_detection.ipynb
#       Blog Post: https://bretahajek.com/2017/01/scanning-documents-photos-opencv/
# 2. https://stackoverflow.com/a/44454619

import argparse
import logging
import os

import cv2
import imageio
import numpy as np
from pygifsicle import optimize
from tqdm import tqdm

from .helpers import make_dir_if_not_exist

logger = logging.getLogger(__name__)

OUTPUT_PATH_MODIFIER = "_cropped"


def resize(img, height=800, allways=False):
    """Resize image to given height."""
    if img.shape[0] > height or allways:
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

    return img


def find_intersection(line1, line2):
    """Find the intersection between ``line1`` and ``line2``."""
    # Extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # Compute determinant
    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    return Px, Py


def segment_lines(lines, delta):
    """Groups lines from ``cv2.HoughLinesP`` into vertical and horizontal bins.

    Args:
        lines (list): the data returned from ``cv2.HoughLinesP``
        delta (int): how far away the x and y coordinates can differ before they're marked as different lines

    Returns:
        [tuple]: (h_lines, v_lines) the horizontal and vertical lines, respectively. each line in each list is formatted as (x1, y1, x2, y2).
    """
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2 - x1) < delta:  # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2 - y1) < delta:  # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines


def cluster_points(points, nclusters):
    """
    Perform KMeans clustering (using ``cv2.kmeans``) on ``points``, creating ``nclusters`` clusters.
    Returns the centroids of the clusters.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(
        points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    return centers


def remove_contours(edges, contour_removal_threshold):
    """Remove contours from an edge map by deleting contours shorter than ``contour_removal_threshold``."""
    result = np.full_like(edges, 0, dtype="uint8")
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        cnt for cnt in contours if cv2.arcLength(cnt, True) > contour_removal_threshold
    ]

    for c in contours:
        result = cv2.drawContours(result, [c], -1, (255, 255, 255), 2)

    return result


def hough_lines_corners(
    img, edges_img, min_line_length, border_size=11, debug_output_imgs=None
):
    """Uses ``cv2.HoughLinesP`` to find horizontal and vertical lines, finds the intersection
        points, and finally clusters those points using KMeans.

    Args:
        img (image): the image as loaded by ``cv2.imread``.
        edges_img (image): edges extracted from ``img`` by :meth:`~lecture2notes.end_to_end.corner_crop_transform.edges_det`.
        min_line_length (int): the shortest line length to consider as a valid line
        border_size (int, optional): the size of the borders added by :meth:`~lecture2notes.end_to_end.corner_crop_transform.edges_det`. Defaults to 11.
        debug_output_imgs (dict, optional): modifies this dictionary by adding ``(image file name, image data)`` pairs. Defaults to None.

    Returns:
        [list]: The corner coordinates as sorted by :meth:`~lecture2notes.end_to_end.corner_crop_transform.four_corners_sort`.
    """
    # Remove `border_size` border from edges_det
    edges_img = edges_img[border_size:-border_size, border_size:-border_size]
    height = img.shape[0]

    # Collect and segment the lines
    lines = cv2.HoughLinesP(
        edges_img,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        maxLineGap=int(height / 54),
        minLineLength=min_line_length,
    )
    if lines is None:
        logger.debug(
            "Hough Lines Corners Failed! `cv2.HoughLinesP` failed to find any lines."
        )
        return None
    h_lines, v_lines = segment_lines(lines, int(height / 54))

    # Find the line intersection points
    Px = []
    Py = []
    for h_line in h_lines:
        for v_line in v_lines:
            px, py = find_intersection(h_line, v_line)
            Px.append(px)
            Py.append(py)

    if len(Px) <= 4:  # there are not at least 4 intersection points
        logger.debug(
            "Hough Lines Corners Failed! Only "
            + str(len(Px))
            + " intersection points found."
        )
        return None

    # Use clustering to find the centers of the point clusters
    P = np.float32(np.column_stack((Px, Py)))
    nclusters = 4  # four corners for a rectangle
    centers = cluster_points(P, nclusters)

    if debug_output_imgs is not None:
        # Draw the segmented lines
        hough_img = img.copy()
        for line in h_lines:
            for x1, y1, x2, y2 in line:
                color = [0, 0, 255]  # color hoz lines red
                cv2.line(hough_img, (x1, y1), (x2, y2), color=color, thickness=1)
        for line in v_lines:
            for x1, y1, x2, y2 in line:
                color = [255, 0, 0]  # color vert lines blue
                cv2.line(hough_img, (x1, y1), (x2, y2), color=color, thickness=1)

        # Draw the intersection points
        intersects_img = img.copy()
        for cx, cy in zip(Px, Py):
            cx = np.round(cx).astype(int)
            cy = np.round(cy).astype(int)
            color = np.random.randint(0, 255, 3).tolist()  # random colors
            cv2.circle(
                intersects_img, (cx, cy), radius=2, color=color, thickness=-1
            )  # -1: filled circle

        # Draw the center of the clusters
        corners_img = img.copy()
        for cx, cy in centers:
            cx = np.round(cx).astype(int)
            cy = np.round(cy).astype(int)
            cv2.circle(
                corners_img, (cx, cy), radius=4, color=[0, 0, 255], thickness=-1
            )  # -1: filled circle

        debug_output_imgs.update(
            {
                "6-hough.jpg": hough_img,
                "7-intersections.jpg": intersects_img,
                "8-corners.jpg": corners_img,
            }
        )

    return four_corners_sort(centers)


def horizontal_vertical_edges_det(img, thresh_blurred, debug_output_imgs=None):
    """Detects horizontal and vertical edges and merges them together.

    Args:
        img (image): the image as provided by ``cv2.imread``
        thresh_blurred (image): the image processed by thresholding. see :meth:`~lecture2notes.end_to_end.corner_crop_transform.edges_det`.
        debug_output_imgs (dict, optional): modifies this dictionary by adding ``(image file name, image data)`` pairs. Defaults to None.

    Returns:
        [image]: result image with a black background and white edges
    """
    # Defining a kernel length
    cols = np.array(img).shape[1]
    horizontal_kernel_length = cols // 15
    rows = np.array(img).shape[0]
    vertical_kernel_length = rows // 15

    result = cv2.cvtColor(np.full_like(img, 0), cv2.COLOR_BGR2GRAY)

    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_kernel_length)
    )
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_kernel_length, 1)
    )

    detect_vertical = cv2.morphologyEx(
        thresh_blurred, cv2.MORPH_OPEN, vertical_kernel, iterations=3
    )
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        result = cv2.drawContours(result, [c], -1, (255, 255, 255), 2)

    detect_horizontal = cv2.morphologyEx(
        thresh_blurred, cv2.MORPH_OPEN, horizontal_kernel, iterations=3
    )
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        result = cv2.drawContours(result, [c], -1, (255, 255, 255), 2)

    if debug_output_imgs is not None:
        cv2.imwrite("debug_imgs/horizontal_vertical_edges.jpg", result)

    return result


def edges_det(img, min_val, max_val, debug_output_imgs=None):
    """Preprocessing (gray, thresh, filter, border) & Canny edge detection

    Args:
        img (image): the image loaded using ``cv2.imread``.
        min_val (int): minimum value for ``cv2.Canny``.
        max_val (int): maximum value for ``cv2.Canny``.
        debug_output_imgs (dict, optional): modifies this dictionary by adding ``(image file name, image data)`` pairs. Defaults to None.

    Returns:
        [tuple]: (dilated, total_border), dialted edges and total border width added
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying blur and threshold
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    # thresh = cv2.adaptiveThreshold(gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    thresh = cv2.threshold(gray_filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
        1
    ]

    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    thresh_blurred = cv2.medianBlur(thresh, 11)

    # horizontal_vertical_edges_det(img, thresh_blurred, debug_output_imgs)

    # Add border - detection of border touching boxes
    # Contour can't touch side of image
    border_amount = 5
    img_border = cv2.copyMakeBorder(
        thresh_blurred,
        border_amount,
        border_amount,
        border_amount,
        border_amount,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    edges = cv2.Canny(img_border, min_val, max_val)

    # Dilate to make the edges wider: "This will make the lines thicker which
    # will help fit the Hough lines better. What the Hough lines function does
    # in the abstract is basically make a grid of lines passing through a ton of
    # angles and distances, and if the lines go over any white pixels from Canny,
    # then it gives that line a score for each point it goes through. However, the
    # lines from Canny won't be perfectly straight, so you'll get a few different
    # lines scoring. Making those Canny lines thicker will mean each line that is
    # really close to fitting well will have better chances of scoring higher."
    # Source: https://stackoverflow.com/a/44454619
    dilation_amount = 6
    dilated = cv2.dilate(
        edges, np.ones((dilation_amount, dilation_amount), dtype=np.uint8)
    )

    if debug_output_imgs is not None:
        debug_output_imgs.update(
            {
                "1-edges_det_threshold.jpg": thresh_blurred,
                "2-edges_det_canny_dilated.jpg": dilated,
            }
        )

    total_border = border_amount + dilation_amount

    return dilated, total_border


def four_corners_sort(pts):
    """Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array(
        [
            pts[np.argmin(summ)],
            pts[np.argmax(diff)],
            pts[np.argmax(summ)],
            pts[np.argmin(diff)],
        ]
    )


def contour_offset(cnt, offset):
    """Offset contour because of 5px border"""
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def straight_lines_in_contour(contour, delta=100):
    """
    Returns True if ``contour`` contains lines that are horizontal or vertical.
    ``delta`` allows the lines to tilt by a certain number of pixels. For instance, if a line is vertical, its y values can change by ``delta`` pixels before it is considered not vertical.
    """
    rectangle = True

    contour = four_corners_sort(contour)
    y_line1 = [contour[0], contour[1]]
    y_line2 = [contour[3], contour[2]]
    x_line1 = [contour[0], contour[3]]
    x_line2 = [contour[1], contour[2]]
    lines = [y_line1, y_line2, x_line1, x_line2]
    for (x1, y1), (x2, y2) in lines:
        if not (abs(x2 - x1) < delta or abs(y2 - y1) < delta):
            rectangle = False
            logger.debug("Contour is not a rectangle based on line straightness.")

    return rectangle


def find_page_contours(
    edges, img, border_size=11, min_area_mult=0.3, debug_output_imgs=None
):
    """Find corner points of page contour

    Args:
        edges (image): edges extracted from ``img`` by :meth:`~lecture2notes.end_to_end.corner_crop_transform.edges_det`.
        img (image): the image loaded by ``cv2.imread``.
        border_size (int, optional): the size of the borders added by :meth:`~lecture2notes.end_to_end.corner_crop_transform.edges_det`. Defaults to 11.
        min_area_mult (float, optional): the minimum percentage of the image area that a contour's
            area must be greater than to be considered as the slide. Defaults to 0.5.

    Returns:
        [contour or NoneType]: ``contour`` is the set of coordinates of the corners sorted
            by :meth:`~lecture2notes.end_to_end.corner_crop_transform.four_corners_sort` or returns None when no contour
            meets the criteria.
    """
    # Getting contours
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * min_area_mult
    # Maximum area is 5 less than the original image dimensions
    MAX_COUNTOUR_AREA = (width - border_size - 5) * (height - border_size - 5)

    min_area = MIN_COUNTOUR_AREA
    page_contour = None
    none_tested = True

    # loop through the contours and select the largest one that is less than MAX_COUNTOUR_AREA
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)

        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h

        # Page has 4 corners and it is convex
        if (
            len(approx) == 4
            and cv2.isContourConvex(approx)
            and min_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA
            and straight_lines_in_contour(approx[:, 0], delta=int(height / 10))
            and 0.9 < aspect_ratio < 3
        ):

            none_tested = False
            min_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]

    if none_tested:
        logger.debug("No contours met the criteria.")
        return None

    # Sort corners and offset them
    page_contour = four_corners_sort(page_contour)

    if debug_output_imgs is not None:
        page_contour_img = img.copy()
        page_contour_img = cv2.drawContours(
            page_contour_img, [page_contour], -1, (0, 255, 0), 3
        )

        debug_output_imgs.update({"5-page_contour.jpg": page_contour_img})

    return contour_offset(page_contour, (-border_size, -border_size))


def persp_transform(img, s_points):
    """Transform perspective of ``img`` from start points to target points."""
    # Euclidean distance - calculate maximum height and width
    height = max(
        np.linalg.norm(s_points[0] - s_points[1]),
        np.linalg.norm(s_points[2] - s_points[3]),
    )
    width = max(
        np.linalg.norm(s_points[1] - s_points[2]),
        np.linalg.norm(s_points[3] - s_points[0]),
    )

    # Create target points
    t_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)

    M = cv2.getPerspectiveTransform(s_points, t_points)
    return cv2.warpPerspective(img, M, (int(width), int(height)))


def write_debug_imgs(debug_output_imgs, base_path="debug_imgs"):
    """Saves images from ``debug_output_imgs`` to disk in ``base_path``.

    Args:
        debug_output_imgs (dict): dictionary in format {image file name: image data}
        base_path (str, optional): the directory to store the debug images. Defaults to "debug_imgs".
    """
    for filename, img_data in debug_output_imgs.items():
        save_path = os.path.join(base_path, filename)
        cv2.imwrite(save_path, img_data)


def crop(
    img_path,
    output_path=None,
    mode="automatic",
    debug_output_imgs=False,
    save_debug_imgs=False,
    create_debug_gif=False,
    debug_gif_optimize=True,
    debug_path="debug_imgs",
):
    """Main method to perspective crop an image to the slide.

    Args:
        img_path (str): path to the image to load
        output_path (str, optional): path to save the image. Defaults to ``[filename]_cropped.[ext]``.
        mode (str, optional): There are three modes available. Defaults to "automatic".

            * ``contours``: uses :meth:`~lecture2notes.end_to_end.corner_crop_transform.find_page_contours` to extract contours from an edge map of the image. is ineffective if there are any gaps or obstructions in the outline around the slide.
            * ``hough_lines``: uses :meth:`~lecture2notes.end_to_end.corner_crop_transform.hough_lines_corners` to get corners by looking for horizontal and vertical lines, finding the intersection points, and clustering the intersection points.
            * ``automatic``: tries to use ``contours`` and falls back to ``hough_lines`` if ``contours`` reports a failure.

        debug_output_imgs (bool or dict, optional): if dictionary, modifies the dictionary by adding ``(image file name, image data)`` pairs. if boolean and True, creates a dictionary in the same way as if a dictionary was passed. Defaults to False.
        save_debug_imgs (bool, optional): uses :meth:`~lecture2notes.end_to_end.corner_crop_transform.write_debug_imgs` to save the debug_output_imgs to disk. Requires ``debug_output_imgs`` to not be False. Defaults to False.
        create_debug_gif (bool, optional): create a gif of the debug images. Requires ``debug_output_imgs`` to not be False. Defaults to False.
        debug_gif_optimize (bool, optional): optimize the gif produced by enabling the ``create_debug_gif`` option using ``pygifsicle``. Defaults to True.
        debug_path (str, optional): location to save the debug images and debug gif. Defaults to "debug_imgs".

    Returns:
        [tuple]: path to cropped image and failed (True if no slide bounding box found, false otherwise)
    """
    if mode not in ["automatic", "contours", "hough_lines"]:
        raise AssertionError

    if not debug_output_imgs:
        debug_output_imgs = None
    else:
        debug_output_imgs = {}

    img = cv2.imread(img_path)

    edges_img, total_border = edges_det(img, 200, 50, debug_output_imgs)

    # Remove contours that are not longer than a third of the height of the image.
    # This should remove countors from words on the slides.
    rows = np.array(img).shape[0]
    contour_removal_threshold = rows // 3
    edges_no_words = remove_contours(edges_img, contour_removal_threshold)

    edges_morphed = cv2.morphologyEx(edges_no_words, cv2.MORPH_CLOSE, np.ones((5, 11)))

    if debug_output_imgs is not None:
        debug_output_imgs.update(
            {
                "3-edges_no_words.jpg": edges_no_words,
                "4-edges_morphed.jpg": edges_morphed,
            }
        )

    if mode == "contours":
        page_contour = find_page_contours(
            edges_morphed, img, total_border, debug_output_imgs=debug_output_imgs
        )
        if page_contour is None:
            logger.warn(
                "Contours method failed. Using entire image. It is recommended to try 'automatic' mode."
            )
        corners = page_contour

    elif mode == "hough_lines":
        corners = hough_lines_corners(
            img,
            edges_morphed,
            contour_removal_threshold,
            total_border,
            debug_output_imgs,
        )

    elif mode == "automatic":
        page_contour = find_page_contours(
            edges_morphed, img, total_border, debug_output_imgs=debug_output_imgs
        )
        if page_contour is None:
            logger.debug("Contours method failed. Using `hough_lines_corners`.")
            corners = hough_lines_corners(
                img,
                edges_morphed,
                contour_removal_threshold,
                total_border,
                debug_output_imgs,
            )
        else:
            corners = page_contour

    # Detection of slide bounding box succeeded
    if corners is not None:
        failed = False
        img_cropped = persp_transform(img, corners)
    # Detection did not succeed so return original image
    else:
        failed = True
        img_cropped = img

    if not output_path:
        file_parse = os.path.splitext(str(img_path))
        filename = file_parse[0]
        ext = file_parse[1]
        output_path = filename + OUTPUT_PATH_MODIFIER + ext

    if debug_output_imgs is not None:
        debug_output_imgs.update({"9-img_cropped.jpg": img_cropped})

        if save_debug_imgs:
            write_debug_imgs(debug_output_imgs, base_path=debug_path)

        if create_debug_gif:
            # Get original image size
            scale_to = img.shape[:2]
            # Swap the axes
            scale_to = (scale_to[1], scale_to[0])
            # Create list of images
            imgs_list = list(debug_output_imgs.values())
            # Scale each debug image to the dimensions of the original image.
            for idx, img in enumerate(imgs_list[1:]):
                img = cv2.resize(img, scale_to)
                imgs_list[idx + 1] = img

            original_file_name = os.path.basename(str(img_path))
            filename = original_file_name + "_debug.gif"
            save_path = os.path.join(debug_path, filename)
            imageio.mimsave(save_path, imgs_list, duration=1.4)

            if debug_gif_optimize:
                optimize(save_path)

    cv2.imwrite(output_path, img_cropped)

    return output_path, failed


def all_in_folder(path, remove_original=False, **kwargs):
    """
    Perform perspective cropping on every file in folder and return new paths.
    ``**kwargs`` is passed to :meth:`~lecture2notes.end_to_end.corner_crop_transform.crop`.
    """
    cropped_imgs_paths = []
    images = os.listdir(path)
    images.sort()
    for item in tqdm(
        images, total=len(images), desc="> Corner Crop Transform: Progress"
    ):
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path) and OUTPUT_PATH_MODIFIER not in str(
            current_path
        ):
            # Above checks that file exists and does not contain `OUTPUT_PATH_MODIFIER` because that would
            # indicate that the file has already been cropped. See `crop()``.
            output_path, failed = crop(current_path, **kwargs)

            # if the crop failed due to detection issues debug mode enabled ("debug_output_imgs" was passed in `**kwargs`)
            if failed and "debug_output_imgs" in locals()["kwargs"]:
                logger.info("Failed: " + str(output_path))
                with open("debug_crop_error_log.txt", "a+") as error_log:
                    error_log.write(output_path + "\n")

            cropped_imgs_paths.append(output_path)
            if remove_original:
                os.remove(current_path)
    logger.debug("> Corner Crop Transform: Returning cropped image paths")
    return cropped_imgs_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perspective Crop to Rectangles (Slides)"
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["file", "folder"],
        help="`file` mode will crop a single image and `folder` mode will crop all the images in a given folder",
    )
    parser.add_argument(
        "path", type=str, help="path to file or folder (depending on `mode`) to process"
    )
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        help="enable the usage of `--debug_imgs`, `--debug_gif`, and `--debug_path`.",
    )
    parser.add_argument(
        "-di",
        "--debug_imgs",
        action="store_true",
        help="Save debug images (JPG of each step of the pipeline). Requires `--debug_mode` to be enabled.",
    )
    parser.add_argument(
        "-dg",
        "--debug_gif",
        action="store_true",
        help="Save debug gif (GIF with 1.4s delay between each debug image). Requires `--debug_mode` to be enabled.",
    )
    parser.add_argument(
        "-dgo",
        "--debug_gif_optimize",
        action="store_true",
        help="Optimize the gif produced by enabling --debug_gif with `gifsicle`.",
    )
    parser.add_argument(
        "-p",
        "--debug_path",
        type=str,
        default="./debug_imgs",
        help="path to folder to store debug images (default: './debug_imgs')",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    args = parser.parse_args()

    if args.debug_gif and not args.debug_mode:
        parser.error("`--debug_gif` requires `--debug_mode` to be enabled.")
    if args.debug_imgs and not args.debug_mode:
        parser.error("`--debug_imgs` requires `--debug_mode` to be enabled.")

    make_dir_if_not_exist(args.debug_path)

    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(args.logLevel),
    )

    if args.mode == "file":
        crop(
            args.path,
            debug_output_imgs=args.debug_mode,
            save_debug_imgs=args.debug_imgs,
            create_debug_gif=args.debug_gif,
            debug_gif_optimize=args.debug_gif_optimize,
            debug_path=args.debug_path,
        )
    else:
        cropped_imgs_paths = all_in_folder(
            args.path,
            debug_output_imgs=args.debug_mode,
            save_debug_imgs=args.debug_imgs,
            create_debug_gif=args.debug_gif,
            debug_gif_optimize=args.debug_gif_optimize,
            debug_path=args.debug_path,
        )
