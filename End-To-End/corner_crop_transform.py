# Based on: 
# https://github.com/Breta01/handwriting-ocr/blob/master/notebooks/page_detection.ipynb

import os
import numpy as np
from tqdm import tqdm
import cv2
import logging

logger = logging.getLogger(__name__)

# TODO: Figure out what `SMALL_HEIGHT` does
SMALL_HEIGHT = 800

OUTPUT_PATH_MODIFIER = "_cropped"

def resize(img, height=SMALL_HEIGHT, allways=False):
    """Resize image to given height."""
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    
    return img

def edges_det(img, min_val, max_val):
    """ Preprocessing (gray, thresh, filter, border) + Canny edge detection """
    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)

    # Applying blur and threshold
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    img = cv2.medianBlur(img, 11)

    # Add black border - detection of border touching pages
    # Contour can't touch side of image
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cv2.Canny(img, min_val, max_val)


def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contour_offset(cnt, offset):
    """ Offset contour because of 5px border """
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def find_page_contours(edges, img, min_area_mult=0.5):
    """ Finding corner points of page contour """
    # Getting contours  
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * min_area_mult
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            
            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]

    # Sort corners and offset them
    page_contour = four_corners_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))

def persp_transform(img, s_points):
    """ Transform perspective from start points to target points """
    # Euclidean distance - calculate maximum height and width
    height = max(np.linalg.norm(s_points[0] - s_points[1]),
                 np.linalg.norm(s_points[2] - s_points[3]))
    width = max(np.linalg.norm(s_points[1] - s_points[2]),
                 np.linalg.norm(s_points[3] - s_points[0]))
    
    # Create target points
    t_points = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)
    
    # getPerspectiveTransform() needs float32
    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)
    
    M = cv2.getPerspectiveTransform(s_points, t_points) 
    return cv2.warpPerspective(img, M, (int(width), int(height)))
    
def crop(img_path, output_path=None, debug_output_imgs=False):
    img = cv2.imread(img_path)

    edges_img = edges_det(img, 300, 250)
    edges_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, np.ones((5, 11)))

    page_contour = find_page_contours(edges_img, resize(img), min_area_mult=0.02)

    if debug_output_imgs:
        cv2.imwrite('edges_img.jpg', edges_img)
        cv2.imwrite("contours.jpg", cv2.drawContours(resize(img), [page_contour], -1, (0, 255, 0), 3))

    # page_contour = page_contour.dot(ratio(img))

    img_cropped = persp_transform(img, page_contour)

    if not output_path:
        file_parse = os.path.splitext(str(img_path))
        filename = file_parse[0]
        ext = file_parse[1]
        output_path = filename + OUTPUT_PATH_MODIFIER + ext

    cv2.imwrite(output_path, img_cropped)
    return output_path

def all_in_folder(path, remove_original=False):
    """Perform perspective cropping on every file in folder and return new paths"""
    cropped_imgs_paths = []
    images = os.listdir(path)
    images.sort()
    for item in tqdm(images, total=len(images), desc="> Corner Crop Transform: Progress"):
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path) and OUTPUT_PATH_MODIFIER not in str(current_path):
            # Above checks that file exists and does not contain `OUTPUT_PATH_MODIFIER` because that would 
            # indicate that the file has already been cropped. See crop().
            output_path = crop(current_path)
            cropped_imgs_paths.append(output_path)
            if remove_original:
                os.remove(current_path)
    logger.debug("> Corner Crop Transform: Returning cropped image paths")
    return cropped_imgs_paths