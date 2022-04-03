# Copied from https://github.com/Vasistareddy/pythonRLSA/blob/7beedfb5949c47a19ab32d39181ea2751379c5a6/pythonRLSA/rlsa.py

import numpy
from typing import Tuple

def iteration(image: numpy.ndarray, value: int) -> numpy.ndarray:
    """
    This method iterates over the provided image by converting 255's to 0's if the number of consecutive 255's are
    less the "value" provided

    Parameter
    image numpy.ndarray
    value int
    """

    rows, cols = image.shape
    for row in range(0,rows):
        try:
            start = image[row].tolist().index(0) # to start the conversion from the 0 pixel
        except ValueError:
            start = 0 # if '0' is not present in that row

        count = start
        for col in range(start, cols):
            if image[row, col] == 0:
                if (col-count) <= value and (col-count) > 0:
                    image[row, count:col] = 0               
                count = col  
    return image 

def valueChecker(value) -> Tuple[int, int]:
    """
    This function checks the user provided value and assign it for horizontal and vertical operations

    Parameters
    value int/tuplePair/listPair
    """
    if type(value) in [tuple, list] and len(value) == 2:
        valueh = value[0]
        valuev = value[1]
    elif type(value) in [tuple, list] and len(value) == 1:
        valueh = value[0]
        valuev = value[0]
    elif type(value) is int:
        valueh = valuev = value
    elif type(value) is float:
        valueh = valuev = int(value)
    else:
        valueh = valuev = 0
    valueh = int(valueh) if valueh > 0 else 0 # consecutive pixel position checker value to convert 255 to 0
    valuev = int(valuev) if valuev > 0 else 0 # consecutive pixel position checker value to convert 255 to 0
    return valueh, valuev

def rlsa(image: numpy.ndarray, horizontal: bool = True, vertical: bool = True, value: int = 0) -> numpy.ndarray:
    """
    The method rlsa(RUN LENGTH SMOOTHING ALGORITHM) is to extract the block-of-text or the Region-of-interest(ROI) from the
    document binary Image provided. Must pass binary image of ndarray type.

    Parameters:
    image numpy.ndarray
    horizontal bool
    vertical bool
    value int/tuplePair/listPair
    """
    valueh, valuev = valueChecker(value)
    if isinstance(image, numpy.ndarray): # image must be binary of ndarray type  
        try:
            # RUN LENGTH SMOOTHING ALGORITHM working horizontally on the image
            if horizontal:
                image = iteration(image, valueh)   

            # RUN LENGTH SMOOTHING ALGORITHM working vertically on the image
            if vertical:
                image = image.T
                image = iteration(image, valuev)
                image = image.T

        except (AttributeError, ValueError) as e:
            image = None
            print("ERROR: ", e, "\n")
            print('Image must be an numpy ndarray and must be in "binary". Use Opencv/PIL to convert the image to binary.\n')
            print("import cv2;\nimage=cv2.imread('path_of_the_image');\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);\n\
                (thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n")
            print("method usage --from pythonRLSA import rlsa;\nrlsa.rlsa(image_binary, True, False, 10)")
    else:
        print('Image must be an numpy ndarray and must be in binary')
        image = None
    return image

