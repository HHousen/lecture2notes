# This code is a version of https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# that has been converted to functions and has some additions.

import logging
import time

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

logger = logging.getLogger(__name__)


def load_east(east_path="frozen_east_text_detection.pb"):
    """Load the pre-trained EAST model.

    Args:
        east_path (str, optional): Path to the EAST model file. Defaults to
            "frozen_east_text_detection.pb".
    """
    if type(east_path) is cv2.dnn_Net:
        return east_path
    logger.debug("Loading EAST text detector...")
    net = cv2.dnn.readNet(east_path)
    return net


def get_text_bounding_boxes(
    image, net, min_confidence=0.5, resized_width=320, resized_height=320
):
    """Determine the locations of text in an image.

    Args:
        image (np.array): The image to be processed.
        net (cv2.dnn_Net): The EAST model loaded with :meth:`~lecture2notes.end_to_end.text_detection.load_east`.
        min_confidence (float, optional): Minimum probability required to inspect a region. Defaults to 0.5.
        resized_width (int, optional): Resized image width (should be multiple of 32). Defaults to 320.
        resized_height (int, optional): Resized image height (should be multiple of 32). Defaults to 320.

    Returns:
        list: The coordinates of bounding boxes containing text.
    """
    if type(net) is str:
        net = load_east(net)

    # load the input image and grab the image dimensions
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (resized_width, resized_height)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    logger.debug("Text detection took %.6f seconds", end - start)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    scaled_boxes = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        scaled_boxes.append((endX, endY, startX, startY))

        # draw the bounding box on the image
        # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # cv2.imwrite("text_bounding_boxes.png", orig)

    return scaled_boxes


# show the output image
# cv2.imshow("Text Detection", orig)
# cv2.waitKey(0)
