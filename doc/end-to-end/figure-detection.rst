.. _figure_detection:

Figure Detection Algorithm
==========================

The figure extraction algorithm identifies and saves images, charts, tables, and diagrams from slide frames so that they can be shown in the final summary. Two sub-algorithms are used during figure extraction, each of which specializes in identifying particular figures. The ``large box detection`` algorithm identifies images that have a border, such as tables, by performing canny edge detection, applying a small dilation, and searching for rectangular contours that meet a size threshold. The ``large dilation detection`` algorithm performs a very large dilation on the canny edge map, computes contours, and finally approximates bounding boxes that meet a size and aspect ratio threshold. This algorithm specializes in locating images without borders since ``large box detection`` will not detect contours that do not closely resemble rectangles.

Text and color checks checks are applied as part of the ``large dilation detection`` algorithm. For each potential figure, the text check calculates the area of text within the figure. Before identifying any figures, the bounding boxes of text within the image are determined using the EAST (Efficient and Accurate Scene Text Detector) text detection algorithm. Then, the overlapping area between the potential figure and each text bounding box is calculated and summed. If the area of the text is lower than a percentage of the total potential figure area, then the check passes. The color check simply checks if the image contains color even if it has red, green, and blue color bands by computing the mean of squared errors.

Finally, two checks are applied to all potential figures, regardless of which algorithm proposed them. The first is an overlapping area check. The overlapping area between every combination of two potential figures is calculated. If one figure overlaps another, then the larger figure is kept since it likely contains the smaller one. The second check ensures the complexity of the figure is above a threshold by calculating Shannon Entropy.

Extracted figures are attached to their respective slide in the slide structure analysis.

You can learn more about figure detection in the API documentation. Please see :meth:`lecture2notes.end_to_end.figure_detection.detect_figures`
