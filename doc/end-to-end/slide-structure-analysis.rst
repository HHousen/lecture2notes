.. _slide_structure_analysis:

Slide Structure Analysis (SSA)
==============================

.. note:: The main function to perform a slide strucutre analysis is :meth:`lecture2notes.end_to_end.slide_structure_analysis.analyze_structure`

The SSA algorithm extracts formatted text from each slide using OCR, stroke width, and the height of text bounding boxes. The SSA process identifies the title as well as bold, normal, and small/footer text. First, the algorithm performs OCR using Tesseract which outputs text with metadata such as the block, paragraph, line, and word number. This is used to identify individual bullet points that may exist on the slide. Tesseract also provides the location and size of the bounding box for each identified word, which is used to determine the stroke width of each word. Next, the words are grouped into their respective lines and the text classification algorithm is applied. The result is saved line by line with the text and its predicted category. All Tesseract outputs are spell checked with `SymSpell <https://github.com/wolfgarbe/SymSpell>`_, a symmetric delete spelling correction algorithm.

The text classification algorithm identifies a line of text as ``bold`` if it has above-average stroke width or above-average height. A line will be identified as ``footer`` text if the stroke width is below average and the height is below average. It is worse to misidentify ``normal`` text as ``footer`` text than ``footer`` text as ``normal`` text because the former causes a loss of content in the created notes. If a line of text does not meet either of the aforementioned checks, it is classified as ``normal``. Both checks (height and stroke width) compare against their respective averages times a scaling factor.

The stroke width algorithm takes an input image, applies Ostu's threshold, computes the distance transformation of the image, identifies peaks in intensity, and returns the average of those peaks.

SSA Title Identification
------------------------

The SSA title identification algorithm determines the title of an input slide given the Tesseract OCR output and image data. The first paragraph will be classified as a title if it meets the following criteria:

1. The mean top y coordinate of the text bounding boxes is in the upper third of the image.
2. The mean of the x coordinate of the text bounding boxes is less than the 77\% of the image width.
3. The number of characters is greater than 3.
4. The mean stroke width of the paragraph is greater than the mean stroke width of all the content on the slide plus one standard deviation.
5. The mean bounding box height of the paragraph is greater than the mean high of all the content on the slide.

If there is only one block and paragraph then the slide might only contain the title. If this situation occurs, the stroke width and height checks are disabled because the averages of all content will only account for the title if these checks were left enabled.
