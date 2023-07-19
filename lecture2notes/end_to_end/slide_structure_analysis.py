import json
import logging
import os
from functools import partial

import cv2
import numpy as np
import pandas as pd
import pytesseract
from skimage import img_as_float

# https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_peak_local_max.html
from skimage.feature import peak_local_max
from tqdm import tqdm

from .helpers import frame_number_from_filename

prev_line_num = 0
logger = logging.getLogger(__name__)


def stroke_width(image):
    """
    Determine the average stroke length in an image.
    Inspired by: https://stackoverflow.com/a/61914060.

    Other Links:

        * `cv2.distanceTransform Documentation <https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga25c259e7e2fa2ac70de4606ea800f12f>`_
        * `OpenCV Distance Transform Tutorial <https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html>`_
        * `Sckit-Image "Finding local maxima" <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_peak_local_max.html>`_
        * `skimage.feature.peak_local_max <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max>`_
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:  # image not found
        return 0
    gray_threshold = cv2.threshold(
        gray, 40, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    # plt.imshow(gray_threshold, "gray"), plt.show()
    dist = cv2.distanceTransform(gray_threshold, cv2.DIST_L2, 5)
    im = img_as_float(dist)
    coordinates = peak_local_max(im, min_distance=2)
    pixel_strength = []
    for element in coordinates:
        y = element[0]
        x = element[1]
        pixel_strength.append(dist[y, x])
    mean_pixel_strength = np.array(pixel_strength).mean()
    return mean_pixel_strength


def identify_title(
    tesseract_df,
    image,
    left_start_maximum=0.77,
    character_limit=3,
    enabled_checks=None,
):
    if enabled_checks is None:
        enabled_checks = [
            "in_upper_third",
            "in_top_left",
            "large_stroke_width",
            "large_height",
            "meets_character_limit",
        ]
    image_height, image_width = image.shape[:2]
    # Critera to be classified as a title:
    # it is in the upper third of the image,
    # it is the first block and first paragraph as recognized by tesseract
    # its stroke width is larger than one standard deviation from the average stroke width
    # its average line height is greater than the average of all line heights
    # the x position of its top-left corner is lower than a ``image_width * left_start_maximum``,
    # the text line has more than three characters.

    all_checks_passed = True
    # If there is only one block and only one paragraph then the slide
    # might only contain the title. Disable the "large_stroke_width" and
    # "large_height" checks.
    if (
        tesseract_df["block_num"].max() == tesseract_df["block_num"].min()
        and tesseract_df["par_num"].max() == tesseract_df["par_num"].min()
    ):
        enabled_checks = [
            x
            for x in enabled_checks
            if x != "large_stroke_width" and x != "large_height"
        ]

    # Start by selecting the first block and first paragraph in that block
    first_block = tesseract_df[
        (tesseract_df["block_num"] == 1) & (tesseract_df["par_num"] == 1)
    ]

    if all_checks_passed and "in_upper_third" in enabled_checks:
        in_upper_third = first_block["top"].mean() < image_height / 3
        all_checks_passed = in_upper_third

    if all_checks_passed and "in_top_left" in enabled_checks:
        in_top_left = first_block["left"].mean() < image_width * left_start_maximum
        all_checks_passed = in_top_left

    if all_checks_passed and "meets_character_limit" in enabled_checks:
        meets_character_limit = len("".join(first_block["text"])) > character_limit
        all_checks_passed = meets_character_limit

    if all_checks_passed and "large_stroke_width" in enabled_checks:
        avg_stroke_width = tesseract_df["stroke_width"].mean()
        std_stroke_width = tesseract_df["stroke_width"].std()
        large_stroke_width = (
            first_block["stroke_width"].mean() > avg_stroke_width + std_stroke_width
        )
        all_checks_passed = large_stroke_width

    if all_checks_passed and "large_height" in enabled_checks:
        avg_height = tesseract_df["height"].mean()
        large_height = first_block["height"].mean() > avg_height
        all_checks_passed = large_height

    if all_checks_passed:
        raw_text = " ".join(first_block["text"]).strip()
        global_line_nums = list(
            set(first_block["global_line_num"])
        )  # use `set` to remove duplicates
        return raw_text, global_line_nums

    return None


def analyze_structure(
    image,
    to_json=None,
    return_unstructured_text=True,
    gamma=0.1,
    beta=0.2,
    orient="index",
    extra_json=None,
):
    """Perform slide structure analysis.

    Args:
        image (np.array): Image to be processed as loaded with ``cv2.imread()``.
        to_json (str or bool, optional): Path to write json output or a boolean to return
            json data as a string. The default return value is a pd.DataFrame. Defaults to
            None.
        return_unstructured_text (bool, optiona): If the raw recognized text should be
            returned in addition to the other return values.
        gamma (float, optional): The percentage greater than or less than the average
            **stroke width** that a text line must meet to be classified as bold/subtitle or
            small text repsectively. Defaults to 0.1.
        beta (float, optional): The percentage greater than or less than the average
            **height** that a text line must meet to be classified as bold/subtitle or
            small text repsectively. This is greater than ``gamma`` because height is on a
            larger scale than gamma. Defaults to 0.2.
        orient (str, optional): The format of the output json data if ``to_json`` is set. The
            acceptable values can be found on the
            `pandas.DataFrame.to_json documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html>`_.
            Defaults to "index".
        extra_json (dict, optional): Additional keys and values to add to the json output if
            ``to_json`` is enabled. Defaults to None.

    Returns:
        pd.DataFrame or str or tuple or ``None``: The default is to return a pd.DataFrame. However,
        setting ``to_json`` to a string will instead write json data to ``to_json`` and return
        the path to the data. Setting ``to_json`` to ``True`` will return the json data
        as a string. Setting ``return_unstructured_text`` returns the previously described data
        and the raw recognized text as a tuple. Will return ``None`` is no text is detected.
    """

    def add_stroke_width_info(row):
        x = row["left"]
        y = row["top"]
        w = row["width"]
        h = row["height"]
        word = image[y : y + h, x : x + w]
        return stroke_width(word)

    prev_line_num = 0

    def add_line_info(row):
        """
        Adds the global line number (``line_num``) independent of page/block/paragraph
        to each row.
        Inspired by https://stackoverflow.com/a/53118102.
        """
        global prev_line_num
        if row["word_num"] == 1:
            current_line_num = prev_line_num  # noqa: F841
            prev_line_num += 1
        return prev_line_num

    def categorize_text(row, avg_height, avg_stroke_width, gamma, beta):
        # Categories:
        # -1: small text
        #  0: normal text
        #  1: subtitle/bold
        #  2: title text
        stroke_width = row["stroke_width"]
        height = row["height"]
        if stroke_width > avg_stroke_width * (1 + gamma) or height > avg_height * (
            1 + beta
        ):
            return 1  # subtitle/bold
        if stroke_width < avg_stroke_width * (1 - gamma) and height < avg_height * (
            1 - beta
        ):
            return -1  # small text

        return 0  # normal text

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pd.set_option("display.max_rows", 300)
    data_df = pytesseract.image_to_data(
        image_rgb, output_type=pytesseract.Output.DATAFRAME
    )
    data_df = data_df.dropna()
    # Remove 0 width or 0 height matches, which are mistakes
    data_df = data_df[(data_df["width"] != 0) & (data_df["height"] != 0)]
    # Remove empty text lines
    data_df = data_df[(data_df["text"] != "") & (data_df["text"] != " ")]

    if data_df.empty:
        logger.warn("No text detected in this slide")
        return None

    prev_line_num = 0
    data_df["global_line_num"] = data_df.apply(add_line_info, axis=1)
    data_df["stroke_width"] = data_df.apply(add_stroke_width_info, axis=1)

    title_output = identify_title(data_df, image)
    if title_output is not None:
        title_text, title_global_line_nums = title_output

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
    grouped = data_df.groupby("global_line_num")
    grouped_text = grouped["text"].apply(
        lambda o: " ".join(o).strip()
    )  # https://stackoverflow.com/a/27298308
    grouped_others = grouped.mean()
    lines = pd.concat((grouped_others, grouped_text), axis=1)

    avg_height = data_df["height"].mean()
    avg_stroke_width = data_df["stroke_width"].mean()
    _categorize_text = partial(
        categorize_text,
        avg_height=avg_height,
        avg_stroke_width=avg_stroke_width,
        gamma=gamma,
        beta=beta,
    )
    lines["category"] = lines.apply(_categorize_text, axis=1)

    # Set the title lines to category 2.
    if title_output is not None:
        lines.loc[title_global_line_nums, "category"] = 2

    lines.drop(
        [
            "level",
            "page_num",
            "word_num",
            "height",
            "left",
            "top",
            "width",
            "conf",
            "stroke_width",
        ],
        axis="columns",
        inplace=True,
    )
    columns_to_int = ["block_num", "par_num", "line_num"]
    lines[columns_to_int] = lines[columns_to_int].astype(int)
    # input(lines)

    non_small_text_series = lines.loc[lines["category"] != -1]["text"]
    raw_text = " ".join(non_small_text_series).strip()

    to_return = []
    if to_json:
        if type(to_json) is bool:
            if extra_json:
                lines_dict = lines.to_dict()
                lines_dict.update(extra_json)
                json_data = json.dumps(lines_dict)
            else:
                json_data = lines.to_json()
            to_return.append(json_data)
        else:
            if extra_json:
                json_data = json.loads(lines.to_json(to_json, orient=orient))
                json_data.update(extra_json)
                with open(to_json, "w+") as json_file:
                    json.dump(json_data, json_file)
            else:
                json_data = lines.to_json(to_json, orient=orient)
            to_return.append(to_json)
    else:
        to_return.append(lines)

    if return_unstructured_text:
        to_return.append(raw_text)

    if len(to_return) == 1:
        return to_return[0]

    return to_return


def all_in_folder(path, do_rename=True, **kwargs):
    """Perform structure analysis and OCR on every file in folder using
    :meth:`~lecture2notes.end_to_end.slide_structure_analysis.analyze_structure`.

    Args:
        path (str): Directory containing images to process.
        do_rename (str, optional): Rename files to just their frame number. Defaults
            to True.
        ``**kwargs`` (dict, optional) is passed to
            :meth:`lecture2notes.end_to_end.slide_structure_analysis.analyze_structure`.

    Returns:
        tuple: (raw_texts, json_texts) A list of the raw text for each slide and a
        list of the json structure analysis data for each slide.
    """
    raw_texts = []
    json_texts = []
    images = os.listdir(path)
    images.sort()
    for item in tqdm(
        images, total=len(images), desc="> Slide Structure Analysis: Progress"
    ):
        # logger.info("> OCR: Processing file " + item)
        current_path = os.path.join(path, item)
        if os.path.isfile(current_path):
            image = cv2.imread(current_path)
            frame_number = frame_number_from_filename(current_path)
            if not frame_number:
                continue

            if do_rename:
                item_directory = os.path.dirname(item)
                file_extension = os.path.splitext(item)[1]
                new_path = os.path.join(
                    path, item_directory, str(frame_number) + file_extension
                )
                os.rename(current_path, new_path)

            frame_number = int(frame_number)
            analyze_structure_outputs = analyze_structure(
                image, to_json=True, extra_json={"frame_number": frame_number}, **kwargs
            )
            if analyze_structure_outputs is not None:
                json_text, raw_text = analyze_structure_outputs
                raw_texts.append(raw_text)
                json_texts.append(json_text)

    return raw_texts, json_texts


def write_to_file(raw_texts, json_texts, raw_save_file, json_save_file):
    """Write the raw text in ``raw_texts`` to ``raw_save_file`` and the json data
    in ``json_texts`` to ``json_save_file``. Used to write results from
    :meth:`~lecture2notes.end_to_end.slide_structure_analysis.all_in_folder` to disk.

    Args:
        raw_texts (list): List of raw text outputs from :meth:`~lecture2notes.end_to_end.slide_structure_analysis.analyze_structure`.
        json_texts (list): List of json ssa outputs from :meth:`~lecture2notes.end_to_end.slide_structure_analysis.analyze_structure`.
        raw_save_file (str): The path to save the raw text. A ".txt" file.
        json_save_file (str): The path to save the json output. A ".json" file.
    """
    logger.info("Writing raw text to file " + str(raw_save_file))
    with open(raw_save_file, "w+") as file_results:
        for item in raw_texts:
            file_results.write(item + "\r\n")
    logger.debug("Raw text written to " + str(raw_save_file))

    logger.info("Writing JSON text to file " + str(json_save_file))
    with open(json_save_file, "w+") as file_results:
        file_results.write("[")
        for idx, item in enumerate(json_texts):
            # If this is the last json string, don't write the comma
            if len(json_texts) == idx + 1:
                file_results.write(item)
            else:
                file_results.write(item + ", ")
        file_results.write("]")
    logger.debug("JSON text written to " + str(json_save_file))


# analyze_structure(cv2.imread("test_data/MIT2_627F13_lec04-11.png"), "test.json")

# outputs = all_in_folder("test_data")
# write_to_file(outputs[0], outputs[1], "remove1.txt", "remove2.json")
