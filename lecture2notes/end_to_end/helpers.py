import hashlib
import logging
import os
import re
import shutil
from distutils.dir_util import copy_tree, remove_tree
from pathlib import Path

logger = logging.getLogger(__name__)


def make_dir_if_not_exist(path):
    """Makes directory `path` if it does not exist"""
    if not os.path.exists(path):
        logger.info("Creating directory " + str(path))
        os.makedirs(path)


def gen_unique_id(input_data, k):
    """Returns the first k characters of the sha1 of input_data"""
    return hashlib.sha1(input_data.encode("UTF-8")).hexdigest()[:k]


def copy_all(list_path_files, output_dir, move=False):
    """Copy (or move) every path in `list_path_files` if list or all files in a path if path to output_dir"""
    if type(list_path_files) is list:
        make_dir_if_not_exist(output_dir)
        for file_path in list_path_files:
            output_file = output_dir / Path(file_path).name
            if move:
                shutil.move(file_path, output_file)
            else:
                shutil.copy(file_path, output_file)
    else:  # not a list
        copy_tree(str(list_path_files), str(output_dir))
        if move:
            remove_tree(str(list_path_files))

    return output_dir


def frame_number_from_filename(filename):
    try:
        return int(re.search(r"(?<=\_)[0-9]+(?=\_|.)", filename).group(0))
    except AttributeError:
        return None


def frame_number_filename_mapping(path, filenames_only=True):
    figures = os.listdir(path)
    figure_mapping = {}

    for figure_filename in figures:
        if filenames_only:
            figure_path = figure_filename
        else:
            figure_path = os.path.join(path, figure_path)

        frame_number = frame_number_from_filename(figure_filename)
        try:
            figure_mapping[frame_number].append(figure_path)
        except KeyError:
            figure_mapping[frame_number] = [figure_path]

    return figure_mapping


# frame_number_filename_mapping("process/frames_sorted/slide_clusters/best_samples_figures")
