import os
import hashlib
import shutil
from pathlib import Path
from distutils.dir_util import copy_tree, remove_tree

def make_dir_if_not_exist(path):
    """Makes directory `path` if it does not exist"""
    if not os.path.exists(path):
        print("> Helper Functions: Creating directory " + str(path))
        os.makedirs(path)

def gen_unique_id(input_data, k):
    """Returns the first k characters of the sha1 of input_data"""
    return hashlib.sha1(input_data.encode("UTF-8")).hexdigest()[:k]

def copy_all(list_path_files, output_dir, move=False):
    """ Copy (or move) every path in `list_path_files` if list or all files in a path if path to output_dir """
    if type(list_path_files) is list:
        make_dir_if_not_exist(output_dir)
        for file_path in list_path_files:
            output_file = output_dir / Path(file_path).stem
            if move:
                shutil.move(file_path, output_file)
            else:
                shutil.copy(file_path, output_file)
    else: # not a list
        copy_tree(str(list_path_files), str(output_dir))
        if move:
            remove_tree(str(list_path_files))

    return output_dir
