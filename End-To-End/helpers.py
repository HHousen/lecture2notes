import os
import hashlib

def make_dir_if_not_exist(path):
    """Makes directory `path` if it does not exist"""
    if not os.path.exists(path):
        print("> Helper Functions: Creating directory " + str(path))
        os.makedirs(path)

def gen_unique_id(input_data, k):
    """Returns the first k characters of the sha1 of input_data"""
    return hashlib.sha1(input_data.encode("UTF-8")).hexdigest()[:k]