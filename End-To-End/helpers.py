import os

def make_dir_if_not_exist(path):
    """Makes directory `path` if it does not exist"""
    if not os.path.exists(path):
        print("> Helper Functions: Creating directory " + str(path))
        os.makedirs(path)