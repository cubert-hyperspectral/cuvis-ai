import os
from contextlib import contextmanager


@contextmanager
def change_working_dir(dir_path):
    old_working_dir = os.getcwd()

    if dir_path is not None and dir_path not in ["", "."]:
        os.chdir(dir_path)
    try:
        yield
    finally:
        if dir_path is not None and dir_path not in ["", "."]:
            os.chdir(old_working_dir)
