from os import path


def get_absolute_data_path(file_or_directory):
    test_dir = path.abspath(path.dirname(path.dirname(__file__)))
    data_dir = path.join(test_dir, "data")
    return path.join(data_dir, file_or_directory)
