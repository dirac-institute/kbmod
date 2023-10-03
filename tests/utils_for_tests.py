from os import path

def get_absolute_data_path(file_or_directory):
    test_dir = path.abspath(path.dirname(__file__))
    data_dir = path.join(test_dir, "data")
    return path.join(data_dir, file_or_directory)

def get_absolute_demo_data_path(file_or_directory):
    project_root_dir = path.abspath(path.dir_name(path.dirname(__file__)))
    data_dir = path.join(project_root_dir, "data")
    return path.join(data_dir, file_or_directory)