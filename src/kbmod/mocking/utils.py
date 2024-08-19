import tarfile
from os import path

from astropy.table import Table, MaskedColumn


# def get_absolute_data_path(file_or_directory):
#    test_dir = path.abspath(path.dirname(__file__))
#    data_dir = path.join(test_dir, "archived_data")
#    return path.join(data_dir, file_or_directory)
#
#
# def get_absolute_demo_data_path(file_or_directory):
#    project_root_dir = path.abspath(path.dirname(path.dirname(__file__)))
#    data_dir = path.join(project_root_dir, "archived_data")
#    return path.join(data_dir, file_or_directory)


def header_archive_to_table(archive_path, fname, compression, format, external=True):
    if not external:
        mocking_dir = path.abspath(path.dirname(__file__))
        archive_path = path.join(mocking_dir, archive_path)
    with tarfile.open(archive_path, f"r:{compression}") as archive:
        tblfile = archive.extractfile(fname)
        table = Table.read(tblfile.read().decode(), format=format)
        # sometimes empty strings get serialized as masked, to cover that
        # eventuality we'll just substitute an empty string
        if isinstance(table["value"], MaskedColumn):
            table["value"].fill_value = ""
            table["value"] = table["value"].filled()

    return table
