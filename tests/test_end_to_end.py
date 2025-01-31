# The tests in this file use data generated by the create_fake_data.py notebook. If the tests
# fail because of a format change, you may need to rerun that notebook to regenerate
# the data/demo_data.fits file.

import os
import tempfile
import unittest

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.demo_helper import make_demo_data
from kbmod.run_search import SearchRunner
from kbmod.search import HAS_GPU
from kbmod.work_unit import WorkUnit


# this is the first test to actually test things like get_all_stamps from
# analysis utils. For now stamps have to be RawImages (because methods like
# convolve are defined to work on RawImage and not as funciton)
# so it makes sense to duplicate all this functionality to return np arrays
# (instead of RawImages), but hopefully we can deduplicate all this by making
# these operations into functions and calling on the .image attribute
# apply_stamp_filter for example is literal copy of the C++ code in RawImage?
class test_end_to_end(unittest.TestCase):
    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_demo_defaults(self):
        with tempfile.TemporaryDirectory() as dir_name:
            # Create a fake data file.
            filename = os.path.join(dir_name, "test_workunit1.fits")
            make_demo_data(filename)

            # Load the WorkUnit.
            input_data = WorkUnit.from_fits(filename)
            input_data.config.set("coadds", ["mean"])

            rs = SearchRunner()
            keep = rs.run_search_from_work_unit(input_data)
            self.assertGreaterEqual(len(keep), 1)
            self.assertEqual(keep["coadd_mean"][0].shape, (21, 21))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_demo_stamp_size(self):
        with tempfile.TemporaryDirectory() as dir_name:
            # Create a fake data file.
            filename = os.path.join(dir_name, "test_workunit2.fits")
            make_demo_data(filename)

            # Load the WorkUnit.
            input_data = WorkUnit.from_fits(filename)

            # Override the stamp settings of the configuration
            input_data.config.set("stamp_radius", 15)
            input_data.config.set("save_all_stamps", True)
            input_data.config.set("coadds", ["mean"])

            rs = SearchRunner()
            keep = rs.run_search_from_work_unit(input_data)
            self.assertGreaterEqual(len(keep), 1)

            self.assertIsNotNone(keep["coadd_mean"][0])
            self.assertEqual(keep["coadd_mean"][0].shape, (31, 31))

            self.assertIsNotNone(keep["all_stamps"][0])
            for s in keep["all_stamps"][0]:
                self.assertEqual(s.shape, (31, 31))


if __name__ == "__main__":
    unittest.main()
