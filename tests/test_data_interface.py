from astropy.wcs import WCS
import os
import tempfile
import unittest
from yaml import dump

from kbmod.configuration import SearchConfiguration
from kbmod.data_interface import (
    load_input_from_config,
    load_input_from_file,
    load_input_from_individual_files,
)
from kbmod.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.search import *
from kbmod.work_unit import WorkUnit
from utils.utils_for_tests import get_absolute_data_path


class test_data_interface(unittest.TestCase):
    def test_file_load_basic(self):
        stack, wcs_list, mjds = load_input_from_individual_files(
            get_absolute_data_path("fake_images"),
            None,
            None,
            [0, 157130.2],
            PSF(1.0),
            verbose=False,
        )
        self.assertEqual(stack.img_count(), 4)

        # Check that each image loaded corrected.
        true_times = [57130.2, 57130.21, 57130.22, 57131.2]
        for i in range(stack.img_count()):
            img = stack.get_single_image(i)
            self.assertEqual(img.get_width(), 64)
            self.assertEqual(img.get_height(), 64)
            self.assertAlmostEqual(img.get_obstime(), true_times[i], delta=0.005)
            self.assertAlmostEqual(1.0, img.get_psf().get_std())

    def test_file_load_extra(self):
        p = PSF(1.0)

        stack, wcs_list, mjds = load_input_from_individual_files(
            get_absolute_data_path("fake_images"),
            get_absolute_data_path("fake_times.dat"),
            get_absolute_data_path("fake_psfs.dat"),
            [0, 157130.2],
            p,
            verbose=False,
        )
        self.assertEqual(stack.img_count(), 4)

        # Check that each image loaded corrected.
        true_times = [57130.2, 57130.21, 57130.22, 57162.0]
        psfs_std = [1.0, 1.0, 1.3, 1.0]
        for i in range(stack.img_count()):
            img = stack.get_single_image(i)
            self.assertEqual(img.get_width(), 64)
            self.assertEqual(img.get_height(), 64)
            self.assertAlmostEqual(img.get_obstime(), true_times[i], delta=0.005)
            self.assertAlmostEqual(psfs_std[i], img.get_psf().get_std())

    def test_file_load_config(self):
        config = SearchConfiguration()
        config.set("im_filepath", get_absolute_data_path("fake_images")),
        config.set("time_file", get_absolute_data_path("fake_times.dat")),
        config.set("psf_file", get_absolute_data_path("fake_psfs.dat")),
        config.set("psf_val", 1.0)

        worku = load_input_from_config(config, verbose=False)

        # Check that each image loaded corrected.
        true_times = [57130.2, 57130.21, 57130.22, 57162.0]
        psfs_std = [1.0, 1.0, 1.3, 1.0]
        for i in range(worku.im_stack.img_count()):
            img = worku.im_stack.get_single_image(i)
            self.assertEqual(img.get_width(), 64)
            self.assertEqual(img.get_height(), 64)
            self.assertAlmostEqual(img.get_obstime(), true_times[i], delta=0.005)
            self.assertAlmostEqual(psfs_std[i], img.get_psf().get_std())

        # Try writing the configuration to a YAML file and loading.
        with tempfile.TemporaryDirectory() as dir_name:
            yaml_file_path = os.path.join(dir_name, "test_config.yml")

            with self.assertRaises(ValueError):
                work_fits = load_input_from_file(yaml_file_path)

            config.to_file(yaml_file_path)

            work_yml = load_input_from_file(yaml_file_path)
            self.assertIsNotNone(work_yml)
            self.assertEqual(work_yml.im_stack.img_count(), 4)

    def test_file_load_workunit(self):
        # Create a fake WCS
        fake_wcs = WCS(
            {
                "WCSAXES": 2,
                "CTYPE1": "RA---TAN-SIP",
                "CTYPE2": "DEC--TAN-SIP",
                "CRVAL1": 200.614997245422,
                "CRVAL2": -7.78878863332778,
                "CRPIX1": 1033.934327,
                "CRPIX2": 2043.548284,
                "CTYPE1A": "LINEAR  ",
                "CTYPE2A": "LINEAR  ",
                "CUNIT1A": "PIXEL   ",
                "CUNIT2A": "PIXEL   ",
            }
        )
        fake_config = SearchConfiguration()
        fake_times = create_fake_times(11, 57130.2, 10, 0.01, 1)
        fake_data = FakeDataSet(64, 64, fake_times, use_seed=True)
        work = WorkUnit(fake_data.stack, fake_config, fake_wcs, None)

        with tempfile.TemporaryDirectory() as dir_name:
            # Save and load as FITS
            fits_file_path = os.path.join(dir_name, "test_workunit.fits")

            with self.assertRaises(ValueError):
                work_fits = load_input_from_file(fits_file_path)

            work.to_fits(fits_file_path)

            work_fits = load_input_from_file(fits_file_path)
            self.assertIsNotNone(work_fits)
            self.assertEqual(work_fits.im_stack.img_count(), 11)

            # Save and load as YAML
            yaml_file_path = os.path.join(dir_name, "test_workunit.yml")
            with open(yaml_file_path, "w") as file:
                file.write(work.to_yaml())

            work_yml = load_input_from_file(yaml_file_path)
            self.assertIsNotNone(work_yml)
            self.assertEqual(work_yml.im_stack.img_count(), 11)

    def test_file_load_invalid(self):
        # Create a YAML file that is neither a configuration nor a WorkUnit.
        yaml_str = dump({"Field1": 1, "Field2": False})

        with tempfile.TemporaryDirectory() as dir_name:
            yaml_file_path = os.path.join(dir_name, "test_invalid.yml")
            with open(yaml_file_path, "w") as file:
                file.write(yaml_str)

            with self.assertRaises(ValueError):
                work = load_input_from_file(yaml_file_path)


if __name__ == "__main__":
    unittest.main()
