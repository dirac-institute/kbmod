from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import tempfile
import unittest
from pathlib import Path

from kbmod.configuration import SearchConfiguration
import kbmod.search as kb
from kbmod.work_unit import hdu_to_raw_image, raw_image_to_hdu, WorkUnit


class test_work_unit(unittest.TestCase):
    def setUp(self):
        self.num_images = 5
        self.width = 50
        self.height = 70
        self.images = [None] * self.num_images
        self.p = [None] * self.num_images
        for i in range(self.num_images):
            self.p[i] = kb.PSF(5.0 / float(2 * i + 1))
            self.images[i] = kb.LayeredImage(
                ("layered_test_%i" % i),
                self.width,
                self.height,
                2.0,  # noise_level
                4.0,  # variance
                2.0 * i + 1.0,  # time
                self.p[i],
            )

            # Include one masked pixel per time step at (10, 10 + i).
            mask = self.images[i].get_mask()
            mask.set_pixel(10, 10 + i, 1)

        self.im_stack = kb.ImageStack(self.images)

        self.config = SearchConfiguration()
        self.config.set("im_filepath", "Here")
        self.config.set("num_obs", self.num_images)
        self.config.set("mask_bits_dict", {"A": 1, "B": 2})
        self.config.set("repeated_flag_keys", None)

        # Create a fake WCS
        header_dict = {
            "WCSAXES": 2,
            "CTYPE1": "RA---TAN-SIP",
            "CTYPE2": "DEC--TAN-SIP",
            "CRVAL1": 200.614997245422,
            "CRVAL2": -7.78878863332778,
            "CRPIX1": 1033.934327,
            "CRPIX2": 2043.548284,
            "CD1_1": -1.13926485986789e-07,
            "CD1_2": 7.31839748843125e-05,
            "CD2_1": -7.30064978350695e-05,
            "CD2_2": -1.27520156332774e-07,
            "CTYPE1A": "LINEAR  ",
            "CTYPE2A": "LINEAR  ",
            "CUNIT1A": "PIXEL   ",
            "CUNIT2A": "PIXEL   ",
        }
        self.wcs = WCS(header_dict)

    def test_create(self):
        work = WorkUnit(self.im_stack, self.config)
        self.assertEqual(work.im_stack.img_count(), 5)
        self.assertEqual(work.config["im_filepath"], "Here")
        self.assertEqual(work.config["num_obs"], 5)
        self.assertIsNone(work.wcs)

        # Create with a global WCS
        work2 = WorkUnit(self.im_stack, self.config, self.wcs)
        self.assertEqual(work2.im_stack.img_count(), 5)
        self.assertIsNotNone(work2.wcs)

    def test_save_and_load_fits(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = f"{dir_name}/test_workunit.fits"
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_fits, file_path)

            # Write out the existing WorkUnit
            work = WorkUnit(self.im_stack, self.config, self.wcs)
            work.to_fits(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_fits(file_path)
            self.assertEqual(work2.im_stack.img_count(), self.num_images)
            self.assertIsNotNone(work2.wcs)
            for i in range(self.num_images):
                li = work2.im_stack.get_single_image(i)
                self.assertEqual(li.get_width(), self.width)
                self.assertEqual(li.get_height(), self.height)
                self.assertEqual(li.get_obstime(), 2 * i + 1)

                # Check the three image layers match.
                sci1 = li.get_science()
                var1 = li.get_variance()
                msk1 = li.get_mask()

                li_org = self.im_stack.get_single_image(i)
                sci2 = li_org.get_science()
                var2 = li_org.get_variance()
                msk2 = li_org.get_mask()

                for y in range(self.height):
                    for x in range(self.width):
                        self.assertAlmostEqual(sci1.get_pixel(y, x), sci2.get_pixel(y, x))
                        self.assertAlmostEqual(var1.get_pixel(y, x), var2.get_pixel(y, x))
                        self.assertAlmostEqual(msk1.get_pixel(y, x), msk2.get_pixel(y, x))

                # Check the PSF layer matches.
                p1 = self.p[i]
                p2 = li.get_psf()
                self.assertEqual(p1.get_dim(), p2.get_dim())

                for y in range(p1.get_dim()):
                    for x in range(p1.get_dim()):
                        self.assertAlmostEqual(p1.get_value(y, x), p2.get_value(y, x))

                # No per-image WCS
                self.assertIsNone(work2.per_image_wcs[i])

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["im_filepath"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)
            self.assertDictEqual(work2.config["mask_bits_dict"], {"A": 1, "B": 2})
            self.assertIsNone(work2.config["repeated_flag_keys"])


if __name__ == "__main__":
    unittest.main()
