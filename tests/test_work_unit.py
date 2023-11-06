from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path
import tempfile
import unittest

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

    def test_create(self):
        work = WorkUnit(self.im_stack, self.config)
        self.assertEqual(work.im_stack.img_count(), 5)
        self.assertEqual(work.config["im_filepath"], "Here")
        self.assertEqual(work.config["num_obs"], 5)

    def test_create_from_dict(self):
        for use_python_types in [True, False]:
            if use_python_types:
                work_unit_dict = {
                    "num_images": self.num_images,
                    "width": self.width,
                    "height": self.height,
                    "config": self.config._params,
                    "times": [self.images[i].get_obstime() for i in range(self.num_images)],
                    "sci_imgs": [self.images[i].get_science().image for i in range(self.num_images)],
                    "var_imgs": [self.images[i].get_variance().image for i in range(self.num_images)],
                    "msk_imgs": [self.images[i].get_mask().image for i in range(self.num_images)],
                    "psfs": [np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim())) for p in self.p],
                }
            else:
                work_unit_dict = {
                    "num_images": self.num_images,
                    "width": self.width,
                    "height": self.height,
                    "config": self.config,
                    "times": [self.images[i].get_obstime() for i in range(self.num_images)],
                    "sci_imgs": [self.images[i].get_science() for i in range(self.num_images)],
                    "var_imgs": [self.images[i].get_variance() for i in range(self.num_images)],
                    "msk_imgs": [self.images[i].get_mask() for i in range(self.num_images)],
                    "psfs": self.p,
                }

            with self.subTest(i=use_python_types):
                work = WorkUnit.from_dict(work_unit_dict)
                self.assertEqual(work.im_stack.img_count(), self.num_images)
                self.assertEqual(work.im_stack.get_width(), self.width)
                self.assertEqual(work.im_stack.get_height(), self.height)
                for i in range(self.num_images):
                    layered1 = work.im_stack.get_single_image(i)
                    layered2 = self.im_stack.get_single_image(i)

                    self.assertTrue(layered1.get_science().l2_allclose(layered2.get_science(), 0.01))
                    self.assertTrue(layered1.get_variance().l2_allclose(layered2.get_variance(), 0.01))
                    self.assertTrue(layered1.get_mask().l2_allclose(layered2.get_mask(), 0.01))
                    self.assertEqual(layered1.get_obstime(), layered2.get_obstime())

                self.assertTrue(type(work.config) is SearchConfiguration)
                self.assertEqual(work.config["im_filepath"], "Here")
                self.assertEqual(work.config["num_obs"], 5)

    def test_save_and_load_fits(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = f"{dir_name}/test_workunit.fits"
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_fits, file_path)

            # Write out the existing WorkUnit
            work = WorkUnit(self.im_stack, self.config)
            work.to_fits(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_fits(file_path)
            self.assertEqual(work2.im_stack.img_count(), self.num_images)
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

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["im_filepath"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)
            self.assertDictEqual(work2.config["mask_bits_dict"], {"A": 1, "B": 2})
            self.assertIsNone(work2.config["repeated_flag_keys"])

    def test_to_from_yaml(self):
        work = WorkUnit(self.im_stack, self.config)
        yaml_str = work.to_yaml()

        work2 = WorkUnit.from_yaml(yaml_str)
        self.assertEqual(work2.im_stack.img_count(), self.num_images)
        self.assertEqual(work2.im_stack.get_width(), self.width)
        self.assertEqual(work2.im_stack.get_height(), self.height)
        for i in range(self.num_images):
            layered1 = work2.im_stack.get_single_image(i)
            layered2 = self.im_stack.get_single_image(i)

            self.assertTrue(layered1.get_science().l2_allclose(layered2.get_science(), 0.01))
            self.assertTrue(layered1.get_variance().l2_allclose(layered2.get_variance(), 0.01))
            self.assertTrue(layered1.get_mask().l2_allclose(layered2.get_mask(), 0.01))
            self.assertAlmostEqual(layered1.get_obstime(), layered2.get_obstime())

        # Check that we read in the configuration values correctly.
        self.assertEqual(work2.config["im_filepath"], "Here")
        self.assertEqual(work2.config["num_obs"], self.num_images)
        self.assertDictEqual(work2.config["mask_bits_dict"], {"A": 1, "B": 2})
        self.assertIsNone(work2.config["repeated_flag_keys"])


if __name__ == "__main__":
    unittest.main()
