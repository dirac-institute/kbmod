from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import os
from pathlib import Path
import tempfile
import unittest
import warnings

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data_creator import make_fake_wcs_info
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
        header_dict = make_fake_wcs_info(200.6145, -7.7888, 2000, 4000)
        self.wcs = WCS(header_dict)
        self.per_image_wcs = per_image_wcs = [
            (self.wcs if i % 2 == 0 else None) for i in range(self.num_images)
        ]

    def test_create(self):
        work = WorkUnit(self.im_stack, self.config)
        self.assertEqual(work.im_stack.img_count(), 5)
        self.assertEqual(work.config["im_filepath"], "Here")
        self.assertEqual(work.config["num_obs"], 5)
        self.assertIsNone(work.wcs)
        self.assertEqual(len(work), self.num_images)
        for i in range(self.num_images):
            self.assertIsNone(work.get_wcs(i))

        # Create with a global WCS
        work2 = WorkUnit(self.im_stack, self.config, self.wcs)
        self.assertEqual(work2.im_stack.img_count(), 5)
        self.assertIsNotNone(work2.wcs)
        for i in range(self.num_images):
            self.assertIsNotNone(work2.get_wcs(i))

        # Mismatch with the number of WCS.
        self.assertRaises(
            ValueError,
            WorkUnit,
            self.im_stack,
            self.config,
            self.wcs,
            [self.wcs, self.wcs, self.wcs],
        )

        # Create with per-image WCS
        per_image_wcs = [self.wcs] * self.num_images
        work3 = WorkUnit(self.im_stack, self.config, per_image_wcs=per_image_wcs)
        self.assertIsNone(work3.wcs)
        for i in range(self.num_images):
            self.assertIsNotNone(work3.get_wcs(i))

        # Create with both global and per-image WCS. Check that a get triggers a warning.
        work4 = WorkUnit(self.im_stack, self.config, self.wcs, per_image_wcs)
        self.assertIsNotNone(work4.wcs)
        with warnings.catch_warnings(record=True) as wrn:
            warnings.simplefilter("always")
            current = work4.get_wcs(0)
            self.assertTrue("Both a global and per-image WCS given." in str(wrn[-1].message))
            self.assertIsNotNone(current)

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
                    "per_image_wcs": self.per_image_wcs,
                    "wcs": self.wcs,
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
                    "per_image_wcs": self.per_image_wcs,
                    "wcs": self.wcs,
                }

            with self.subTest(i=use_python_types):
                work = WorkUnit.from_dict(work_unit_dict)
                self.assertEqual(work.im_stack.img_count(), self.num_images)
                self.assertEqual(work.im_stack.get_width(), self.width)
                self.assertEqual(work.im_stack.get_height(), self.height)
                self.assertIsNotNone(work.wcs)
                for i in range(self.num_images):
                    layered1 = work.im_stack.get_single_image(i)
                    layered2 = self.im_stack.get_single_image(i)

                    self.assertTrue(layered1.get_science().l2_allclose(layered2.get_science(), 0.01))
                    self.assertTrue(layered1.get_variance().l2_allclose(layered2.get_variance(), 0.01))
                    self.assertTrue(layered1.get_mask().l2_allclose(layered2.get_mask(), 0.01))
                    self.assertEqual(layered1.get_obstime(), layered2.get_obstime())
                    self.assertEqual(work.per_image_wcs[i] is None, i % 2 == 1)

                self.assertTrue(type(work.config) is SearchConfiguration)
                self.assertEqual(work.config["im_filepath"], "Here")
                self.assertEqual(work.config["num_obs"], 5)

    def test_save_and_load_fits(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_fits, file_path)

            # Write out the existing WorkUnit with a per image wcs for the even entries.
            work = WorkUnit(self.im_stack, self.config, self.wcs, self.per_image_wcs)
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

                # No per-image WCS on the odd entries
                self.assertEqual(work2.per_image_wcs[i] is None, i % 2 == 1)

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["im_filepath"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)
            self.assertDictEqual(work2.config["mask_bits_dict"], {"A": 1, "B": 2})
            self.assertIsNone(work2.config["repeated_flag_keys"])

    def test_to_from_yaml(self):
        work = WorkUnit(self.im_stack, self.config, self.wcs, self.per_image_wcs)
        yaml_str = work.to_yaml()

        work2 = WorkUnit.from_yaml(yaml_str)
        self.assertEqual(work2.im_stack.img_count(), self.num_images)
        self.assertEqual(work2.im_stack.get_width(), self.width)
        self.assertEqual(work2.im_stack.get_height(), self.height)
        self.assertIsNotNone(work2.wcs)
        for i in range(self.num_images):
            layered1 = work2.im_stack.get_single_image(i)
            layered2 = self.im_stack.get_single_image(i)

            self.assertTrue(layered1.get_science().l2_allclose(layered2.get_science(), 0.01))
            self.assertTrue(layered1.get_variance().l2_allclose(layered2.get_variance(), 0.01))
            self.assertTrue(layered1.get_mask().l2_allclose(layered2.get_mask(), 0.01))
            self.assertAlmostEqual(layered1.get_obstime(), layered2.get_obstime())
            self.assertEqual(work2.per_image_wcs[i] is None, i % 2 == 1)

        # Check that we read in the configuration values correctly.
        self.assertEqual(work2.config["im_filepath"], "Here")
        self.assertEqual(work2.config["num_obs"], self.num_images)
        self.assertDictEqual(work2.config["mask_bits_dict"], {"A": 1, "B": 2})
        self.assertIsNone(work2.config["repeated_flag_keys"])


if __name__ == "__main__":
    unittest.main()
