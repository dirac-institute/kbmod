from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import numpy as np
import os
from pathlib import Path
import tempfile
import unittest
import warnings

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import make_fake_layered_image
import kbmod.search as kb
from kbmod.reprojection_utils import fit_barycentric_wcs
from kbmod.wcs_utils import make_fake_wcs, wcs_fits_equal
from kbmod.work_unit import hdu_to_raw_image, raw_image_to_hdu, WorkUnit

import numpy.testing as npt


class test_work_unit(unittest.TestCase):
    def setUp(self):
        self.num_images = 5
        self.width = 50
        self.height = 70
        self.images = [None] * self.num_images
        self.p = [None] * self.num_images
        for i in range(self.num_images):
            self.p[i] = kb.PSF(5.0 / float(2 * i + 1))
            self.images[i] = make_fake_layered_image(
                self.width,
                self.height,
                2.0,  # noise_level
                4.0,  # variance
                59000.0 + (2.0 * i + 1.0),  # time
                self.p[i],
            )

            # Include one masked pixel per time step at (10, 10 + i).
            mask = self.images[i].get_mask()
            mask.set_pixel(10, 10 + i, 1)

        self.im_stack = kb.ImageStack(self.images)

        self.config = SearchConfiguration()
        self.config.set("im_filepath", "Here")
        self.config.set("num_obs", self.num_images)

        # Create a fake WCS
        self.wcs = make_fake_wcs(200.6145, -7.7888, 500, 700, 0.00027)
        self.per_image_wcs = per_image_wcs = [self.wcs for i in range(self.num_images)]

        self.diff_wcs = []
        for i in range(self.num_images):
            self.diff_wcs.append(make_fake_wcs(200.0 + i, -7.7888, 500, 700))

        self.indices = [0, 1, 2, 3]
        self.pixel_positions = [(350 + i, 300 + i) for i in self.indices]

        self.input_radec_positions = [
            SkyCoord(201.60941351, -8.19964405, unit="deg"),
            SkyCoord(201.60968108, -8.19937797, unit="deg"),
            SkyCoord(201.60994864, -8.19911188, unit="deg"),
            SkyCoord(201.61021621, -8.19884579, unit="deg"),
        ]
        self.expected_radec_positions = [
            SkyCoord(200.62673991, -7.79623142, unit="deg"),
            SkyCoord(200.59733711, -7.78473232, unit="deg"),
            SkyCoord(200.56914856, -7.77372976, unit="deg"),
            SkyCoord(200.54220037, -7.76323338, unit="deg"),
        ]

        self.expected_pixel_positions = [
            (293.91409096900713, 321.4755237663834),
            (186.0196821526124, 364.0641470322672),
            (82.57542144600637, 404.8067348560266),
            (-16.322177615492762, 443.6685337511032),
        ]

        self.per_image_ebd_wcs, self.geo_dist = fit_barycentric_wcs(
            self.wcs,
            self.width,
            self.height,
            41.0,
            Time(59000, format="mjd"),
            EarthLocation.of_site("ctio"),
        )

        self.constituent_images = [
            "one.fits",
            "two.fits",
            "three.fits",
            "four.fits",
            "five.fits",
        ]

    def test_create(self):
        # Test the creation of a WorkUnit with no WCS. Should throw a warning.
        with warnings.catch_warnings(record=True) as wrn:
            warnings.simplefilter("always")
            work = WorkUnit(self.im_stack, self.config)
            self.assertTrue("No WCS provided." in str(wrn[-1].message))

            self.assertIsNotNone(work)
            self.assertEqual(work.im_stack.img_count(), 5)
            self.assertEqual(work.config["im_filepath"], "Here")
            self.assertEqual(work.config["num_obs"], 5)
            self.assertFalse(work.has_common_wcs())
            self.assertIsNone(work.wcs)
            self.assertEqual(len(work), self.num_images)
            for i in range(self.num_images):
                self.assertIsNone(work.get_wcs(i))

        # Create with a global WCS
        work2 = WorkUnit(self.im_stack, self.config, self.wcs)
        self.assertEqual(work2.im_stack.img_count(), 5)
        self.assertTrue(work2.has_common_wcs())
        self.assertIsNotNone(work2.wcs)
        for i in range(self.num_images):
            self.assertIsNotNone(work2.get_wcs(i))
            self.assertTrue(wcs_fits_equal(self.wcs, work2.get_wcs(i)))

        # Mismatch with the number of WCS.
        self.assertRaises(
            ValueError,
            WorkUnit,
            self.im_stack,
            self.config,
            self.wcs,
            [f"img_{i}" for i in range(self.im_stack.img_count())],
            [self.wcs, self.wcs, self.wcs],
        )

        # Create with per-image WCS that can be compressed to a global WCS.
        per_image_wcs = [self.wcs] * self.num_images
        work3 = WorkUnit(self.im_stack, self.config, per_image_wcs=per_image_wcs)
        self.assertIsNotNone(work3.wcs)
        self.assertTrue(work3.has_common_wcs())
        for i in range(self.num_images):
            self.assertIsNotNone(work3.get_wcs(i))
            self.assertTrue(wcs_fits_equal(self.wcs, work3.get_wcs(i)))

        # Create with per-image WCS that cannot be compressed to a global WCS.
        work3 = WorkUnit(self.im_stack, self.config, per_image_wcs=self.diff_wcs)
        self.assertIsNone(work3.wcs)
        self.assertFalse(work3.has_common_wcs())
        for i in range(self.num_images):
            self.assertIsNotNone(work3.get_wcs(i))
            self.assertTrue(wcs_fits_equal(work3.get_wcs(i), self.diff_wcs[i]))

    def test_save_and_load_fits(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_fits, file_path)

            # Write out the existing WorkUnit with a different per-image wcs for all the entries.
            # work = WorkUnit(self.im_stack, self.config, None, self.diff_wcs)
            work = WorkUnit(im_stack=self.im_stack, config=self.config, wcs=None, per_image_wcs=self.diff_wcs)
            work.to_fits(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_fits(file_path)
            self.assertEqual(work2.im_stack.img_count(), self.num_images)
            self.assertIsNone(work2.wcs)
            self.assertFalse(work2.has_common_wcs())
            for i in range(self.num_images):
                li = work2.im_stack.get_single_image(i)
                self.assertEqual(li.get_width(), self.width)
                self.assertEqual(li.get_height(), self.height)
                self.assertEqual(li.get_obstime(), 59000.0 + (2 * i + 1))

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
                self.assertIsNotNone(work2.get_wcs(i))
                self.assertTrue(wcs_fits_equal(work2.get_wcs(i), self.diff_wcs[i]))

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["im_filepath"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)

            # We throw an error if we try to overwrite a file with overwrite=False
            self.assertRaises(FileExistsError, work.to_fits, file_path)

            # We succeed if overwrite=True
            work.to_fits(file_path, overwrite=True)

    def test_save_and_load_fits_shard(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_sharded_fits, "test_workunit.fits", dir_name)

            # Write out the existing WorkUnit with a different per-image wcs for all the entries.
            # work = WorkUnit(self.im_stack, self.config, None, self.diff_wcs)
            work = WorkUnit(im_stack=self.im_stack, config=self.config, wcs=None, per_image_wcs=self.diff_wcs)
            work.to_sharded_fits("test_workunit.fits", dir_name)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_sharded_fits(filename="test_workunit.fits", directory=dir_name)
            self.assertEqual(work2.im_stack.img_count(), self.num_images)
            self.assertIsNone(work2.wcs)
            self.assertFalse(work2.has_common_wcs())
            for i in range(self.num_images):
                li = work2.im_stack.get_single_image(i)
                self.assertEqual(li.get_width(), self.width)
                self.assertEqual(li.get_height(), self.height)
                self.assertEqual(li.get_obstime(), 59000.0 + (2 * i + 1))

                # Check the three image layers match.
                sci1 = li.get_science()
                var1 = li.get_variance()
                msk1 = li.get_mask()

                li_org = self.im_stack.get_single_image(i)
                sci2 = li_org.get_science()
                var2 = li_org.get_variance()
                msk2 = li_org.get_mask()

                self.assertTrue(sci1.l2_allclose(sci2, 1e-3))

                # Check the PSF layer matches.
                p1 = self.p[i]
                p2 = li.get_psf()
                p1.is_close(p2, 1e-3)

                # No per-image WCS on the odd entries
                self.assertIsNotNone(work2.get_wcs(i))
                self.assertTrue(wcs_fits_equal(work2.get_wcs(i), self.diff_wcs[i]))

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["im_filepath"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)

            # We throw an error if we try to overwrite a file with overwrite=False
            self.assertRaises(FileExistsError, work.to_fits, file_path)

            # We succeed if overwrite=True
            work.to_fits(file_path, overwrite=True)

    def test_save_and_load_fits_shard_lazy(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_sharded_fits, "test_workunit.fits", dir_name)

            # Write out the existing WorkUnit with a different per-image wcs for all the entries.
            # work = WorkUnit(self.im_stack, self.config, None, self.diff_wcs)
            work = WorkUnit(im_stack=self.im_stack, config=self.config, wcs=None, per_image_wcs=self.diff_wcs)
            work.to_sharded_fits("test_workunit.fits", dir_name)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_sharded_fits(filename="test_workunit.fits", directory=dir_name, lazy=True)
            self.assertEqual(len(work2.file_paths), self.num_images)
            self.assertIsNone(work2.wcs)
            self.assertFalse(work2.has_common_wcs())

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["im_filepath"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)
            self.assertEqual(work2.im_stack.img_count(), 0)

            work2.load_images()

            self.assertEqual(work2.im_stack.img_count(), self.num_images)
            self.assertEqual(work2.lazy, False)

    def test_save_and_load_fits_global_wcs(self):
        """This check only confirms that we can read and write the global WCS. The other
        values are tested in test_save_and_load_fits()."""
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit_b.fits")
            work = WorkUnit(self.im_stack, self.config, self.wcs, None)
            work.to_fits(file_path)

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_fits(file_path)
            self.assertIsNotNone(work2.wcs)
            self.assertTrue(work2.has_common_wcs())
            self.assertTrue(wcs_fits_equal(work2.wcs, self.wcs))
            for i in range(self.num_images):
                self.assertIsNotNone(work2.get_wcs(i))
                self.assertTrue(wcs_fits_equal(work2.get_wcs(i), self.wcs))

    def test_image_positions_to_original_icrs_invalid_format(self):
        work = WorkUnit(
            im_stack=self.im_stack,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            per_image_wcs=self.per_image_wcs,
            per_image_ebd_wcs=[self.per_image_ebd_wcs] * self.num_images,
            geocentric_distances=[self.geo_dist] * self.num_images,
            heliocentric_distance=41.0,
            constituent_images=self.constituent_images,
            reprojected=True,
        )

        # Incorrect format for 'xy'
        self.assertRaises(
            ValueError,
            work.image_positions_to_original_icrs,
            [0],
            [("0", "1", "2")],
            "xy",
        )

        # Incorrect format for 'radec'
        self.assertRaises(
            ValueError,
            work.image_positions_to_original_icrs,
            [0],
            [(24, 601)],
            "radec",
        )

        # Incorrect length for positions
        self.assertRaises(
            ValueError,
            work.image_positions_to_original_icrs,
            [0],
            [(24, 601), (0, 1)],
            "xy",
        )

    def test_image_positions_to_original_icrs_basic_inputs(self):
        work = WorkUnit(
            im_stack=self.im_stack,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            per_image_wcs=self.per_image_wcs,
            per_image_ebd_wcs=[self.per_image_ebd_wcs] * self.num_images,
            geocentric_distances=[self.geo_dist] * self.num_images,
            heliocentric_distance=41.0,
            constituent_images=self.constituent_images,
            reprojected=True,
        )

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="radec",
            filter_in_frame=False,
        )

        for r, e in zip(res, self.expected_radec_positions):
            npt.assert_almost_equal(r.separation(e).deg, 0.0, decimal=5)

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.input_radec_positions,
            input_format="radec",
            output_format="radec",
            filter_in_frame=False,
        )

        for r, e in zip(res, self.expected_radec_positions):
            npt.assert_almost_equal(r.separation(e).deg, 0.0, decimal=5)

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="xy",
            filter_in_frame=False,
        )

        for r, e in zip(res, self.expected_pixel_positions):
            rx, ry = r[0]
            ex, ey = e
            npt.assert_almost_equal(rx, ex, decimal=1)
            npt.assert_almost_equal(ry, ey, decimal=1)

    def test_image_positions_to_original_icrs_filtering(self):
        work = WorkUnit(
            im_stack=self.im_stack,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            per_image_wcs=self.per_image_wcs,
            per_image_ebd_wcs=[self.per_image_ebd_wcs] * self.num_images,
            geocentric_distances=[self.geo_dist] * self.num_images,
            heliocentric_distance=41.0,
            constituent_images=self.constituent_images,
            reprojected=True,
        )

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="xy",
            filter_in_frame=True,
        )

        assert res[3] is None
        for r, e in zip(res, self.expected_pixel_positions[:3]):
            rx, ry = r[0]
            ex, ey = e
            npt.assert_almost_equal(rx, ex, decimal=1)
            npt.assert_almost_equal(ry, ey, decimal=1)

    def test_image_positions_to_original_icrs_mosaicking(self):
        work = WorkUnit(
            im_stack=self.im_stack,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            per_image_wcs=self.per_image_wcs,
            per_image_ebd_wcs=[self.per_image_ebd_wcs] * self.num_images,
            geocentric_distances=[self.geo_dist] * self.num_images,
            heliocentric_distance=41.0,
            constituent_images=self.constituent_images,
            reprojected=True,
        )

        new_wcs = make_fake_wcs(190.0, -7.7888, 500, 700)
        work._per_image_wcs[-1] = new_wcs
        work._per_image_indices[3] = [3, 4]

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="xy",
            filter_in_frame=True,
        )

        rx, ry = res[3][0]
        assert rx > 0 and rx < 500
        assert ry > 0 and ry < 700
        assert res[3][1] == "five.fits"

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="radec",
            filter_in_frame=True,
        )

        npt.assert_almost_equal(res[3][0].separation(self.expected_radec_positions[3]).deg, 0.0, decimal=5)
        assert res[3][1] == "five.fits"

        # work._per_image_wcs[4] = work._per_image_wcs[3]
        # work._per_image_ebd_wcs[4] = work._per_image_ebd_wcs[3]

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="xy",
            filter_in_frame=False,
        )

        rx, ry = res[3][0][0]
        ex, ey = self.expected_pixel_positions[3]
        npt.assert_almost_equal(rx, ex, decimal=1)
        npt.assert_almost_equal(ry, ey, decimal=1)
        assert res[3][0][1] == "four.fits"
        assert res[3][1][1] == "five.fits"

    def test_get_unique_obstimes_and_indices(self):
        work = WorkUnit(
            im_stack=self.im_stack,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            per_image_wcs=self.per_image_wcs,
            per_image_ebd_wcs=[self.per_image_ebd_wcs] * self.num_images,
            geocentric_distances=[self.geo_dist] * self.num_images,
            heliocentric_distance=41.0,
            constituent_images=self.constituent_images,
        )
        times = work.get_all_obstimes()
        times[-1] = times[-2]
        work._obstimes = times

        obstimes, indices = work.get_unique_obstimes_and_indices()

        assert len(obstimes) == 4
        assert len(indices) == 4
        assert indices[3] == [3, 4]


if __name__ == "__main__":
    unittest.main()
