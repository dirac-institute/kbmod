from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

import numpy as np
import numpy.testing as npt
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import make_fake_image_stack
from kbmod.core.psf import PSF
from kbmod.image_utils import image_stack_py_to_cpp
import kbmod.search as kb
from kbmod.reprojection_utils import fit_barycentric_wcs
from kbmod.wcs_utils import make_fake_wcs, wcs_fits_equal
from kbmod.work_unit import (
    create_image_metadata,
    hdu_to_image_metadata_table,
    image_metadata_table_to_hdu,
    WorkUnit,
)


class test_work_unit(unittest.TestCase):
    def setUp(self):
        self.num_images = 5
        self.width = 50
        self.height = 70
        self.images = [None] * self.num_images
        self.psfs = [PSF.make_gaussian_kernel(5.0 / float(2 * i + 1)) for i in range(self.num_images)]
        self.times = [59000.0 + (2.0 * i + 1.0) for i in range(self.num_images)]

        rng = np.random.default_rng(1002)
        self.im_stack_py = make_fake_image_stack(
            self.height,
            self.width,
            self.times,
            noise_level=2.0,
            psfs=self.psfs,
            rng=rng,
        )

        # Mask one of the pixels in each image.  This is done directly to the science
        # and variance layers since ImageStackPy does not have a separate mask layer.
        for i in range(self.num_images):
            self.im_stack_py.sci[i][10, 10 + i] = np.nan
            self.im_stack_py.var[i][10, 10 + i] = np.nan

        self.config = SearchConfiguration()
        self.config.set("result_filename", "Here")
        self.config.set("num_obs", self.num_images)
        self.config.set("results_per_pixel", 8)

        # Create a fake WCS
        self.wcs = make_fake_wcs(200.6145, -7.7888, 500, 700, 0.00027)
        self.per_image_wcs = [self.wcs for _ in range(self.num_images)]

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

        self.org_image_meta = Table(
            {
                "data_loc": np.array(self.constituent_images),
                "ebd_wcs": np.array([self.per_image_ebd_wcs] * self.num_images),
                "geocentric_distance": np.array([self.geo_dist] * self.num_images),
                "per_image_wcs": np.array(self.per_image_wcs),
            }
        )

    def test_create(self):
        # Test the creation of a WorkUnit with no WCS. Should throw a warning.
        with warnings.catch_warnings(record=True) as wrn:
            warnings.simplefilter("always")
            work = WorkUnit(self.im_stack_py, self.config)

            self.assertIsNotNone(work)
            self.assertEqual(work.im_stack.num_times, 5)
            self.assertEqual(work.config["result_filename"], "Here")
            self.assertEqual(work.config["num_obs"], 5)
            self.assertIsNone(work.wcs)
            self.assertEqual(len(work), self.num_images)
            for i in range(self.num_images):
                self.assertIsNone(work.get_wcs(i))

        # Create with a global WCS
        work2 = WorkUnit(self.im_stack_py, self.config, self.wcs)
        self.assertEqual(work2.im_stack.num_times, 5)
        self.assertIsNotNone(work2.wcs)
        for i in range(self.num_images):
            self.assertIsNotNone(work2.get_wcs(i))
            self.assertTrue(wcs_fits_equal(self.wcs, work2.get_wcs(i)))

    def test_metadata_helpers(self):
        """Test that we can roundtrip an astropy table of metadata (including) WCS
        into a BinTableHDU.
        """
        metadata_dict = {
            "col1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Floats
            "uri": np.array(["a", "bc", "def", "ghij", "other_strings"]),  # Strings
            "wcs": np.array(self.per_image_wcs),  # WCSes
            "none_col": np.array([None] * self.num_images),  # Empty column
            "Other": np.arange(5),  # ints
        }
        metadata_table = Table(metadata_dict)

        # Convert to an HDU
        hdu = image_metadata_table_to_hdu(metadata_table)
        self.assertIsNotNone(hdu)

        # Convert it back. We should have dropped the column of all None.
        md_table2 = hdu_to_image_metadata_table(hdu)
        self.assertEqual(len(md_table2.colnames), 4)
        npt.assert_array_equal(metadata_dict["col1"], md_table2["col1"])
        npt.assert_array_equal(metadata_dict["uri"], md_table2["uri"])
        npt.assert_array_equal(metadata_dict["Other"], md_table2["Other"])
        self.assertFalse("none_col" in md_table2.colnames)
        for i in range(len(md_table2)):
            self.assertTrue(isinstance(md_table2["wcs"][i], WCS))

    def test_create_image_metadata(self):
        # Empty constituent image data.
        org_img_meta = create_image_metadata(3, data=None)
        self.assertEqual(len(org_img_meta), 3)
        self.assertTrue("data_loc" in org_img_meta.colnames)
        self.assertTrue("ebd_wcs" in org_img_meta.colnames)
        self.assertTrue("geocentric_distance" in org_img_meta.colnames)
        self.assertTrue("per_image_wcs" in org_img_meta.colnames)

        # We can create from a Table.
        data = Table(
            {
                "uri": ["file1", "file2", "file3"],
                "geocentric_distance": [1.0, 2.0, 3.0],
            }
        )
        org_img_meta2 = create_image_metadata(3, data)
        self.assertEqual(len(org_img_meta2), 3)
        self.assertTrue("data_loc" in org_img_meta2.colnames)
        self.assertTrue("ebd_wcs" in org_img_meta2.colnames)
        self.assertTrue("geocentric_distance" in org_img_meta2.colnames)
        self.assertTrue("per_image_wcs" in org_img_meta2.colnames)
        self.assertTrue("uri" in org_img_meta2.colnames)

        npt.assert_array_equal(org_img_meta2["geocentric_distance"], data["geocentric_distance"])
        npt.assert_array_equal(org_img_meta2["uri"], data["uri"])
        self.assertTrue(np.all(org_img_meta2["ebd_wcs"] == None))
        self.assertTrue(np.all(org_img_meta2["per_image_wcs"] == None))
        self.assertTrue(np.all(org_img_meta2["data_loc"] == None))

        # We need a positive number of images that matches the length of data (if provided).
        self.assertRaises(ValueError, create_image_metadata, -1, None)
        self.assertRaises(ValueError, create_image_metadata, 2, data)

    def test_save_and_load_fits(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_fits, file_path)

            # Write out the existing WorkUnit with a different per-image wcs for all the entries.
            extra_meta = {
                "data_loc": np.array(self.constituent_images),
                "int_index": np.arange(self.num_images),
                "uri": np.array([f"file_loc_{i}" for i in range(self.num_images)]),
            }
            work = WorkUnit(
                im_stack=self.im_stack_py,
                config=self.config,
                wcs=None,
                per_image_wcs=self.diff_wcs,
                org_image_meta=Table(extra_meta),
            )
            work.to_fits(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_fits(file_path, show_progress=False)
            self.assertEqual(work2.im_stack.num_times, self.num_images)
            self.assertIsNone(work2.wcs)
            for i in range(self.num_images):
                self.assertEqual(work2.im_stack.times[i], 59000.0 + (2 * i + 1))

                # Check the three image layers match. We use more permissive values for science and
                # variance because of quantization during compression.
                self.assertTrue(
                    np.allclose(work2.im_stack.sci[i], self.im_stack_py.sci[i], atol=0.05, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(work2.im_stack.var[i], self.im_stack_py.var[i], atol=0.05, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(
                        work2.im_stack.get_mask(i), self.im_stack_py.get_mask(i), atol=0.001, equal_nan=True
                    )
                )

                # Check the PSF layer matches.
                p1 = self.psfs[i]
                p2 = work2.im_stack.psfs[i]
                npt.assert_array_almost_equal(p1, p2, decimal=3)

                # No per-image WCS on the odd entries
                self.assertIsNotNone(work2.get_wcs(i))
                self.assertTrue(wcs_fits_equal(work2.get_wcs(i), self.diff_wcs[i]))

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["result_filename"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)

            # Check that we retrieved the extra metadata that we added.
            npt.assert_array_equal(work2.get_constituent_meta("uri"), extra_meta["uri"])
            npt.assert_array_equal(work2.get_constituent_meta("int_index"), extra_meta["int_index"])
            npt.assert_array_equal(work2.get_constituent_meta("data_loc"), self.constituent_images)

            # Check that we can retrieve the extra metadata in a single request.
            meta2 = work2.get_constituent_meta(["uri", "int_index", "nonexistent_column"])
            self.assertEqual(len(meta2), 2)
            npt.assert_array_equal(meta2["uri"], extra_meta["uri"])
            npt.assert_array_equal(meta2["int_index"], extra_meta["int_index"])
            self.assertFalse("nonexistent_column" in extra_meta)

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
            work = WorkUnit(
                im_stack=self.im_stack_py, config=self.config, wcs=None, per_image_wcs=self.diff_wcs
            )
            work.to_sharded_fits("test_workunit.fits", dir_name)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_sharded_fits(filename="test_workunit.fits", directory=dir_name)
            self.assertEqual(work2.im_stack.num_times, self.num_images)
            self.assertIsNone(work2.wcs)
            for i in range(self.num_images):
                self.assertEqual(work2.im_stack.times[i], 59000.0 + (2 * i + 1))

                # Check the three image layers match. We use more permissive values for science and
                # variance because of quantization during compression.
                self.assertTrue(
                    np.allclose(work2.im_stack.sci[i], self.im_stack_py.sci[i], atol=0.05, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(work2.im_stack.var[i], self.im_stack_py.var[i], atol=0.05, equal_nan=True)
                )
                self.assertTrue(
                    np.allclose(
                        work2.im_stack.get_mask(i), self.im_stack_py.get_mask(i), atol=0.001, equal_nan=True
                    )
                )

                # Check the PSF layer matches.
                p1 = self.psfs[i]
                p2 = work2.im_stack.psfs[i]
                npt.assert_array_almost_equal(p1, p2, decimal=3)

                # No per-image WCS on the odd entries
                self.assertIsNotNone(work2.get_wcs(i))
                self.assertTrue(wcs_fits_equal(work2.get_wcs(i), self.diff_wcs[i]))

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["result_filename"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)

            # We throw an error if we try to overwrite a file with overwrite=False
            self.assertRaises(FileExistsError, work.to_fits, file_path)

            # We succeed if overwrite=True
            work.to_sharded_fits("test_workunit.fits", dir_name, overwrite=True)

    def test_save_and_load_fits_shard_lazy(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_sharded_fits, "test_workunit.fits", dir_name)

            # Write out the existing WorkUnit with a different per-image wcs for all the entries.
            work = WorkUnit(
                im_stack=self.im_stack_py, config=self.config, wcs=None, per_image_wcs=self.diff_wcs
            )
            work.to_sharded_fits("test_workunit.fits", dir_name)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_sharded_fits(filename="test_workunit.fits", directory=dir_name, lazy=True)
            self.assertEqual(len(work2.file_paths), self.num_images)
            self.assertIsNone(work2.wcs)

            # Check that we read in the configuration values correctly.
            self.assertEqual(work2.config["result_filename"], "Here")
            self.assertEqual(work2.config["num_obs"], self.num_images)
            self.assertEqual(work2.im_stack.num_times, 0)

            work2.load_images()

            self.assertEqual(work2.im_stack.num_times, self.num_images)
            self.assertEqual(work2.lazy, False)

    def test_save_and_load_fits_global_wcs(self):
        """This check only confirms that we can read and write the global WCS. The other
        values are tested in test_save_and_load_fits()."""
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit_b.fits")
            work = WorkUnit(
                self.im_stack_py,
                self.config,
                self.wcs,
                None,
                reprojected=True,
                reprojection_frame="original",
            )
            work.to_fits(file_path)

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_fits(file_path, show_progress=False)
            self.assertIsNotNone(work2.wcs)
            self.assertTrue(work2.reprojected)
            self.assertIsNotNone(work2.reprojection_frame)
            self.assertTrue(wcs_fits_equal(work2.wcs, self.wcs))
            for i in range(self.num_images):
                self.assertIsNotNone(work2.get_wcs(i))
                self.assertTrue(wcs_fits_equal(work2.get_wcs(i), self.wcs))

    def test_save_and_load_fits_shard_lazy_global_wcs(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, WorkUnit.from_sharded_fits, "test_workunit.fits", dir_name)

            # Write out the existing WorkUnit with a different per-image wcs for all the entries.
            work = WorkUnit(
                im_stack=self.im_stack_py,
                config=self.config,
                wcs=self.wcs,
                per_image_wcs=self.diff_wcs,
                reprojected=True,
                reprojection_frame="original",
            )
            work.to_sharded_fits("test_workunit.fits", dir_name)
            self.assertTrue(Path(file_path).is_file())

            # Read in the file and check that the values agree.
            work2 = WorkUnit.from_sharded_fits(filename="test_workunit.fits", directory=dir_name, lazy=True)
            self.assertEqual(len(work2.file_paths), self.num_images)
            self.assertTrue(work2.reprojected)
            self.assertEqual(work2.reprojection_frame, "original")
            self.assertTrue(wcs_fits_equal(work2.wcs, self.wcs))

    def test_get_ecliptic_angle(self):
        """Check that we can compute an ecliptic angle."""
        work = WorkUnit(self.im_stack_py, self.config, self.wcs, None)
        self.assertAlmostEqual(work.compute_ecliptic_angle(), -0.381541020495931)

        # If we do not have a WCS, we get None for the ecliptic angle.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            work2 = WorkUnit(self.im_stack_py, self.config, None, None)
            self.assertIsNone(work2.compute_ecliptic_angle())

    def test_image_positions_to_original_icrs_invalid_format(self):
        work = WorkUnit(
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            barycentric_distance=41.0,
            reprojected=True,
            org_image_meta=self.org_image_meta,
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
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            barycentric_distance=41.0,
            reprojected=True,
            reprojection_frame="ebd",
            org_image_meta=self.org_image_meta,
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
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            barycentric_distance=41.0,
            reprojected=True,
            reprojection_frame="ebd",
            org_image_meta=self.org_image_meta,
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
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            barycentric_distance=41.0,
            reprojected=True,
            reprojection_frame="ebd",
            org_image_meta=self.org_image_meta,
        )

        new_wcs = make_fake_wcs(190.0, -7.7888, 500, 700)
        work.org_img_meta["per_image_wcs"][-1] = new_wcs
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

    def test_image_positions_to_original_icrs_non_ebd(self):
        work = WorkUnit(
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.wcs,
            barycentric_distance=41.0,
            reprojected=True,
            reprojection_frame="original",
            org_image_meta=self.org_image_meta,
        )

        res = work.image_positions_to_original_icrs(
            self.indices,
            self.pixel_positions,
            input_format="xy",
            output_format="xy",
            filter_in_frame=False,
        )

        # since our test `WorkUnit` uses the same wcs for
        # the original per_image_wcs and then shifts them
        # for the ebd case (effectively opposite our normal
        # order of operations) we check that the xy->xy
        # transformation is a no-op when frame is original.
        for r, e in zip(res, self.pixel_positions):
            rx, ry = r[0]
            ex, ey = e
            npt.assert_almost_equal(rx, ex, decimal=1)
            npt.assert_almost_equal(ry, ey, decimal=1)

    def test_get_unique_obstimes_and_indices(self):
        work = WorkUnit(
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            barycentric_distance=41.0,
            org_image_meta=self.org_image_meta,
        )
        times = work.get_all_obstimes()
        times[-1] = times[-2]
        work._obstimes = times

        obstimes, indices = work.get_unique_obstimes_and_indices()

        assert len(obstimes) == 4
        assert len(indices) == 4
        assert indices[3] == [3, 4]

    def test_get_pixel_coordinates_global(self):
        simple_wcs = make_fake_wcs(200.5, -7.5, 500, 700, 0.01)
        work = WorkUnit(
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=simple_wcs,
        )

        # Compute the pixel locations of the SkyCoords.
        ra = np.array([200.5, 200.55, 200.6])
        dec = np.array([-7.5, -7.55, -7.60])
        expected_x = np.array([249, 254, 259])
        expected_y = np.array([349, 344, 339])

        x_pos, y_pos = work.get_pixel_coordinates(ra, dec)
        np.testing.assert_allclose(x_pos, expected_x, atol=0.2)
        np.testing.assert_allclose(y_pos, expected_y, atol=0.2)

        # We see an error if the arrays are the wrong length.
        self.assertRaises(ValueError, work.get_pixel_coordinates, ra, np.array([-7.7888, -7.79015]))

    def test_get_pixel_coordinates_per_image(self):
        per_wcs = [make_fake_wcs(200.5 + 0.5 * i, -7.5, 500, 700, 0.01) for i in range(self.num_images)]
        obstimes = [float(i) for i in range(self.num_images)]
        work = WorkUnit(
            im_stack=self.im_stack_py,
            config=self.config,
            per_image_wcs=per_wcs,
            obstimes=obstimes,
        )

        # Compute the pixel locations of the SkyCoords.
        ra = np.array([200.5 + 0.5 * i for i in range(self.num_images)])
        dec = np.array([-7.5 + 0.05 * i for i in range(self.num_images)])

        expected_x = np.full(self.num_images, 249)
        expected_y = np.array([349 + 5 * i for i in range(self.num_images)])

        x_pos, y_pos = work.get_pixel_coordinates(ra, dec)
        np.testing.assert_allclose(x_pos, expected_x, atol=0.2)
        np.testing.assert_allclose(y_pos, expected_y, atol=0.2)

        # Test that we can query only a subset of the images.
        x_pos, y_pos = work.get_pixel_coordinates(
            np.array([201.0, 202.0]),  # RA
            np.array([-7.45, -7.35]),  # dec
            np.array([1.0, 3.0]),  # time
        )
        np.testing.assert_allclose(x_pos, [249, 249], atol=0.2)
        np.testing.assert_allclose(y_pos, [354, 364], atol=0.2)

        # We see an error if a time is nowhere near any of the images.
        self.assertRaises(
            ValueError,
            work.get_pixel_coordinates,
            np.array([201.0, 202.0]),  # RA
            np.array([-7.45, -7.35]),  # dec
            np.array([1.0, 300.0]),  # time
        )

    def test_clear_metadata(self):
        """Test that we can clear the metadata from a WorkUnit."""
        work = WorkUnit(
            im_stack=self.im_stack_py,
            config=self.config,
            wcs=self.per_image_ebd_wcs,
            barycentric_distance=41.0,
            org_image_meta=self.org_image_meta,
        )
        # Change the per image metadata to something other than the default
        work._per_image_indices[3] = [3, 4]
        default_img_indices = [[i] for i in range(self.num_images)]

        # Check that the metadata is present.
        self.assertEqual(len(work.org_img_meta), self.num_images)

        # Check thet the per_image_idices are not a single single mapping
        # to the original image.
        self.assertNotEqual(work._per_image_indices, default_img_indices)

        # Clear the metadata.
        work.clear_metadata()

        # Check that the metadata has been cleared.
        self.assertEqual(len(work.org_img_meta), 0)
        self.assertEqual(len(work.org_img_meta.columns), 0)
        self.assertEqual(work._per_image_indices, default_img_indices)

    def test_disorder_obstimes(self):
        # Check that we can disorder the obstimes.
        test_times = [
            [59000.0 + (2 * i + 1) for i in range(self.num_images)],
            [59000.0, 59001.0, 59002.0, 59003.0, 59004.0],
            [59000.0, 59004.0, 59002.0, 59001.0, 59004.0],  # Duplicates
            [59000.0, 59001.62, 59002.0, 59001.62, 59002.8],  # Duplicates
        ]
        for curr_times in test_times:
            # Update the obstimes in the ImageStack
            # assert that the number of times is the same
            self.assertEqual(len(curr_times), self.num_images)
            for i in range(self.num_images):
                self.im_stack_py.times[i] = curr_times[i]

            work = WorkUnit(
                im_stack=self.im_stack_py,
                config=self.config,
                wcs=self.per_image_ebd_wcs,
                barycentric_distance=41.0,
                org_image_meta=self.org_image_meta,
            )
            # Change the per image metadata to something other than the default
            work._per_image_indices[3] = [3, 4]

            # Set numpy random seed
            np.random.seed(0)

            # Check that the obstimes are in order.
            obstimes = work.get_all_obstimes()
            # Disorder the obstimes.
            work.disorder_obstimes()

            # Check that the obstimes have changed
            disordered_obstimes = work.get_all_obstimes()
            self.assertFalse(np.array_equal(disordered_obstimes, obstimes))

            # Check that the range of obstimes is unchanged
            self.assertGreaterEqual(min(disordered_obstimes), min(obstimes))
            time_range = max(max(obstimes) - min(obstimes), self.num_images)
            self.assertLessEqual(max(disordered_obstimes), max(obstimes) + time_range)

            # Assert that the disordered obstimes are now sorted
            self.assertTrue(
                np.array_equal(
                    sorted(disordered_obstimes),
                    disordered_obstimes,
                )
            )

            # Check that uniqueness is preserved by comparing the frequency maps of obstimes
            disordered_obstimes_freq = {}
            obstime_freq = {}
            for obstime in obstimes:
                if obstime not in obstime_freq:
                    obstime_freq[obstime] = 0
                obstime_freq[obstime] += 1

            for obstime in disordered_obstimes:
                if obstime not in disordered_obstimes_freq:
                    disordered_obstimes_freq[obstime] = 0
                disordered_obstimes_freq[obstime] += 1

            self.assertTrue(
                np.array_equal(
                    sorted(obstime_freq.values()),
                    sorted(disordered_obstimes_freq.values()),
                )
            )

            # Check that old metadata was cleared
            self.assertEqual(len(work.org_img_meta), 0)
            self.assertEqual(len(work.org_img_meta.columns), 0)
            self.assertEqual(work._per_image_indices, [[i] for i in range(self.num_images)])


if __name__ == "__main__":
    unittest.main()
