import unittest

import unittest
import numpy as np
from astropy.coordinates import EarthLocation
from kbmod import ImageCollection
from kbmod.region_search import RegionSearch, Ephems, Patch, patch_arcmin_to_pixels

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from utils import DECamImdiffFactory


class TestRegionSearch(unittest.TestCase):

    # Factory for generating test images
    fitsFactory = DECamImdiffFactory()

    def setUp(self):
        # Set seed
        np.random.seed(42)
        self.fits = self.fitsFactory.get_n(10)
        self.ic = ImageCollection.fromTargets(self.fits)

        # To simulate chip overlap and multiple visits, we will create a copy of the ImageCollection
        # and stack it multiple times.
        sub_ic = self.ic.copy()
        sub_ic._standardizers = None
        self.ic._standardizers = None
        self.ic["std_idx"] = range(len(self.ic.data))
        sub_ic["std_idx"] = range(len(sub_ic.data))
        self.ic = self.ic.vstack([sub_ic] * 3)
        assert len(self.ic) == 40

        # Apply detector column where detector is just an index for [0: num_detectors)
        num_detectors = 10
        # Cycle through the number of detectors for each visit, and increment the visit each
        # full cycle of detectors
        detectors = np.arange(num_detectors)
        self.ic.data["detector"] = np.tile(detectors, len(self.ic.data) // num_detectors + 1)[
            : len(self.ic.data)
        ]
        self.ic.data["detector"] = self.ic.data["detector"].astype(int)

        # Now every num_detectors rows we will increment the visit
        inc_amt = 0
        for i in range(len(self.ic.data)):
            if i % num_detectors == 0 and i != 0:
                inc_amt += 1
            # Calculate inc_amt seconds in mjd 
            inc_sec = (inc_amt * (1.0 / (24 * 60 * 60)))
            self.ic.data["visit"][i] = self.ic.data["visit"][i] + inc_sec
            # Calculate 1 second in mjd
            self.ic.data["mjd_mid"][i] = self.ic.data["mjd_mid"][i] + inc_sec

        # Create an ephems table to search through
        # Have all of our ephems RA, Decs be near 3 different detectors
        def get_first_row_with_detector(ic, detector):
            return ic.data[ic.data["detector"] == detector][0]
        detectors = list(set(self.ic.data["detector"]))
        test_ras = [
            get_first_row_with_detector(self.ic, detectors[0])["ra"] + 0.1,
            get_first_row_with_detector(self.ic, detectors[0])["ra"] + 0.05,
            get_first_row_with_detector(self.ic, detectors[1])["ra"] + 0.1,
            get_first_row_with_detector(self.ic, detectors[1])["ra"] + 0.05,
            get_first_row_with_detector(self.ic, detectors[2])["ra"] + 0.1,
        ]
        test_decs = [
            get_first_row_with_detector(self.ic, detectors[0])["dec"] + 0.1,
            get_first_row_with_detector(self.ic, detectors[0])["dec"] + 0.05,
            get_first_row_with_detector(self.ic, detectors[1])["dec"] + 0.1,
            get_first_row_with_detector(self.ic, detectors[1])["dec"] + 0.05,
            get_first_row_with_detector(self.ic, detectors[2])["dec"] + 0.1,
        ]


        self.test_ephems = Table()
        curr_ras = []
        curr_decs = []
        curr_mjds = []
        # For simplicity for each ephems test point generated above, insert
        # all of the points at the given time step.
        for j in range(len(test_ras)):
            for i in range(len(self.ic)):
                curr_ras.append(test_ras[j])
                curr_decs.append(test_decs[j])
                curr_mjds.append(self.ic.data["mjd_mid"][i])
        self.test_ephems["mjd_mid"] = curr_mjds
        self.test_ephems["Name"] = "TestObject"
        self.test_ephems["ra"] = curr_ras
        self.test_ephems["dec"] = curr_decs

        # Create a mock EarthLocation
        self.earth_loc = EarthLocation.of_site("ctio")

        # Set the patch size to 20 x 20 arcminutes
        self.patch_size = [20, 20]

        # Filter to a dec range near the range of our test collection.
        self.dec_range = (min(self.ic.data["dec"] - 0.5), max(self.ic.data["dec"] + 0.5))

    def test_init(self):
        ic_cols = len(self.ic.data.columns)
        rs = RegionSearch(self.ic)
        self.assertIsInstance(rs, RegionSearch)
        self.assertEqual(rs.ic, self.ic)
        self.assertEqual(len(rs.guess_dists), 0)
        self.assertEqual(len(rs.ic.data.columns), ic_cols)

    def test_init_with_guess_dists(self):
        guess_dists = [0.1, 0.2, 0.3]
        # Assert that we fail if we don't provide an earth location
        with self.assertRaises(ValueError):
            RegionSearch(self.ic, guess_dists=guess_dists)

        rs = RegionSearch(self.ic, guess_dists=guess_dists, earth_loc=self.earth_loc)
        self.assertIsInstance(rs, RegionSearch)
        self.assertEqual(rs.ic, self.ic)
        self.assertEqual(rs.guess_dists, guess_dists)

    def test_patch_arcmin_to_pixels(self):
        """ Test the conversion of arcminutes to pixels."""
        test_arcmin = [1.0, 2.5, 8, 19.9, 20.0] 
        test_pixel_scale = [0.2, 1.0, 1.3, 2, 3.6]
        test_expected = [300, 150, 370, 597, 334]
        for arcmin, pixel_scale, expected in zip(test_arcmin, test_pixel_scale, test_expected):
            result = patch_arcmin_to_pixels(arcmin, pixel_scale)
            self.assertAlmostEqual(result, expected)

    def test_patch_creation(self):
        guess_dists = [0.1, 0.2, 0.3]
        rs = RegionSearch(self.ic, guess_dists=guess_dists, earth_loc=self.earth_loc)
        rs.generate_patches(
            arcminutes=self.patch_size[0],
            overlap_percentage=0,
            image_width=patch_arcmin_to_pixels(self.patch_size[0], 0.2),
            image_height=patch_arcmin_to_pixels(self.patch_size[1], 0.2),
            pixel_scale=0.2,
            dec_range=(-5, 5),
        )
        for patch in rs.get_patches():
            self.assertIsInstance(patch, Patch)
            self.assertEqual(patch.pixel_scale, 0.2)
            self.assertGreaterEqual(patch.dec, -5)
            self.assertLessEqual(patch.dec, 5)
        n_patches = len(rs.get_patches())
        rs.generate_patches(
            arcminutes=self.patch_size[0],
            overlap_percentage=50,
            image_width=patch_arcmin_to_pixels(self.patch_size[0], 0.2),
            image_height=patch_arcmin_to_pixels(self.patch_size[1], 0.2),
            pixel_scale=0.2,
            dec_range=(-5, 5),
        )
        # Because we generate patches with 50% overlap along both RA and Dec dimensions,
        # we should now have 4 times the number of patches as before.
        self.assertEqual(len(rs.get_patches()), n_patches * 4)

    def test_search_patches_by_ephems(self):
        region_search_test = RegionSearch(self.ic, guess_dists=[], earth_loc=self.earth_loc)
        region_search_test.generate_patches(
            arcminutes=self.patch_size[0],
            overlap_percentage=0,
            image_width=patch_arcmin_to_pixels(self.patch_size[0], 0.2),
            image_height=patch_arcmin_to_pixels(self.patch_size[1], 0.2),
            pixel_scale=0.2,
            dec_range=self.dec_range,
        )

        region_search_test_ephems = Ephems(
            self.test_ephems,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=[],
            earth_loc=EarthLocation.of_site("ctio"),
        )

        region_search_test.filter_by_mjds(self.test_ephems["mjd_mid"], time_sep_s=60.0)

        found_test_patches = region_search_test.search_patches_by_ephems(region_search_test_ephems)
        self.assertGreater(len(found_test_patches), 0)
        self.assertGreater(len(region_search_test.ic), 0)

        for patch_id in found_test_patches:
            ic = region_search_test.get_image_collection_from_patch(patch_id)
            self.assertGreater(len(ic), 0)

    def test_reflex_corrected_search_patches_by_ephems(self):
        # We test reflex-correction by ensuring that a) we detect a new set of patches
        # B) wcs.world_to_pixel returns valid pixel coordinates in the resulting ImageCollection

        test_dists = [5.0, 39.0]
        region_search_test = RegionSearch(self.ic, guess_dists=test_dists, earth_loc=self.earth_loc)
        region_search_test.generate_patches(
            arcminutes=self.patch_size[0],
            overlap_percentage=0,
            image_width=patch_arcmin_to_pixels(self.patch_size[0], 0.2),
            image_height=patch_arcmin_to_pixels(self.patch_size[1], 0.2),
            pixel_scale=0.2,
            dec_range=self.dec_range,
        )

        region_search_test_ephems = Ephems(
            self.test_ephems,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=test_dists,
            earth_loc=EarthLocation.of_site("ctio"),
        )

        region_search_test.filter_by_mjds(self.test_ephems["mjd_mid"], time_sep_s=60.0)

        for test_dist in test_dists:
            # Check that for the ic we generate all of our columns correctly for this test_dist
            self.assertIn(
                region_search_test.ic.reflex_corrected_col("ra", test_dist),
                region_search_test.ic.data.columns,
            )
            self.assertIn(
                region_search_test.ic.reflex_corrected_col("dec", test_dist),
                region_search_test.ic.data.columns,
            )
            if test_dist == 0.0:
                self.assertEqual(region_search_test.ic.reflex_corrected_col("ra", test_dist), "ra")
                self.assertEqual(region_search_test.ic.reflex_corrected_col("dec", test_dist), "dec")

            found_test_patches = region_search_test.search_patches_by_ephems(region_search_test_ephems, guess_dist=test_dist)

            self.assertGreater(len(found_test_patches), 0)
            self.assertGreater(len(self.ic), 0)
            self.assertGreater(len(region_search_test.ic), 0)

            for patch_id in found_test_patches:
                patch_ic = region_search_test.get_image_collection_from_patch(patch_id, guess_dist=test_dist, min_overlap=0)
                self.assertGreater(len(patch_ic), 0)
                # Check the applied the WCS of the ImageCollection
                self.assertEqual(len(set(patch_ic.data["global_wcs_pixel_shape_0"])), 1)
                self.assertEqual(len(set(patch_ic.data["global_wcs_pixel_shape_1"])), 1)
                self.assertEqual(
                    patch_ic.data["global_wcs_pixel_shape_0"][0],
                    patch_arcmin_to_pixels(self.patch_size[0], 0.2),
                )
                self.assertEqual(
                    patch_ic.data["global_wcs_pixel_shape_1"][0],
                    patch_arcmin_to_pixels(self.patch_size[0], 0.2),
                )

                # Check that the WCS is valid
                self.assertEqual(len(set(patch_ic.data["global_wcs"])), 1)
                wcs = WCS(patch_ic.data["global_wcs"][0])
                # Check that the WCS is valid
                self.assertIsInstance(wcs, WCS)

                # Assert that each corner of our patch is within the bounds of the WCS
                for ra, dec in region_search_test.get_patch(patch_id).corners:
                    x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs"))
                    pixel_discrep = 2
                    self.assertGreaterEqual(x, 0 - pixel_discrep)
                    self.assertLessEqual(x, patch_arcmin_to_pixels(self.patch_size[0], 0.2) + pixel_discrep)
                    self.assertGreaterEqual(y, 0 - pixel_discrep)
                    self.assertLessEqual(y, patch_arcmin_to_pixels(self.patch_size[1], 0.2) + pixel_discrep)

                # Check that each pre-existing column of the original ImageCollection is present
                for col in self.ic.data.columns:
                    self.assertIn(col, patch_ic.data.columns)

                visit_detectors = set(zip(patch_ic.data["visit"], patch_ic.data["detector"]))
                self.assertEqual(len(visit_detectors), len(patch_ic.data))

                # Check that the patch_ic data matches the original ic data for each visit-detector combination.
                cols_changed_by_slicing = set(["std_idx", "ext_idx", "std_name", "config"])
                for patch_idx in range(len(patch_ic.data)):
                    patch_row = patch_ic[patch_idx]
                    orig_ic_row = self.ic.data[
                        (self.ic.data["visit"] == patch_row["visit"])
                        & (self.ic.data["detector"] == patch_row["detector"])
                    ][0]
                    for col in self.ic.data.columns:
                        if col in cols_changed_by_slicing:
                            continue
                        self.assertEqual(orig_ic_row[col], patch_row[col])
                    # Assert that the image has a non-zero overlap with the patch
                    self.assertGreater(patch_row["overlap_deg"], 0)

                # Build a smaller ImageCollection from the patch with the images that have the highest overlap
                small_ic = region_search_test.get_image_collection_from_patch(patch_id, guess_dist=test_dist, min_overlap=0, max_images=3)
                self.assertEqual(len(small_ic), 3)

                # Check that the overlap_deg column is sorted in descending order
                self.assertTrue(np.all(np.diff(small_ic.data["overlap_deg"]) <= 0))

                # Check that we included the image with the highest degree of overlap when cutting down the images
                # which confirms that we chose the highest overlap.
                self.assertEqual(small_ic.data["overlap_deg"][0], max(patch_ic.data["overlap_deg"]))

    def test_patch_overlap(self):
        """
        Test the patch overlap calculation.
        """
        # Create a patch with a known overlap
        patch1 = Patch(
            center_ra=10.0,
            center_dec=10.0,
            width=5.0,
            height=5.0,
            pixel_scale=0.2,
            image_width=100,
            image_height=100,
            id=1,
        )
        patch2 = Patch(
            center_ra=12.5,
            center_dec=12.5,
            width=5.0,
            height=5.0,
            pixel_scale=0.2,
            image_width=100,
            image_height=100,
            id=2,
        )
        overlap = patch1.calculate_overlap(patch2)
        self.assertAlmostEqual(overlap, 25.0)
        self.assertTrue(patch1.overlaps_polygon(patch2))
       
        # Test with a patch that has no overlap
        patch3 = Patch(
            center_ra=20.0,
            center_dec=20.0,
            width=5.0,
            height=5.0,
            pixel_scale=0.2,
            image_width=100,
            image_height=100,
            id=3,
        )
        overlap = patch1.calculate_overlap(patch3)
        self.assertEqual(overlap, 0.0)
        self.assertFalse(patch1.overlaps_polygon(patch3))

