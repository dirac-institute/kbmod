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
    def setUp(self):
        # Set seed
        np.random.seed(42)
        # Factory for generating test images
        self.fits = DECamImdiffFactory().get_n(10)
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

        # Now every num_detectors rows increment the visit and mjd_mid
        inc_amt = 0
        for i in range(len(self.ic.data)):
            if i % num_detectors == 0 and i != 0:
                inc_amt += 30
            self.ic.data["visit"][i] = self.ic.data["visit"][i] + inc_amt
            self.ic.data["mjd_mid"][i] = self.ic.data["mjd_mid"][i] + inc_amt

        # Create an ephems table to search through
        # Have all of our ephems RA, Decs be near 3 different detectors
        detector_1 = self.ic.data[self.ic.data["detector"] == 0][0]
        detector_2 = self.ic.data[self.ic.data["detector"] == 1][0]
        detector_3 = self.ic.data[self.ic.data["detector"] == 2][0]
        test_ras = [
            detector_1["ra"] + 0.1,
            detector_1["ra"] + 0.05,
            detector_2["ra"] + 0.1,
            detector_2["ra"] + 0.05,
            detector_3["ra"] + 0.1,
        ]
        test_decs = [
            detector_1["dec"] + 0.1,
            detector_1["dec"] + 0.05,
            detector_2["dec"] + 0.1,
            detector_2["dec"] + 0.05,
            detector_3["dec"] + 0.1,
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

        # Set the patch size to a square with 20 arcminutes sides.
        self.patch_size = 20

    def test_init(self):
        """Test basic initialization of the RegionSearch class."""
        ic_cols = len(self.ic.data.columns)
        rs = RegionSearch(self.ic)
        self.assertIsInstance(rs, RegionSearch)
        self.assertEqual(rs.ic, self.ic)
        self.assertEqual(len(rs.guess_dists), 0)
        self.assertEqual(len(rs.ic.data.columns), ic_cols)

    def test_init_with_guess_dists(self):
        """Test initialization of the RegionSearch class with guess_dists."""
        guess_dists = [0.1, 0.2, 0.3]
        # Assert that we fail if we don't provide an earth location
        with self.assertRaises(ValueError):
            RegionSearch(self.ic, guess_dists=guess_dists)

        rs = RegionSearch(self.ic, guess_dists=guess_dists, earth_loc=self.earth_loc)
        self.assertIsInstance(rs, RegionSearch)
        self.assertEqual(rs.ic, self.ic)
        self.assertEqual(rs.guess_dists, guess_dists)

        # Test that we now have reflex-corrected columns in the ImageCollection
        for dist in guess_dists:
            self.assertIn(rs.ic.reflex_corrected_col("ra", dist), rs.ic.data.columns)
            self.assertIn(rs.ic.reflex_corrected_col("dec", dist), rs.ic.data.columns)

    def test_patch_arcmin_to_pixels(self):
        """Test the conversion of arcminutes to pixels."""
        test_arcmin = [1.0, 2.5, 8, 19.9, 20.0]
        test_pixel_scale = [0.2, 1.0, 1.3, 2, 3.6]
        test_expected = [300, 150, 370, 597, 334]
        for arcmin, pixel_scale, expected in zip(test_arcmin, test_pixel_scale, test_expected):
            result = patch_arcmin_to_pixels(arcmin, pixel_scale)
            self.assertAlmostEqual(result, expected)

    def test_patch_creation(self):
        # Guess dists to provide (note that this shouldn't affect the numenr of patches)
        guess_dists = [0.1, 0.2, 0.3]
        rs = RegionSearch(self.ic, guess_dists=guess_dists, earth_loc=self.earth_loc)
        rs.generate_patches(
            arcminutes=self.patch_size,
            overlap_percentage=0,
            pixel_scale=0.2,
            dec_range=(-5, 5),
        )
        for patch in rs.get_patches():
            # Assert basic properties of each patch
            self.assertIsInstance(patch, Patch)
            self.assertEqual(patch.pixel_scale, 0.2)
            self.assertGreaterEqual(patch.dec, -5)
            self.assertLessEqual(patch.dec, 5)

        # Now generate the patch grid again with 50% overlap
        n_patches = len(rs.get_patches())
        rs.generate_patches(
            arcminutes=self.patch_size,
            overlap_percentage=50,
            pixel_scale=0.2,
            dec_range=(-5, 5),
        )
        # Because we generate patches with 50% overlap along both RA and Dec dimensions,
        # we should now have 4 times the number of patches as before.
        self.assertEqual(len(rs.get_patches()), n_patches * 4)

    def test_search_patches_by_ephems(self):
        """Test using a ephemeris to search patches and find an ImageCollection."""
        region_search_test = RegionSearch(self.ic, guess_dists=[], earth_loc=self.earth_loc)

        # Generate a basic patch grid with no overlap
        dec_range = (min(self.ic.data["dec"] - 0.5), max(self.ic.data["dec"] + 0.5))
        region_search_test.generate_patches(
            arcminutes=self.patch_size,
            overlap_percentage=0,
            pixel_scale=0.2,
            dec_range=dec_range,
        )

        # Create an ephemeris object to search through, providing some
        # guess distances to reflex-correct even though the search below
        # won't be reflex-corrected.
        region_search_test_ephems = Ephems(
            self.test_ephems,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=[5.0, 50.0],
            earth_loc=EarthLocation.of_site("ctio"),
        )

        # Filter our ImageCollection to times near the ephemeris times
        region_search_test.filter_by_mjds(self.test_ephems["mjd_mid"], time_sep_s=60.0)

        # Search for patches that contain the ephemeris points
        found_test_patches = region_search_test.search_patches_by_ephems(region_search_test_ephems)
        self.assertGreater(len(found_test_patches), 0)
        self.assertGreater(len(region_search_test.ic), 0)

        # Validate that each patch can export an ImageCollection
        for patch_id in found_test_patches:
            min_overlap = 0.000001  # Min overlap in square degrees
            ic = region_search_test.get_image_collection_from_patch(patch_id, min_overlap=min_overlap)
            self.assertGreater(len(ic), 0)
            # Check that the overlap_deg column is all greater than 0
            self.assertGreater(ic.data["overlap_deg"].min(), min_overlap)

    def test_search_patches_by_ephems_mapping(self):
        """Test search_patches_by_ephems with map_obj_to_patches=True."""
        region_search_test = RegionSearch(self.ic, guess_dists=[], earth_loc=self.earth_loc)

        # Generate a basic patch grid
        dec_range = (min(self.ic.data["dec"] - 0.5), max(self.ic.data["dec"] + 0.5))
        region_search_test.generate_patches(
            arcminutes=self.patch_size,
            overlap_percentage=0,
            pixel_scale=0.2,
            dec_range=dec_range,
        )

        region_search_test_ephems = Ephems(
            self.test_ephems,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=[5.0, 50.0],
            earth_loc=EarthLocation.of_site("ctio"),
        )

        # Test mapping return
        found_patches, obj_mapping = region_search_test.search_patches_by_ephems(
            region_search_test_ephems, map_obj_to_patches=True
        )

        self.assertGreater(len(found_patches), 0)
        self.assertIsInstance(obj_mapping, dict)
        self.assertGreater(len(obj_mapping), 0)

        # Check mapping structure
        for obj_name, patch_ids in obj_mapping.items():
            self.assertEqual(obj_name, "TestObject")
            self.assertIsInstance(patch_ids, set)
            self.assertGreater(len(patch_ids), 0)
            # Verify all mapped patches are in the found set
            for pid in patch_ids:
                self.assertIn(pid, found_patches)



    def test_search_patches_within_radius(self):
        """Test search_patches_within_radius filtering."""
        region_search_test = RegionSearch(self.ic, guess_dists=[], earth_loc=self.earth_loc)

        # Generate a basic patch grid
        # Use a known dec range to have predictable patches
        region_search_test.generate_patches(
            arcminutes=self.patch_size,
            overlap_percentage=0,
            pixel_scale=0.2,
            dec_range=(-10, 10),
        )
        # Use a patch from the middle of the ID list to avoid RA=0/360 wrapping issues
        mid_idx = len(region_search_test.patches) // 2
        patch_0 = region_search_test.patches[mid_idx]
        test_ra = patch_0.ra
        test_dec = patch_0.dec

        # Single point ephemeris
        ephem_table = Table({
            "mjd_mid": [self.ic.data["mjd_mid"][0]],
            "ra": [test_ra],
            "dec": [test_dec],
            "Name": ["TestObject"]
        })
        
        region_search_test_ephems = Ephems(
            ephem_table,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=[0.0],
            earth_loc=self.earth_loc,
        )

        # Search with a small radius
        found_patches = region_search_test.search_patches_within_radius(
            region_search_test_ephems, search_radius=0.5
        )
        self.assertIn(mid_idx, found_patches)

        
        # Test that a far away search returns nothing
        # Shift ephemeris way off
        ephem_table["ra"] = [test_ra + 10.0]
        region_search_test_ephems_far = Ephems(
            ephem_table,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=[0.0],
            earth_loc=self.earth_loc,
        )
        found_patches_far = region_search_test.search_patches_within_radius(
            region_search_test_ephems_far, search_radius=0.5
        )
        # Should be empty or at least not contain patch 0
        self.assertNotIn(0, found_patches_far)

    def test_reflex_corrected_search_patches_by_ephems(self):
        """Test using a ephemeris to search patches and find an ImageCollection across multiple reflex-corrected distances."""
        test_dists = [5.0, 39.0]
        region_search_test = RegionSearch(self.ic, guess_dists=test_dists, earth_loc=self.earth_loc)

        # Limit our patch grid to the range of reflex-corrected decs in our ImageCollection
        min_dec = float("inf")
        max_dec = float("-inf")
        for guess_dist in test_dists:
            min_dec = min(
                min_dec,
                min(region_search_test.ic.data[region_search_test.ic.reflex_corrected_col("dec", guess_dist)])
                - 1,
            )
            max_dec = max(
                max_dec,
                max(region_search_test.ic.data[region_search_test.ic.reflex_corrected_col("dec", guess_dist)])
                + 1,
            )

        # Generate a basic patch grid with no overlap
        region_search_test.generate_patches(
            arcminutes=self.patch_size,
            overlap_percentage=0,
            pixel_scale=0.2,
            dec_range=(min_dec, max_dec),
        )

        # Create an ephemeris object to search through
        region_search_test_ephems = Ephems(
            self.test_ephems,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=test_dists,
            earth_loc=EarthLocation.of_site("ctio"),
        )

        # Filter our ImageCollection to times near the ephemeris times
        region_search_test.filter_by_mjds(self.test_ephems["mjd_mid"], time_sep_s=60.0)

        # For each guess distace to test, check that we can find patches and that we can produce
        # valid ImageCollecions from those patches.
        for test_dist in test_dists:
            # Check that for the ic we generate all of our columns correctly for this guess distance
            self.assertIn(
                region_search_test.ic.reflex_corrected_col("ra", test_dist),
                region_search_test.ic.data.columns,
            )
            self.assertIn(
                region_search_test.ic.reflex_corrected_col("dec", test_dist),
                region_search_test.ic.data.columns,
            )
            # Perform the acutal search for the patches
            found_test_patches = region_search_test.search_patches_by_ephems(
                region_search_test_ephems, guess_dist=test_dist
            )

            # Check that we found some patches
            self.assertGreater(len(found_test_patches), 0)
            self.assertGreater(len(region_search_test.ic), 0)

            max_images_filtered = False
            # For each patch we found, check that we can produce a valid ImageCollection
            for patch_id in found_test_patches:
                # Create an ImageCollection with images that overlap from this patch at
                # at the given guess distance
                patch_ic = region_search_test.get_image_collection_from_patch(
                    patch_id, guess_dist=test_dist, min_overlap=0
                )
                self.assertGreater(len(patch_ic), 0)

                # Check the applied the WCS of the ImageCollection
                self.assertEqual(len(set(patch_ic.data["global_wcs_pixel_shape_0"])), 1)
                self.assertEqual(len(set(patch_ic.data["global_wcs_pixel_shape_1"])), 1)
                self.assertEqual(
                    patch_ic.data["global_wcs_pixel_shape_0"][0],
                    patch_arcmin_to_pixels(self.patch_size, 0.2),
                )
                self.assertEqual(
                    patch_ic.data["global_wcs_pixel_shape_1"][0],
                    patch_arcmin_to_pixels(self.patch_size, 0.2),
                )

                # Check that the WCS is valid
                self.assertEqual(len(set(patch_ic.data["global_wcs"])), 1)
                wcs = WCS(patch_ic.data["global_wcs"][0])
                # Check that the WCS is valid
                self.assertIsInstance(wcs, WCS)

                # Assert that each corner of our patch is within the bounds of the WCS
                for ra, dec in region_search_test.get_patch(patch_id).corners:
                    x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs"))
                    pixel_discrep = 2  # Allow 2 pixels of discrepancy for our corners
                    self.assertGreaterEqual(x, 0 - pixel_discrep)
                    self.assertLessEqual(x, patch_arcmin_to_pixels(self.patch_size, 0.2) + pixel_discrep)
                    self.assertGreaterEqual(y, 0 - pixel_discrep)
                    self.assertLessEqual(y, patch_arcmin_to_pixels(self.patch_size, 0.2) + pixel_discrep)

                # Check that each pre-existing column of the original ImageCollection is present
                for col in self.ic.data.columns:
                    self.assertIn(col, patch_ic.data.columns)

                # Check that the patch ImageCollection still has unique visit-detector combinations
                visit_detectors = set(zip(patch_ic.data["visit"], patch_ic.data["detector"]))
                self.assertEqual(len(visit_detectors), len(patch_ic.data))

                # Check that the patch_ic data matches the original ic data for each visit-detector combination.
                cols_changed_by_slicing = set(["std_idx", "ext_idx", "std_name", "config"])
                for patch_idx in range(len(patch_ic)):
                    patch_row = patch_ic[patch_idx]
                    # Get the original row from the ic data for the same visit and detector
                    orig_ic_row = self.ic.data[
                        (self.ic.data["visit"] == patch_row["visit"])
                        & (self.ic.data["detector"] == patch_row["detector"])
                    ][0]
                    # Check that the original row matches the patch row for all columns
                    # except standardizer-related columns changed by slicing.
                    for col in self.ic.data.columns:
                        if col in cols_changed_by_slicing:
                            continue
                        self.assertEqual(orig_ic_row[col], patch_row[col])
                    # Assert that the image has a non-zero overlap with the patch
                    self.assertGreater(patch_row["overlap_deg"], 0)

                if len(patch_ic) > 3:
                    max_images_filtered = True
                    # Build a smaller ImageCollection from the patch with the images that have the highest overlap
                    small_ic = region_search_test.get_image_collection_from_patch(
                        patch_id, guess_dist=test_dist, min_overlap=0, max_images=3
                    )
                    # We had to filter down to the images through sorting by overlap. So check that the ImageCollection
                    # is now sorted by overlap.
                    self.assertTrue(
                        np.all(np.diff(small_ic.data["overlap_deg"]) <= 0), f"{small_ic.data['overlap_deg']}"
                    )

                    # Check that we included the image with the highest degree of overlap when cutting down the images
                    # We know this is the first image since we checked for sorting above.
                    self.assertEqual(small_ic.data["overlap_deg"][0], max(patch_ic.data["overlap_deg"]))
            self.assertTrue(
                max_images_filtered, "No patches had at least 3 images to test max image filtering."
            )

    def test_patch_overlap(self):
        """
        Basic tests of patch initialization and overlap overlap
        """
        # Create two patches we know should overlap.
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
        overlap = patch1.measure_overlap(patch2.polygon)
        self.assertGreater(overlap, 0)
        self.assertTrue(patch1.overlaps_polygon(patch2.polygon))

        # Test with a patch that we know should not overlap.
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
        overlap = patch1.measure_overlap(patch3.polygon)
        self.assertEqual(overlap, 0.0)
        self.assertFalse(patch1.overlaps_polygon(patch3.polygon))
