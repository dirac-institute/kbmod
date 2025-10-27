import logging
import os
import shutil
import tempfile
import unittest

import astropy.table as atbl
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

import numpy as np

from kbmod import ImageCollection, Standardizer
from utils import DECamImdiffFactory


class TestImageCollection(unittest.TestCase):
    """Test ImageCollection class."""

    fitsFactory = DECamImdiffFactory()

    def setUp(self):
        self.fits = self.fitsFactory.get_n(3)

    def test_basics(self):
        """Test image collection indexing, equality and special attributes."""
        # Test basics work
        ic = ImageCollection.fromTargets(self.fits)
        ic2 = ImageCollection.fromTargets(self.fits)
        ic3 = ImageCollection.fromTargets(self.fitsFactory.get_n(5))

        self.assertEqual(ic.meta["n_stds"], 3)
        self.assertEqual(ic, ic2)
        self.assertNotEqual(ic, ic3)

        # Test wcs, bbox and lazy loaded standardizers
        std, ext = ic.get_standardizer(0).values()
        self.assertIsInstance(std, Standardizer)
        self.assertEqual(ext, 0)

        stds = ic.get_standardizers([0, 1])
        for entry in stds:
            std, ext = entry.values()
            self.assertIsInstance(std, Standardizer)
        self.assertEqual(len(stds), 2)

        # Make sure the arrays are not empty. In case there are no wcs or
        # bboxes - we're still expecting arrays of None
        self.assertEqual(len(list(ic.wcs)), 3)
        self.assertEqual(len(list(ic.bbox)), 3)

        # Test indexing works as expected
        # * int -> row
        # * lists or slice -> new ImageCollection
        # * string -> column
        row = ic[0]
        self.assertIsInstance(row, atbl.Row)

        self.assertIsInstance(ic[[0, 1, 2]], ImageCollection)
        self.assertTrue((ic[[0, 1, 2]] == ic).all())

        self.assertIsInstance(ic["location"], atbl.Column)
        self.assertEqual(len(ic["location"]), 3)
        self.assertIsInstance(ic["mjd_mid", "location"], atbl.Table)

        # This is kind of a thing of the standardizers themselves, but to
        # ensure the standardization results are becoming columns we test for
        # content, knowing KBMODV1 is the standardizer in question.
        # Test internal book-keeping columns are not returned
        expected_cols = [
            "mjd_mid",
            "obs_lon",
            "obs_lat",
            "obs_elev",
            "FILTER",
            "IDNUM",
            "visit",
            "OBSID",
            "DTNSANAM",
            "AIRMASS",
            "DIMM2SEE",
            "GAINA",
            "GAINB",
            "location",
            "ra",
            "dec",
            "ra_tl",
            "dec_tl",
            "ra_tr",
            "dec_tr",
            "ra_bl",
            "dec_bl",
            "ra_br",
            "dec_br",
            "wcs",
        ]
        self.assertEqual(list(ic.columns.keys()), expected_cols)
        self.assertEqual(list(row.keys()), expected_cols)

    def test_missing_metadata(self):
        """Test ImageCollection raises error when required metadata is missing."""
        # Generate a set of 5 test targets to standardize
        n_targets = 5
        fits = self.fitsFactory.get_n(n_targets)

        # Remove a required metadata keyword from one of the targets.
        missing_header = "DATE-AVG"
        del fits[1]["PRIMARY"].header[missing_header]

        # Test that an exception is raised when fail_on_error is True, and that
        # the exception includes the missing header keyword.
        with self.assertRaisesRegex(Exception, missing_header):
            ImageCollection.fromTargets(fits, fail_on_error=True)

        # Test that the ImageCollection is still created when fail_on_error is
        # False, skipping the one failed target with missing metadata.
        logging.disable(logging.WARNING)
        ic = ImageCollection.fromTargets(fits, fail_on_error=False)
        self.assertEqual(len(ic), n_targets - 1)
        self.assertEqual(ic.meta["n_stds"], n_targets - 1)
        self.assertEqual(len(ic._standardizers), n_targets - 1)

    def test_write_read_unreachable(self):
        """Test ImageCollection can write itself to disk, and read the written
        table without raising errors when original data is unreachable.
        """
        ic = ImageCollection.fromTargets(self.fitsFactory.get_n(3))

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "test.ecsv")
            ic.write(fname)
            ic2 = ImageCollection.read(fname)

        self.assertEqual(ic, ic2)
        with self.assertRaisesRegex(FileNotFoundError, "location is not a file, but no hdulist"):
            ic2.get_standardizer(0)

    def test_write_read_reachable(self):
        """Test ImageCollection can write itself to disk, and fully reconstruct
        itself from disk again.
        """
        hduls = self.fitsFactory.get_n(3, spoof_data=True)
        ic = ImageCollection.fromTargets(hduls)

        tmpdir = tempfile.mkdtemp()
        for i, hdul in enumerate(hduls):
            fname = os.path.join(tmpdir, f"{i:0>3}.fits")
            hdul.writeto(fname)
            hdul.close()
        ic2 = ImageCollection.fromDir(tmpdir)

        # We can't compare location, ra, dec spoofing data changes the
        # image data which updates header metadata related to these values
        subset = ("mjd_mid", "FILTER", "obs_lon")
        self.assertTrue((ic[subset] == ic2[subset]).all())

        fname = os.path.join(tmpdir, "reachable.ecsv")
        ic2.write(fname)
        ic3 = ImageCollection.read(fname)

        self.assertEqual(ic2, ic3)

        # cleanup resources
        shutil.rmtree(tmpdir)

    def test_bintablehdu(self):
        ic2 = ImageCollection.fromTargets(self.fits)

        tbl = ic2.toBinTableHDU()
        test = ImageCollection.fromBinTableHDU(tbl)
        self.assertEqual(ic2, test)

        ic2.pack()
        tbl = ic2.toBinTableHDU()
        test = ImageCollection.fromBinTableHDU(tbl)
        self.assertEqual(ic2, test)

    def test_workunit(self):
        """Tests imagecollection exports a work unit without error."""
        # not too sure how to validate, so just call to make sure
        # no errors are thrown. Of course WU throws an error because
        # no config was given even though its an optional.
        from kbmod.configuration import SearchConfiguration

        # Disable the warnings because the fitFactory will generate empty layers.
        logging.disable(logging.CRITICAL)

        data = self.fitsFactory.get_n(3, spoof_data=True)
        ic = ImageCollection.fromTargets(data)
        wu = ic.toWorkUnit(search_config=SearchConfiguration())
        self.assertEqual(len(wu), 3)

        # Re-enable the warnings.
        logging.disable(logging.NOTSET)

        # We can retrieve the meta data from the WorkUnit, including the renamed
        # "data_loc" column.
        filter_info = wu.get_constituent_meta("visit")
        self.assertEqual(len(filter_info), 3)
        self.assertIsNotNone(filter_info[0])

        data_loc = wu.get_constituent_meta("data_loc")
        self.assertEqual(len(data_loc), 3)
        self.assertEqual(data_loc[0], ":memory:")

        # We can write the whole work unit to a file.
        with tempfile.TemporaryDirectory() as dir_name:
            wu.to_fits(f"{dir_name}/test.fits")

    def test_packing(self):
        """Test packing behaves as expected."""
        ic = ImageCollection.fromTargets(self.fits)
        ic2 = ImageCollection.fromTargets(self.fits)

        old_meta = ic2.meta
        self.assertEqual(ic, ic2)

        ic2.pack()
        self.assertNotEqual(ic, ic2)
        self.assertTrue("shared_cols" in ic2.meta)
        for k in ic2.meta["shared_cols"]:
            with self.subTest("Shared key not found in meta: ", key=k):
                self.assertTrue(k in ic2.meta)

        ic2.unpack()
        self.assertEqual(old_meta, ic2.meta)
        self.assertEqual(ic, ic2)

    def test_indexing(self):
        """Tests indexing behaves as expected."""
        # mostly what we can hope for here is none of the operations
        # raise an error.
        fits = self.fitsFactory.get_n(5)
        ic = ImageCollection.fromTargets(fits)

        from astropy.table import Row, Column, Table

        self.assertIsInstance(ic[0], Row)
        self.assertIsInstance(ic["mjd_mid"], Column)
        self.assertIsInstance(ic["mjd_mid", "FILTER"], Table)
        self.assertIsInstance(ic[["mjd_mid", "FILTER"]], Table)
        with self.assertRaisesRegex(ValueError, "Illegal type"):
            ic[0, 1]
        self.assertIsInstance(ic[[0, 1]], ImageCollection)
        self.assertIsInstance(ic[:3], ImageCollection)
        self.assertIsInstance(ic[1:3], ImageCollection)

        old_stds = list(ic._standardizers.copy())
        subset = ic[[0, 2, 3, 4]]
        self.assertListEqual(list(subset._standardizers), old_stds)
        self.assertListEqual(list(subset.data["std_idx"]), [0, 2, 3, 4])

        subset.reset_lazy_loading_indices()
        self.assertListEqual(list(subset.data["std_idx"]), [0, 1, 2, 3])
        self.assertEqual(len(subset._standardizers), 4)
        self.assertListEqual(list(subset._standardizers), list(ic._standardizers[[0, 2, 3, 4]]))

        ic.data["std_idx"] = [0, 0, 1, 1, 2]
        ic.data["ext_idx"] = [0, 1, 0, 1, 0]
        ic._standardizers = ["first", "second", "third"]

        subset = ic[[0, 2, 3, 4]]
        self.assertListEqual(list(subset.data["std_idx"]), [0, 1, 1, 2])
        self.assertListEqual(list(subset.data["ext_idx"]), [0, 0, 1, 0])

        subset.reset_lazy_loading_indices()
        self.assertListEqual(list(subset._standardizers), ["first", "second", "third"])
        self.assertListEqual(list(subset.data["std_idx"]), [0, 1, 1, 2])
        self.assertListEqual(list(subset.data["ext_idx"]), [0, 0, 1, 0])

    def test_reflex_correct(self):
        """Test the reflex_correct method in ImageCollection generates columns reflex_corrected RA and Dec columns."""
        # Create a mock ImageCollection
        fits = self.fitsFactory.get_n(3, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)

        # Mock EarthLocation and guess distance
        earth_loc = EarthLocation.of_site("ctio")
        guess_distance = 50.0  # in AU

        # Apply reflex correction
        ic.reflex_correct(guess_distance, earth_loc)

        # Retrieve the reflex-corrected coordinates
        corrected_ra = ic.data[ic.reflex_corrected_col("ra", guess_distance)]
        corrected_dec = ic.data[ic.reflex_corrected_col("dec", guess_distance)]
        corrected_coords = SkyCoord(ra=corrected_ra * u.deg, dec=corrected_dec * u.deg, frame="icrs")

        # Verify that the original RA/Dec differ from the corrected ones
        original_coords = SkyCoord(ra=ic.data["ra"] * u.deg, dec=ic.data["dec"] * u.deg, frame="icrs")
        original_separations = original_coords.separation(corrected_coords).arcsecond
        self.assertTrue(np.all(original_separations > 0), "Original and corrected coordinates should differ.")

    def test_reflex_correct_col(self):
        """Tests the helper function for generating reflex-corrected column names"""
        # Create a mock ImageCollection
        fits = self.fitsFactory.get_n(3, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)

        # Assert that invalid column names raise an error
        with self.assertRaises(ValueError):
            ic.reflex_corrected_col("invalid_column", 50.0)

        # Assert that invalid guess_distance raises an error
        with self.assertRaises(ValueError):
            ic.reflex_corrected_col("ra", None)
        with self.assertRaises(ValueError):
            ic.reflex_corrected_col("ra", 5)  # guess_distance must be floats

        # Test reflex_corrected_col function
        guess_distance = 50.0
        self.assertEqual(ic.reflex_corrected_col("ra", guess_distance), "ra_50.0")
        self.assertEqual(ic.reflex_corrected_col("dec", guess_distance), "dec_50.0")

    def test_filter_by_mjds(self):
        """Test filtering to images near a set of mjds."""
        # Create a mock ImageCollection
        init_len = 10
        fits = self.fitsFactory.get_n(init_len, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)
        # Spoof some mjds
        ic.data["mjd_mid"] = range(52864, 52884, int((52884 - 52864) / init_len))

        # Test we raise a value error for a negative separation
        with self.assertRaises(ValueError):
            ic.filter_by_mjds([52864], time_sep_s=-1)

        # Filter by our same mjds
        mjds_to_filter = [mjd for mjd in ic.data["mjd_mid"]]
        ic.filter_by_mjds(mjds_to_filter)
        self.assertEqual(len(ic), init_len)

        # Now filter all of our MJDs against a set off by an amount less than the
        # default tolerance
        mjds_to_filter = [mjd - 0.0001 / (60 * 60 * 24) for mjd in ic.data["mjd_mid"]]
        ic.filter_by_mjds(mjds_to_filter[:8])
        # Since we are still within our tolerance no images were removed
        # except for the final 2
        self.assertEqual(len(ic), init_len - 2)

        # Now filter all of our MJDs with a tolerance of 30 seconds.
        mjds_to_filter = [mjd - 29 / (60 * 60 * 24) for mjd in ic.data["mjd_mid"]]
        ic.filter_by_mjds(mjds_to_filter[:5], time_sep_s=30)
        # We are still within our tolerance for the five times we provide,
        # so we should still have 5 images left
        self.assertEqual(len(ic), init_len - 5)

        # Now filter again but ask for an exact match, filtering out all results.
        ic.filter_by_mjds(mjds_to_filter, time_sep_s=0)
        self.assertEqual(len(ic), 0)

    def test_filter_by_time_range(self):
        """Test filtering an ImageCollection to a min and max time range."""
        # Create a mock ImageCollection
        init_len = 10
        fits = self.fitsFactory.get_n(init_len, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)
        # Spoof some mjds
        mjds = np.arange(52864, 52884, int((52884 - 52864) / init_len))
        # Randomly sort to further verify filtering
        np.random.shuffle(mjds)
        ic.data["mjd_mid"] = mjds

        # Test we raise a value error if both start and end are none
        with self.assertRaises(ValueError):
            ic.filter_by_time_range(None, None)

        # Test that we raise a value error if start is greater than end
        with self.assertRaises(ValueError):
            ic.filter_by_time_range(max(mjds), min(mjds))

        # Since we filter by the ImageCollection's starting and end bounds,
        # each time confirming that no images were removed.
        ic.filter_by_time_range(start_mjd=min(mjds))
        self.assertEqual(len(ic), init_len)
        ic.filter_by_time_range(end_mjd=max(mjds))
        self.assertEqual(len(ic), init_len)
        ic.filter_by_time_range(start_mjd=min(mjds), end_mjd=max(mjds))
        self.assertEqual(len(ic), init_len)

        # Filter off our first time
        ic.filter_by_time_range(start_mjd=min(mjds) + 1)
        self.assertEqual(len(ic), init_len - 1)

        # Filter off our last time
        ic.filter_by_time_range(end_mjd=max(mjds) - 1)
        self.assertEqual(len(ic), init_len - 2)

        # Filter off two more times based on the new min and max mjds in the ic
        min_time = min(ic["mjd_mid"]) + 1
        max_time = max(ic["mjd_mid"]) - 1
        ic.filter_by_time_range(start_mjd=min_time, end_mjd=max_time)
        self.assertEqual(len(ic), init_len - 4)

    def test_modifying_columns(self):
        """Test adding and removing columns."""
        # Create a mock ImageCollection
        init_len = 10
        fits = self.fitsFactory.get_n(init_len, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)
        ic["testcol"] = [
            1,
        ] * len(ic)
        self.assertTrue("testcol" in ic._userColumns)
        self.assertTrue("testcol" in ic.data.columns)

        ic.remove_column("testcol")
        self.assertFalse("testcol" in ic._userColumns)
        self.assertFalse("testcol" in ic.data.columns)

        ic["testcol"] = [
            1,
        ] * len(ic)
        with self.assertWarns(Warning):
            ic.remove_columns(["testcol", "config"])
        self.assertFalse("testcol" in ic._userColumns)
        self.assertFalse("testcol" in ic.data.columns)
        self.assertTrue("config" in ic.data.columns)

        with self.assertRaises(KeyError):
            ic.remove_column("testcol")

    def test_drop_bands(self):
        """Test filtering an ImageCollection to drop specified bands."""
        # Create a mock ImageCollection
        fits = self.fitsFactory.get_n(10, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)

        # Spoof some gilters
        filters = ["u", "g", "r", "i", "z", "y"]
        ic.data["band"] = [filters[i % len(filters)] for i in range(len(ic))]

        # Drop 'u' and 'y' bands
        ic.drop_bands(["u", "y"])
        self.assertTrue(all(filt not in ic.data["band"] for filt in ["u", "y"]))
        self.assertEqual(len(ic), 7)  # Dropped two u band and 1 y band images

        # Drop 'g' band
        ic.drop_bands(["g"])
        self.assertTrue(all(filt != "g" for filt in ic.data["band"]))
        self.assertEqual(len(ic), 5)  # Dropped 2 g band images

    def test_filter_by_wcs_error(self):
        """Test filtering an ImageCollection by WCS error."""
        # Create a mock ImageCollection
        fits = self.fitsFactory.get_n(10, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)

        # Spoof some WCS error values in degrees
        wcs_errors = [0.1 * i for i in range(1, len(ic) + 1)]
        ic.data["wcs_error"] = wcs_errors

        # Filter by max WCS error of 0.5 degrees
        ic.filter_by_wcs_error(0.5, in_arcsec=False)
        self.assertTrue(all(err <= 0.5 for err in ic.data["wcs_error"]))
        self.assertEqual(len(ic), 5)

        # Filter again but in arcseconds
        ic.filter_by_wcs_error(0.31 * 3600, in_arcsec=True)
        self.assertTrue(all(err <= 0.31 for err in ic.data["wcs_error"]))
        self.assertEqual(len(ic), 3)

    def test_obs_nights_spanned(self):
        """Test calculating the number of observation nights spanned by the ImageCollection."""
        # Create a mock ImageCollection
        fits = self.fitsFactory.get_n(10, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)

        # Populate mjd_mid with consecutive nights (2023-01-01 .. 2023-01-10)
        dates = [f"2023-01-{str(i).zfill(2)}" for i in range(1, 11)]
        ic.data["mjd_mid"] = Time(dates).mjd
        self.assertEqual(ic.obs_nights_spanned(), 10)

        # Spoof some nights with repeats, out of order
        # Note that 2024 has a leap day
        repeated_dates = 5 * ["2024-03-03"] + 2 * ["2024-02-27"] + 3 * ["2024-03-02"]
        ic.data["mjd_mid"] = Time(repeated_dates).mjd
        self.assertEqual(ic.obs_nights_spanned(), 6)


if __name__ == "__main__":
    unittest.main()
