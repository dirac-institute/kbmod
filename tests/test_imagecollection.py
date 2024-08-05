import os
import shutil
import tempfile
import unittest

import astropy.table as atbl

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

        data = self.fitsFactory.get_n(3, spoof_data=True)
        ic = ImageCollection.fromTargets(data)
        wu = ic.toWorkUnit(search_config=SearchConfiguration())
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


if __name__ == "__main__":
    unittest.main()
