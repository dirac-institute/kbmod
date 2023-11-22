import os
import shutil
import tempfile
import unittest

import numpy as np
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

        self.assertEqual(ic.meta["n_entries"], 3)
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
        self.assertEqual(len(ic.wcs), 3)
        self.assertEqual(len(ic.bbox), 3)

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
        self.assertIsInstance(ic["mjd", "location"], atbl.Table)

        # This is kind of a thing of the standardizers themselves, but to
        # ensure the standardization results are becoming columns we test for
        # content, knowing KBMODV1 is the standardizer in question.
        # Test internal book-keeping columns are not returned
        expected_cols = ["mjd", "filter", "visit_id", "observat", "obs_lat",
                         "obs_lon", "obs_elev", "location", "ra", "dec"]
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
        with self.assertRaisesRegex(FileNotFoundError, ":memory:"):
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
        subset = ("mjd", "filter", "obs_lon")
        self.assertTrue((ic[subset] == ic2[subset]).all())

        fname = os.path.join(tmpdir, "reachable.ecsv")
        ic2.write(fname)
        ic3 = ImageCollection.read(fname)

        self.assertEqual(ic2, ic3)

        # cleanup resources
        shutil.rmtree(tmpdir)



if __name__ == "__main__":
    unittest.main()
