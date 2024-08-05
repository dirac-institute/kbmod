import unittest

import numpy as np
from astropy.table import Table, vstack

import kbmod.mocking as kbmock


class TestEmptyFits(unittest.TestCase):
    def test(self):
        """Test basics of EmptyFits factory."""
        factory = kbmock.EmptyFits()
        hduls = factory.mock(2)

        hdul = hduls[0]
        zeros = np.zeros((100, 100))
        self.assertEqual(len(hduls), 2)
        self.assertEqual(len(hduls[0]), 4)
        for name, hdu in zip(("PRIMARY", "IMAGE", "VARIANCE", "MASK"), hduls[0]):
            self.assertEqual(name, hdu.name)
        self.assertEqual(hdul["PRIMARY"].data, None)
        self.assertTrue((hdul["IMAGE"].data == zeros).all())
        self.assertTrue((hdul["VARIANCE"].data == zeros).all())
        self.assertTrue((hdul["MASK"].data == zeros).all())

        factory = kbmock.EmptyFits(shape=(10, 100), step_mjd=1)
        hduls = factory.mock(2)
        hdul = hduls[0]
        self.assertEqual(hdul["IMAGE"].data.shape, (10, 100))
        self.assertEqual(hdul["VARIANCE"].data.shape, (10, 100))
        self.assertEqual(hdul["MASK"].data.shape, (10, 100))
        dt = hduls[1]["PRIMARY"].header["OBS-MJD"] - hduls[0]["PRIMARY"].header["OBS-MJD"]
        self.assertEqual(dt, 1)

        with self.assertRaisesRegex(ValueError, "destination is read-only"):
            hdul["IMAGE"].data[0, 0] = 0

        factory = kbmock.EmptyFits(editable_images=True)
        hduls = factory.mock(2)
        hdul = hduls[0]
        hdul["IMAGE"].data[0, 0] = 1
        hdul["VARIANCE"].data[0, 0] = 2
        self.assertEqual(hdul["IMAGE"].data[0, 0], 1)
        self.assertEqual(hduls[1]["IMAGE"].data[0, 0], 0)
        with self.assertRaisesRegex(ValueError, "destination is read-only"):
            hdul["MASK"].data[0, 0] = 0

        factory = kbmock.EmptyFits(editable_images=True, editable_masks=True)
        hduls = factory.mock(2)
        hdul = hduls[0]
        hdul["MASK"].data[0, 0] = 1
        self.assertEqual(hduls[0]["MASK"].data[0, 0], 1)
        self.assertEqual(hduls[1]["MASK"].data[0, 0], 0)


class TestSimpleFits(unittest.TestCase):
    def test(self):
        """Test basic functionality of SimpleFits factory."""
        factory = kbmock.SimpleFits()
        hduls = factory.mock(2)

        hdul = hduls[0]
        zeros = np.zeros((100, 100))
        self.assertEqual(len(hduls), 2)
        self.assertEqual(len(hduls[0]), 4)
        for name, hdu in zip(("PRIMARY", "IMAGE", "VARIANCE", "MASK"), hduls[0]):
            self.assertEqual(name, hdu.name)
        self.assertEqual(hdul["PRIMARY"].data, None)
        self.assertTrue((hdul["IMAGE"].data == zeros).all())
        self.assertTrue((hdul["VARIANCE"].data == zeros).all())
        self.assertTrue((hdul["MASK"].data == zeros).all())

        factory = kbmock.SimpleFits(shape=(10, 100), step_mjd=1)
        hduls = factory.mock(2)
        hdul = hduls[0]
        self.assertEqual(hdul["IMAGE"].data.shape, (10, 100))
        self.assertEqual(hdul["VARIANCE"].data.shape, (10, 100))
        self.assertEqual(hdul["MASK"].data.shape, (10, 100))
        step_mjd = hduls[1]["PRIMARY"].header["OBS-MJD"] - hduls[0]["PRIMARY"].header["OBS-MJD"]
        self.assertEqual(step_mjd, 1)

    def test_static_src_cat(self):
        src_cat = kbmock.SourceCatalog.from_defaults()
        src_cat2 = kbmock.SourceCatalog.from_defaults()
        self.assertEqual(src_cat.config.mode, "static")
        self.assertFalse((src_cat.table == src_cat2.table).all())

        src_cat = kbmock.SourceCatalog.from_defaults(n=3)
        self.assertEqual(len(src_cat.table), 3)

        shape = (300, 500)
        param_ranges = {
            "amplitude": [100, 100],
            "x_mean": (100, 200),
            "y_mean": (50, 80),
            "x_stddev": [2., 2.],
            "y_stddev": [2., 2.]
        }
        src_cat = kbmock.SourceCatalog.from_defaults(param_ranges, seed=100)
        src_cat2 = kbmock.SourceCatalog.from_defaults(param_ranges, seed=100)
        self.assertTrue((src_cat.table == src_cat2.table).all())
        self.assertLessEqual(src_cat.table["x_mean"].max(), shape[1])
        self.assertLessEqual(src_cat.table["y_mean"].max(), shape[0])

        factory = kbmock.SimpleFits(shape=shape, src_cat=src_cat)
        hdul = factory.mock()[0]

        for x, y in src_cat.table["x_mean", "y_mean"]:
            # Can only test greater or equal because objects may overlap
            self.assertGreaterEqual(hdul["IMAGE"].data[int(y), int(x)], 80)

    def test_progressive_obj_cat(self):
        obj_cat = kbmock.ObjectCatalog.from_defaults()
        obj_cat2 = kbmock.ObjectCatalog.from_defaults()
        self.assertEqual(obj_cat.config.mode, "progressive")
        self.assertFalse((obj_cat.table == obj_cat2.table).all())

        obj_cat = kbmock.ObjectCatalog.from_defaults(n=3)
        self.assertEqual(len(obj_cat.table), 3)

        shape = (300, 500)
        param_ranges = {
            "amplitude": [100, 100],
            "x_mean": (0, 50),
            "y_mean": (50, shape[0]),
            "x_stddev": [2., 2.],
            "y_stddev": [2., 2.],
            "vx": [100, 300],
            "vy": [0, 0]
        }
        obj_cat = kbmock.ObjectCatalog.from_defaults(param_ranges, seed=100)
        obj_cat2 = kbmock.ObjectCatalog.from_defaults(param_ranges, seed=100)
        self.assertTrue((obj_cat.table == obj_cat2.table).all())
        self.assertLessEqual(obj_cat.table["x_mean"].max(), 50)
        self.assertLessEqual(obj_cat.table["y_mean"].max(), shape[0])

        step_mjd = 0.1
        obj_cat = kbmock.ObjectCatalog.from_defaults(param_ranges, n=5, seed=100)
        factory = kbmock.SimpleFits(shape=shape, step_mjd=step_mjd, obj_cat=obj_cat)
        hduls = factory.mock(n=5)

        obj_cat.reset()
        for i in range(5):
            newcat = obj_cat.mock(dt=step_mjd)[0]
            for x, y in newcat["x_mean", "y_mean"]:
                if x < shape[1] and y < shape[0]:
                    # Can only test greater or equal because objects may overlap
                    # probably a rounding-off error while moving drops 1 flux count
                    self.assertGreaterEqual(hduls[i]["IMAGE"].data[int(y), int(x)], 79)
                    self.assertGreaterEqual(hduls[i]["VARIANCE"].data[int(y), int(x)], 0)

    def test_folding_obj_cat(self):
        nobj = 5
        shape = (300, 300)
        timestamps = np.arange(58915, 58920, 1)

        start_x = np.ones((nobj, ))*10
        start_y = np.linspace(10, shape[0]-10, nobj)

        cats = []
        for i, t in enumerate(timestamps):
            cats.append(Table({
                "amplitude": [100]*nobj,
                "obstime": [t]*nobj,
                "x_mean": start_x + 15*i*i,
                "y_mean": start_y,
                "stddev": [2.0]*nobj
            }))
        catalog = vstack(cats)

        obj_cat = kbmock.ObjectCatalog.from_table(catalog)
        obj_cat.mode = "folding"

        prim_hdr_factory = kbmock.HeaderFactory.from_primary_template(
            mutables=["OBS-MJD"],
            callbacks=[kbmock.ObstimeIterator(timestamps), ],
        )

        factory = kbmock.SimpleFits(shape=shape, obj_cat=obj_cat)
        factory.prim_hdr = prim_hdr_factory
        hduls = factory.mock(n=len(timestamps))

        obj_cat.reset()
        cats = obj_cat.mock(t=timestamps)
        for hdul, cat in zip(hduls, cats):
            for x, y in cat["x_mean", "y_mean"]:
                if x < shape[1] and y < shape[0]:
                    # Can only test greater or equal because objects may overlap
                    # probably a rounding-off error while moving drops 1 flux count
                    self.assertGreaterEqual(hdul["IMAGE"].data[int(y), int(x)], 79)
                    self.assertGreaterEqual(hdul["VARIANCE"].data[int(y), int(x)], 0)

    # TODO: move to pytest and mark as xfail
    def test_noise_gen(self):
        factory = kbmock.SimpleFits(shape=(1000, 1000), with_noise=True)
        hdul = factory.mock()[0]
        self.assertAlmostEqual(hdul["IMAGE"].data.mean(), 10, 1)
        self.assertAlmostEqual(hdul["IMAGE"].data.std(), 1, 1)

        factory = kbmock.SimpleFits(shape=(1000, 1000), with_noise=True, noise="realistic")
        hdul = factory.mock()[0]
        print(hdul["IMAGE"].data.mean())
        self.assertAlmostEqual(hdul["IMAGE"].data.mean(), 32, 1)
        self.assertAlmostEqual(hdul["IMAGE"].data.std(), 7.5, 1)

        img_factory = kbmock.SimpleImage(shape=(1000, 1000), add_noise=True, noise=5, noise_std=2.0)
        factory = kbmock.SimpleFits()
        factory.img_data = img_factory
        hduls = factory.mock(n=3)

        for hdul in hduls[1:]:
            self.assertFalse((hduls[0]["IMAGE"].data == hdul["IMAGE"].data).all())
            self.assertAlmostEqual(hdul["IMAGE"].data.mean(), 5, 1)
            self.assertAlmostEqual(hdul["IMAGE"].data.std(), 2, 1)


if __name__ == "__main__":
    unittest.main()







