import unittest

import numpy as np

from astropy.wcs import WCS
from astropy.time import Time
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

        factory = kbmock.EmptyFits(shape=(10, 100), step_t=1)
        hduls = factory.mock(2)
        hdul = hduls[0]
        self.assertEqual(hdul["IMAGE"].data.shape, (10, 100))
        self.assertEqual(hdul["VARIANCE"].data.shape, (10, 100))
        self.assertEqual(hdul["MASK"].data.shape, (10, 100))
        dt = Time(hduls[1]["PRIMARY"].header["DATE-OBS"]) - Time(hduls[0]["PRIMARY"].header["DATE-OBS"])
        self.assertEqual(dt.to("day").value, 1)

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
    def setUp(self):
        self.n_obj = 5
        self.n_imgs = 3
        self.shape = (100, 300)
        self.padded = ((10, 90), (10, 290))
        self.timestamps = Time(np.arange(58915, 58915 + self.n_imgs, 1), format="mjd")
        self.step_t = 1

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

        factory = kbmock.SimpleFits(shape=(10, 100), step_t=1)
        hduls = factory.mock(2)
        hdul = hduls[0]
        self.assertEqual(hdul["IMAGE"].data.shape, (10, 100))
        self.assertEqual(hdul["VARIANCE"].data.shape, (10, 100))
        self.assertEqual(hdul["MASK"].data.shape, (10, 100))
        step_t = Time(hduls[1]["PRIMARY"].header["DATE-OBS"]) - Time(hduls[0]["PRIMARY"].header["DATE-OBS"])
        self.assertEqual(step_t.to("day").value, 1.0)

    def test_static_src_cat(self):
        """Test that static source catalog works and is correctly rendered."""
        src_cat = kbmock.SourceCatalog.from_defaults()
        src_cat2 = kbmock.SourceCatalog.from_defaults()
        self.assertEqual(src_cat.config["mode"], "static")
        self.assertFalse((src_cat.table == src_cat2.table).all())

        src_cat = kbmock.SourceCatalog.from_defaults(n=self.n_obj)
        self.assertEqual(len(src_cat.table), self.n_obj)

        param_ranges = {
            "amplitude": [100, 100],
            "x_mean": self.padded[1],
            "y_mean": self.padded[0],
            "x_stddev": [2.0, 2.0],
            "y_stddev": [2.0, 2.0],
        }
        src_cat = kbmock.SourceCatalog.from_defaults(param_ranges, seed=100)
        src_cat2 = kbmock.SourceCatalog.from_defaults(param_ranges, seed=100)
        self.assertTrue((src_cat.table == src_cat2.table).all())
        self.assertLessEqual(src_cat.table["x_mean"].max(), self.shape[1])
        self.assertLessEqual(src_cat.table["y_mean"].max(), self.shape[0])

        factory = kbmock.SimpleFits(shape=self.shape, src_cat=src_cat)
        hdul = factory.mock()[0]

        x = np.round(src_cat.table["x_mean"].data).astype(int)
        y = np.round(src_cat.table["y_mean"].data).astype(int)
        self.assertGreaterEqual(hdul["IMAGE"].data[y, x].min(), 90)

    def validate_cat_render(self, hduls, cats, expected_gte=90):
        """Validate that catalog objects appear in the given images.

        Parameters
        ----------
        hduls : `list[astropy.io.fits.HDUList]`
            List of FITS files to check.
        cats : `list[astropy.table.Table]`
            List of catalog realizations containing the coordinate of objects
            to check for.
        expected_gte : `float`
            Expected minimal value of the pixel at the object's location.
        """
        for hdul, cat in zip(hduls, cats):
            x = np.round(cat["x_mean"].data).astype(int)
            y = np.round(cat["y_mean"].data).astype(int)
            self.assertGreaterEqual(hdul["IMAGE"].data[y, x].min(), expected_gte)
            self.assertGreaterEqual(hdul["VARIANCE"].data[y, x].min(), expected_gte)

    def test_progressive_obj_cat(self):
        """Test progressive catalog renders properly."""
        obj_cat = kbmock.ObjectCatalog.from_defaults()
        obj_cat2 = kbmock.ObjectCatalog.from_defaults()
        self.assertEqual(obj_cat.config["mode"], "progressive")
        self.assertFalse((obj_cat.table == obj_cat2.table).all())

        obj_cat = kbmock.ObjectCatalog.from_defaults(n=self.n_obj)
        self.assertEqual(len(obj_cat.table), self.n_obj)

        param_ranges = {
            "amplitude": [100, 100],
            "x_mean": (0, 90),
            "y_mean": self.padded[0],
            "x_stddev": [2.0, 2.0],
            "y_stddev": [2.0, 2.0],
            "vx": [10, 20],
            "vy": [0, 0],
        }
        seed = 200
        obj_cat = kbmock.ObjectCatalog.from_defaults(param_ranges, seed=seed)
        obj_cat2 = kbmock.ObjectCatalog.from_defaults(param_ranges, seed=seed)
        self.assertTrue((obj_cat.table == obj_cat2.table).all())

        obj_cat = kbmock.ObjectCatalog.from_defaults(param_ranges, n=self.n_obj)
        factory = kbmock.SimpleFits(shape=self.shape, step_t=self.step_t, obj_cat=obj_cat)
        hduls = factory.mock(n=self.n_imgs)

        obj_cat.reset()
        cats = obj_cat.mock(n=self.n_imgs, dt=self.step_t)
        self.validate_cat_render(hduls, cats)

    def test_folding_obj_cat(self):
        """Test folding catalog renders properly."""
        # Set up shared values for the whole setup
        # like starting positions of object and timestamps
        start_x = np.ones((self.n_obj,)) * 10
        start_y = np.linspace(10, self.shape[0] - 10, self.n_obj)

        # Set up non-linear catalog (objects will move as counter^2*v)
        cats = []
        for i, t in enumerate(self.timestamps):
            cats.append(
                Table(
                    {
                        "amplitude": [100] * self.n_obj,
                        "obstime": [t] * self.n_obj,
                        "x_mean": start_x + 15 * i * i,
                        "y_mean": start_y,
                        "stddev": [2.0] * self.n_obj,
                    }
                )
            )
        catalog = vstack(cats)

        # Mock data based on that catalog
        obj_cat = kbmock.ObjectCatalog.from_table(catalog, mode="folding")

        prim_hdr_factory = kbmock.HeaderFactory.from_primary_template(
            mutables=["DATE-OBS"],
            callbacks=[kbmock.ObstimeIterator(self.timestamps)],
        )

        factory = kbmock.SimpleFits(shape=self.shape, obj_cat=obj_cat)
        factory.prim_hdr = prim_hdr_factory
        hduls = factory.mock(n=self.n_imgs)

        obj_cat.reset()
        cats = obj_cat.mock(n=self.n_imgs, t=self.timestamps)
        self.validate_cat_render(hduls, cats)

    def test_progressive_sky_cat(self):
        """Test progressive catalog based on on-sky coordinates."""
        # a 10-50 in x by a 10-90 in y box using default WCS
        # self.shape = (500, 500)
        param_ranges = {
            "ra_mean": (350.998, 351.002),
            "dec_mean": (-5.0077, -5.0039),
            "v_ra": [-0.001, 0.0001],
            "v_dec": [0, 0],
            "amplitude": [100, 100],
            "x_stddev": [2.0, 2.0],
            "y_stddev": [2.0, 2.0],
        }
        catalog = kbmock.gen_random_catalog(self.n_obj, param_ranges)
        obj_cat = kbmock.ObjectCatalog.from_table(catalog, kind="world")

        factory = kbmock.SimpleFits(shape=self.shape, step_t=self.step_t, obj_cat=obj_cat)
        hduls = factory.mock(n=self.n_imgs)

        # Run tests and ensure we have rendered the object in correct
        # positions
        obj_cat.reset()
        wcs = [WCS(h["IMAGE"].header) for h in hduls]
        cats = obj_cat.mock(n=self.n_imgs, dt=self.step_t, wcs=wcs)
        self.validate_cat_render(hduls, cats)

    def test_folding_sky_cat(self):
        """Test folding catalog based on on-sky coordinates."""
        # a 20x20 box in pixels using a default WCS
        start_ra = np.linspace(350.998, 351.002, self.n_obj)
        start_dec = np.linspace(-5.0077, -5.0039, self.n_obj)

        cats = []
        for i, t in enumerate(self.timestamps):
            cats.append(
                Table(
                    {
                        "amplitude": [100] * self.n_obj,
                        "obstime": [t] * self.n_obj,
                        "ra_mean": start_ra - 0.001 * i,
                        "dec_mean": start_dec,  # + 0.00011 * i,
                        "stddev": [2.0] * self.n_obj,
                    }
                )
            )
        catalog = vstack(cats)
        obj_cat = kbmock.ObjectCatalog.from_table(catalog, kind="world", mode="folding")

        prim_hdr_factory = kbmock.HeaderFactory.from_primary_template(
            mutables=["DATE-OBS"],
            callbacks=[kbmock.ObstimeIterator(self.timestamps)],
        )

        factory = kbmock.SimpleFits(shape=self.shape, obj_cat=obj_cat)
        factory.prim_hdr = prim_hdr_factory
        hduls = factory.mock(n=self.n_imgs)

        obj_cat.reset()
        wcs = [WCS(h[1].header) for h in hduls]
        cats = obj_cat.mock(n=self.n_imgs, t=self.timestamps, wcs=wcs)
        self.validate_cat_render(hduls, cats)

    # TODO: move to pytest and mark as xfail
    def test_noise_gen(self):
        """Test noise renders with expected statistical properties."""
        factory = kbmock.SimpleFits(shape=(1000, 1000), with_noise=True)
        hdul = factory.mock()[0]
        self.assertAlmostEqual(hdul["IMAGE"].data.mean(), 10, 1)
        self.assertAlmostEqual(hdul["IMAGE"].data.std(), 1, 1)

        factory = kbmock.SimpleFits(shape=(1000, 1000), with_noise=True, noise="realistic")
        hdul = factory.mock()[0]
        self.assertAlmostEqual(hdul["IMAGE"].data.mean(), 32, 1)
        self.assertAlmostEqual(hdul["IMAGE"].data.std(), 7.5, 0)

        img_factory = kbmock.SimpleImage(shape=(1000, 1000), add_noise=True, noise=5, noise_std=2.0)
        factory = kbmock.SimpleFits()
        factory.img_data = img_factory
        hduls = factory.mock(n=3)

        for hdul in hduls[1:]:
            self.assertFalse((hduls[0]["IMAGE"].data == hdul["IMAGE"].data).all())
            self.assertAlmostEqual(hdul["IMAGE"].data.mean(), 5, 1)
            self.assertAlmostEqual(hdul["IMAGE"].data.std(), 2, 1)


class TestDiffIm(unittest.TestCase):
    def test(self):
        """Test basic functionality of SimpleFits factory."""
        factory = kbmock.DECamImdiff()
        hduls = factory.mock(2)

        names = [
            "IMAGE",
            "MASK",
            "VARIANCE",
            "ARCHIVE_INDEX",
            "FilterLabel",
            "Detector",
            "TransformMap",
            "ExposureSummaryStats",
            "Detector",
            "KernelPsf",
            "FixedKernel",
            "SkyWcs",
            "ApCorrMap",
            "ChebyshevBoundedField",
            "ChebyshevBoundedField",
        ]
        hdul = hduls[0]
        self.assertEqual(len(hduls), 2)
        self.assertEqual(len(hduls[0]), 16)
        for name, hdu in zip(names, hdul[1:]):
            self.assertEqual(name, hdu.name)
        self.assertEqual(hdul["PRIMARY"].data, None)

        factory = kbmock.DECamImdiff(with_data=True)
        hduls = factory.mock(2)
        hdul = hduls[0]
        self.assertEqual(hdul["IMAGE"].data.shape, (2048, 4096))
        self.assertEqual(hdul["VARIANCE"].data.shape, (2048, 4096))
        self.assertEqual(hdul["MASK"].data.shape, (2048, 4096))


if __name__ == "__main__":
    unittest.main()
