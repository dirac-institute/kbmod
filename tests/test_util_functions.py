import numpy as np
import unittest

from pathlib import Path

from kbmod.search import LayeredImage, PSF
from kbmod.util_functions import (
    get_matched_obstimes,
    load_deccam_layered_image,
    mjd_to_day,
)


class test_util_functions(unittest.TestCase):
    def test_get_matched_obstimes(self):
        obstimes = [1.0, 2.0, 3.0, 4.0, 6.0, 7.5, 9.0, 10.1]
        query_times = [-1.0, 0.999999, 1.001, 1.1, 1.999, 6.001, 7.499, 7.5, 10.099999, 10.10001, 20.0]
        matched_inds = get_matched_obstimes(obstimes, query_times, threshold=0.01)
        expected = [-1, 0, 0, -1, 1, 4, 5, 5, 7, 7, -1]
        self.assertTrue(np.array_equal(matched_inds, expected))

    def test_mjd_to_day(self):
        self.assertEqual(mjd_to_day(60481.04237269), "2024-06-20")
        self.assertEqual(mjd_to_day(60500), "2024-07-09")
        self.assertEqual(mjd_to_day(58100.5), "2017-12-13")

    def test_load_deccam_layered_image(self):
        base_path = Path(__file__).parent.parent
        img_path = base_path / "data" / "demo_image.fits"
        img = load_deccam_layered_image(img_path, PSF(1.0))

        self.assertTrue(isinstance(img, LayeredImage))
        self.assertGreater(img.get_width(), 0)
        self.assertGreater(img.get_height(), 0)


if __name__ == "__main__":
    unittest.main()
