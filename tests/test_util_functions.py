from astropy.table import Table
import numpy as np
import unittest

from pathlib import Path

from kbmod.core.image_stack_py import LayeredImagePy
from kbmod.core.psf import PSF
from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod.util_functions import (
    get_matched_obstimes,
    load_deccam_layered_image,
    mjd_to_day,
    unravel_results,
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
        img = load_deccam_layered_image(img_path, PSF.make_gaussian_kernel(1.0))

        self.assertTrue(isinstance(img, LayeredImagePy))
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)

    def test_unravel_results(self):
        trj_list = []
        num_images = 10
        num_trjs = 11

        for i in range(num_trjs):
            trj = Trajectory(
                x=i,
                y=i + 0,
                vx=i - 2.0,
                vy=i + 5.0,
                flux=5.0 * i,
                lh=100.0 + i,
                obs_count=num_images,
            )
            trj_list.append(trj)

        res = Results.from_trajectories(trj_list)

        # create an "image collection" with all required fields
        ic = Table()
        ic["mjd_mid"] = [60000 + i for i in range(num_images)]
        ic["band"] = ["g" for _ in range(num_images)]
        ic["zeroPoint"] = [31.4 for _ in range(num_images)]

        res.table.meta["mjd_mid"] = ic["mjd_mid"]
        valids = [[True] * num_images for _ in range(num_trjs)]
        valids[int(num_images / 2)][-1] = False  # make one observation invalid
        res.table["obs_valid"] = valids
        obs_count = [num_images for _ in range(num_trjs)]
        obs_count[int(num_images / 2)] = num_images - 1
        res.table["obs_count"] = obs_count
        res.table["img_ra"] = [
            np.array([j + (i * 0.1) for j in range(num_images)]) for i in range(num_trjs)
        ]
        res.table["img_dec"] = [
            np.array([j + (i * 0.1) for j in range(num_images)]) for i in range(num_trjs)
        ]
        print(res.table["img_ra"])
        print(res.table["img_ra"][0])
        print(res.table["obs_valid"])

        df = unravel_results(res, ic)
        self.assertEqual(len(df), (num_images * num_trjs) - 1)


if __name__ == "__main__":
    unittest.main()
