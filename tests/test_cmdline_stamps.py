import unittest

import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.fake_data.fake_data_creator import create_fake_times
from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod_cmdline.kbmod_stamps import generate_coadds


class TestCmdlineStamps(unittest.TestCase):
    def setUp(self):
        self.times = create_fake_times(10, 57130.2, 4, 0.01, 1)
        self.images = ImageStackPy(
            times=self.times,
            sci=[np.full((25, 35), float(i)) for i in range(10)],
            var=[np.full((25, 35), 0.5) for _ in range(10)],
        )
        trj = Trajectory(8, 7, 0.0, 0.0)
        self.results = Results.from_trajectories([trj, trj])

        valid = np.full((2, 10), True)
        valid[1, [1, 4, 6, 7, 9]] = False
        self.results.update_obs_valid(valid)

    def test_generate_coadds_uses_valid_observations(self):
        coadds = generate_coadds(self.results, self.images, "mean", radius=1)

        self.assertEqual(coadds.shape, (2, 3, 3))
        self.assertTrue(np.allclose(coadds[0], 4.5))
        self.assertTrue(np.allclose(coadds[1], 3.6))

    def test_generate_coadds_preserves_selection_and_nightly_shape(self):
        coadds = generate_coadds(
            self.results,
            self.images,
            "mean",
            radius=1,
            indices=[1],
            nightly=True,
        )

        self.assertEqual(coadds.shape, (1, 3, 3, 3))
        self.assertTrue(np.allclose(coadds[0, 0], 5.0 / 3.0))
        self.assertTrue(np.allclose(coadds[0, 1], 5.0))
        self.assertTrue(np.allclose(coadds[0, 2], 8.0))

    def test_generate_weighted_nightly_coadds(self):
        coadds = generate_coadds(
            self.results,
            self.images,
            "weighted",
            radius=1,
            indices=[1],
            nightly=True,
        )

        self.assertEqual(coadds.shape, (1, 3, 3, 3))
        self.assertTrue(np.allclose(coadds[0, 0], 5.0 / 3.0))
        self.assertTrue(np.allclose(coadds[0, 1], 5.0))
        self.assertTrue(np.allclose(coadds[0, 2], 8.0))


if __name__ == "__main__":
    unittest.main()
