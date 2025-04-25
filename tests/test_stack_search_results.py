import unittest

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.run_search import SearchRunner
from kbmod.search import ImageStack, LayeredImage, StackSearch, Trajectory


class test_search(unittest.TestCase):
    def setUp(self):
        self.num_times = 10
        self.width = 256
        self.height = 256
        self.num_objs = 5

        self.times = create_fake_times(self.num_times, obs_per_day=3)
        self.fake_ds = FakeDataSet(self.width, self.height, self.times)
        for _ in range(self.num_objs):
            self.fake_ds.insert_random_object(500)

        self.search = StackSearch(self.fake_ds.stack)
        self.fake_trjs = self.fake_ds.trajectories

    def test_set_get_results(self):
        results = self.search.get_results(0, 10)
        self.assertEqual(len(results), 0)

        trjs = [Trajectory(i, i, 0.0, 0.0) for i in range(10)]
        self.search.set_results(trjs)

        # Check that we extract them all.
        results = self.search.get_results(0, 10)
        self.assertEqual(len(results), 10)
        for i in range(10):
            self.assertEqual(results[i].x, i)

        # Check that we can run past the end of the results.
        results = self.search.get_results(0, 100)
        self.assertEqual(len(results), 10)

        # Check that we can pull a subset.
        results = self.search.get_results(2, 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].x, 2)
        self.assertEqual(results[1].x, 3)

        # Check that we can pull a subset aligned with the end.
        results = self.search.get_results(8, 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].x, 8)
        self.assertEqual(results[1].x, 9)

        # Check invalid settings
        self.assertRaises(RuntimeError, self.search.get_results, 0, 0)

        # Check that we can clear the results.
        self.search.clear_results()
        self.assertEqual(len(self.search.get_all_results()), 0)

    def test_psi_phi_curves(self):
        psi_phi_curves = self.search.get_all_psi_phi_curves(self.fake_trjs)
        psi_curves = psi_phi_curves[:, : self.num_times]
        self.assertEqual(psi_curves.shape[0], self.num_objs)
        self.assertEqual(psi_curves.shape[1], self.num_times)
        self.assertTrue(np.all(psi_curves > 0.0))

        phi_curves = psi_phi_curves[:, self.num_times :]
        self.assertEqual(phi_curves.shape[0], self.num_objs)
        self.assertEqual(phi_curves.shape[1], self.num_times)
        self.assertTrue(np.all(phi_curves > 0.0))

    def test_psi_phi_curves_known(self):
        height = 5
        width = 4
        num_times = 5

        images = []
        expected_psi = []
        expected_phi = []
        for i in range(num_times):
            img = LayeredImage(
                np.full((height, width), float(i), dtype=np.float32),  # sci
                np.full((height, width), 0.1, dtype=np.float32),  # var
                np.zeros((height, width), dtype=np.float32),  # mask
                np.array([[1.0]], dtype=np.float32),  # no-op PSF
                float(i),
            )
            images.append(img)
            expected_psi.append(float(i) / 0.1)
            expected_phi.append(1.0 / 0.1)

        stack = ImageStack(images)
        search = StackSearch(stack)

        trj = Trajectory(x=2, y=2, vx=0.0, vy=0.0)
        psi_phi = search.get_all_psi_phi_curves([trj])
        self.assertEqual(psi_phi.shape, (1, 2 * num_times))
        self.assertTrue(np.allclose(psi_phi[0, :num_times], expected_psi))
        self.assertTrue(np.allclose(psi_phi[0, num_times:], expected_phi))

    def test_load_and_filter_results_lh(self):
        time_list = [i / self.num_times for i in range(self.num_times)]
        fake_ds = FakeDataSet(
            self.width,
            self.height,
            time_list,
            noise_level=1.0,
            psf_val=0.5,
            use_seed=True,
        )

        # Create fake result trajectories with given initial likelihoods. The two final ones
        # are filtered by min likelihood.
        trjs = [
            Trajectory(20, 20, 0, 0, 110.0, 110.0, self.num_times),
            Trajectory(30, 30, 0, 0, 100.0, 100.0, self.num_times),
            Trajectory(40, 40, 0, 0, 50.0, 50.0, self.num_times),
            Trajectory(41, 41, 0, 0, 50.0, 50.0, self.num_times),
            Trajectory(42, 42, 0, 0, 50.0, 50.0, self.num_times),
            Trajectory(43, 43, 0, 0, 50.0, 50.0, self.num_times),
            Trajectory(50, 50, 0, 0, 1.0, 2.0, self.num_times),
            Trajectory(60, 60, 0, 0, 1.0, 1.0, self.num_times),
        ]
        for trj in trjs:
            fake_ds.insert_object(trj)

        # Create the stack search and insert the fake results.
        search = StackSearch(fake_ds.stack)
        search.set_results(trjs)

        # Do the loading and filtering.
        config = SearchConfiguration()
        overrides = {
            "clip_negative": False,
            "chunk_size": 500000,
            "lh_level": 10.0,
            "num_cores": 1,
            "num_obs": 5,
            "sigmaG_lims": [25, 75],
        }
        config.set_multiple(overrides)

        runner = SearchRunner()
        results = runner.load_and_filter_results(search, config)

        # Only two of the middle results should pass the filtering.
        self.assertEqual(len(results), 6)
        self.assertEqual(results["y"][0], 20)
        self.assertEqual(results["y"][1], 30)
        self.assertEqual(results["y"][2], 40)
        self.assertEqual(results["y"][3], 41)
        self.assertEqual(results["y"][4], 42)
        self.assertEqual(results["y"][5], 43)

        # Rerun the search with a small chunk_size to make sure we still
        # find everything.
        overrides["chunk_size"] = 2
        results = runner.load_and_filter_results(search, config)
        self.assertEqual(len(results), 6)
        self.assertEqual(results["y"][0], 20)
        self.assertEqual(results["y"][1], 30)
        self.assertEqual(results["y"][2], 40)
        self.assertEqual(results["y"][3], 41)
        self.assertEqual(results["y"][4], 42)
        self.assertEqual(results["y"][5], 43)


if __name__ == "__main__":
    unittest.main()
