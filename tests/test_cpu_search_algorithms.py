import unittest

import numpy as np

from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.search import (
    evaluate_trajectory_cpu,
    fill_psi_phi_array_from_image_arrays,
    search_cpu_only,
    PsiPhiArray,
    SearchParameters,
    Trajectory,
    TrajectoryList,
)


class test_cpu_search_algorithms(unittest.TestCase):
    def setUp(self):
        # Create fake data
        self.num_times = 10
        self.width = 128
        self.height = 125
        self.num_objs = 5

        self.times = create_fake_times(self.num_times, obs_per_day=3)
        self.fake_ds = FakeDataSet(self.width, self.height, self.times)
        for _ in range(self.num_objs):
            self.fake_ds.insert_random_object(500)
        self.fake_trjs = self.fake_ds.trajectories

        # Create the phi and psi data.
        self.psi_phi = PsiPhiArray()
        fill_psi_phi_array_from_image_arrays(
            self.psi_phi,
            4,
            self.fake_ds.stack_py.sci,
            self.fake_ds.stack_py.var,
            self.fake_ds.stack_py.psfs,
            self.fake_ds.stack_py.zeroed_times,
        )

    def test_evaluate_trajectory_cpu(self):
        candidate = Trajectory(
            x=self.fake_trjs[0].x,
            y=self.fake_trjs[0].y,
            vx=self.fake_trjs[0].vx,
            vy=self.fake_trjs[0].vy,
        )
        self.assertEqual(candidate.obs_count, 0)
        self.assertEqual(candidate.lh, 0.0)

        evaluate_trajectory_cpu(self.psi_phi, candidate)
        self.assertGreater(candidate.obs_count, 0)
        self.assertGreater(candidate.lh, 0.0)

    def test_search_cpu_only(self):
        params = SearchParameters()
        params.min_observations = 5
        params.min_lh = 1.0
        params.do_sigmag_filter = False
        params.x_start_min = 0
        params.x_start_max = self.width
        params.y_start_min = 0
        params.y_start_max = self.height
        params.results_per_pixel = 4

        # Create a search candidate for each of the fake object's velocities.
        candidates = TrajectoryList(self.num_objs)
        for idx in range(self.num_objs):
            candidates.set_trajectory(
                idx, Trajectory(x=0, y=0, vx=self.fake_trjs[idx].vx, vy=self.fake_trjs[idx].vy)
            )

        # Run the search
        num_results = params.results_per_pixel * self.width * self.height
        results = TrajectoryList(num_results)
        search_cpu_only(self.psi_phi, params, candidates, results)
        self.assertEqual(len(results), num_results)

        # Check that we see the correct number of results per pixel and that the
        # matches with the true fakes are in there.
        counts = np.zeros((self.height, self.width), dtype=int)
        for idx in range(num_results):
            trj = results.get_trajectory(idx)
            counts[trj.y, trj.x] += 1

            # If this result corresponds to a fake, check that it has a high likelihood
            # and is the first result returned for that pixel.
            for fake in self.fake_trjs:
                trj_vals = np.array([trj.x, trj.y, trj.vx, trj.vy])
                fake_vals = np.array([fake.x, fake.y, fake.vx, fake.vy])
                if np.all(np.abs(trj_vals - fake_vals) <= 0.5):
                    self.assertEqual(counts[trj.y, trj.x], 1)
                    self.assertGreater(trj.lh, 10.0)


if __name__ == "__main__":
    unittest.main()
