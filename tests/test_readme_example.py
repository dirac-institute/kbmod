import unittest

from kbmod.search import HAS_CUDA, StackSearch, Trajectory
from kbmod.trajectory_generator import KBMODV1Search
from kbmod.fake_data.fake_data_creator import *


class test_readme_example(unittest.TestCase):
    @unittest.skipIf(not HAS_CUDA, "Skipping test (no GPU detected)")
    def test_make_and_copy(self):
        fake_times = create_fake_times(10, t0=57130.2)
        ds = FakeDataSet(512, 512, fake_times)

        # Insert an artificial object with starting position x=2, y=0,
        # velocity vx=10.7, vy=15.3, and flux = 275.0.
        trj = Trajectory(x=2, y=0, vx=10.7, vy=15.3, flux=275.0)
        ds.insert_object(trj)

        # Generate a set of trajectories to test from each pixel.
        gen = KBMODV1Search(
            5,  # Number of search velocities to try (0, 0.8, 1.6, 2.4, 3.2)
            0,  # The minimum search velocity to test (inclusive)
            4,  # The maximum search velocity to test (exclusive)
            5,  # Number of search angles to try (-0.1, -0.06, -0.02, 0.02, 0.6)
            -0.1,  # The minimum search angle to test (inclusive)
            0.1,  # The maximum search angle to test (exclusive)
        )
        candidates = [trj for trj in gen]

        # Do the actual search (on CPU).  This requires passing in the science
        # images, the variance images, the PSF information, and the times.
        search = StackSearch(
            ds.stack_py.sci,
            ds.stack_py.var,
            ds.stack_py.psfs,
            ds.stack_py.zeroed_times,
        )
        search.set_min_obs(7)
        search.search_all(candidates, False)

        # Get the top 10 results.
        results = search.get_results(0, 10)


if __name__ == "__main__":
    unittest.main()
