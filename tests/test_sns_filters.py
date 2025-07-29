import unittest
import numpy as np
from kbmod.results import Results
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.configuration import SearchConfiguration
from kbmod.search import Trajectory
from kbmod.filters.sns_filters import *


class TestSnsFilter(unittest.TestCase):
    def test_peak_offset_filter_throws_exception(self):
        # empty results raises runtime exception since there is not "coadd_mean" column
        empty_results = Results()
        self.assertRaises(RuntimeError, peak_offset_filter, empty_results)

    def test_peak_offset_filter(self):
        num_times = 10
        height = 40
        width = 50
        times = np.arange(num_times) + 60676  # MJD for Jan 1, 2025

        # Create a fake data set a few fake objects with different fluxes.
        ds = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)
        for itr in np.arange(5):
            ds.insert_random_object(flux=5 * itr)
            itr += 1

        results = ds.make_results()
        self.assertTrue("coadd_mean" in results.colnames)

        # Ensure that a peak_offset_max of 0 filters everything out
        peak_offset_filter(results, peak_offset_max=0)
        self.assertEqual(0, len(results))

        # Ensure that a peak_offset_max of 10,000 filters nothing out
        results = ds.make_results()
        peak_offset_filter(results, peak_offset_max=10000)
        self.assertEqual(5, len(results))

        # Insert a sixth object and edit it to be outside of a peak offset of 2.
        # Two objects should be filtered out.
        trj = ds.insert_random_object(flux=25)
        trj.x = trj.x - 2 if trj.x >= 38 else trj.x + 2
        results = ds.make_results()
        peak_offset_filter(results, peak_offset_max=1)
        self.assertEqual(4, len(results))

    def test_predictive_line_cluster(self):
        num_times = 10
        height = 40
        width = 50
        times = np.arange(num_times) + 60676  # MJD for Jan 1, 2025

        # Create a fake data set a few fake objects with different fluxes.
        ds1 = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)

        trj = ds1.insert_random_object(5)
        for it in np.arange(5):
            ds1.trajectories.append(trj)
        results = ds1.make_results()

        self.assertEqual(6, len(results))  # 6 total trajectories inserted
        predictive_line_cluster(results, times)
        # All 6 trajectories are the same so expect them to be clustered into 1
        self.assertEqual(1, len(results))

        # Insert 5 trajectories slightly perturbed from each other
        ds2 = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)
        trj = ds2.insert_random_object(5)

        trj1 = Trajectory()
        trj1.x = trj.x + 1
        trj1.y = trj.y + 1

        trj2 = Trajectory()
        trj2.x = trj.x + 1
        trj2.y = trj.y - 1

        trj3 = Trajectory()
        trj3.x = trj.x - 1
        trj3.x = trj.x + 1

        trj4 = Trajectory()
        trj4.x = trj.x - 1
        trj4.x = trj.x - 1

        ds2.trajectories.append(trj1)
        ds2.trajectories.append(trj2)
        ds2.trajectories.append(trj3)
        ds2.trajectories.append(trj4)
        results = ds2.make_results()
        self.assertEqual(5, len(results))  # 5 total trajectories inserted
        predictive_line_cluster(results, times)
        # All 5 trajectories are close together so expect them to be clustered into 1
        self.assertEqual(1, len(results))


if __name__ == "__main__":
    unittest.main()
