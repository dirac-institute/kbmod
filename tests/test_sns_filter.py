import unittest
import numpy as np
from kbmod.results import Results
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.filters.sns_filters import no_op_filter


class TestSnsFilter(unittest.TestCase):
    def test_no_op_does_not_modify_results(self):
        num_times = 10
        height = 40
        width = 50
        times = np.arange(num_times)

        # Create a fake data set a few fake objects with different fluxes.
        ds = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)
        ds.insert_random_object(flux=5)
        ds.insert_random_object(flux=20)
        ds.insert_random_object(flux=50)
        results = Results.from_trajectories(ds.trajectories, track_filtered=False)
        length = len(results)
        
        no_op_filter(results)
        self.assertEqual(length, len(results))


if __name__ == "__main__":
    unittest.main()
