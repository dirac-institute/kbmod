import numpy as np
import pathlib
import unittest

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.filters.stamp_filters import *
from kbmod.results import Results
from kbmod.search import Trajectory


class test_stamp_filters(unittest.TestCase):
    def setUp(self):
        # Create a fake data set to use in the tests.
        self.image_count = 10

        # 4 observations per day with a 1 day gap between nights.
        self.fake_times = create_fake_times(self.image_count, 57130.2, 4, 0.01, 1)
        self.ds = FakeDataSet(
            25,  # width
            35,  # height
            self.fake_times,  # time stamps
            noise_level=1.0,  # noise level
            psf_val=0.5,  # psf value
            psfs=None,  # No per-image PSFs
            use_seed=101,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        self.trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        self.ds.insert_object(self.trj)

        # Create a second simpler fake data set where each science layer is constant
        # according the the time index.
        height = 25
        width = 35
        self.known_stack = ImageStackPy(
            times=self.fake_times,
            sci=[np.full((height, width), float(i)) for i in range(self.image_count)],
            var=[np.full((height, width), 0.5) for i in range(self.image_count)],
        )

    def test_make_coadds(self):
        # Three trajectories: One in the image the whole time, the second off the edge,
        # and the third off the edge at the end.
        trj_list = [
            self.trj,
            Trajectory(1, 1, 0.0, 0.0),
            Trajectory(15, 25, 2.0, 3.0),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("stamp" in keep.colnames)

        # Make the stamps.
        coadd_types = ["mean"]
        append_coadds(keep, self.known_stack, coadd_types, 5)

        # We do not filter, so everything should be saved and have the same mean.
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertEqual(len(keep), 3)

        # The first stamp should be the mean of all the images.
        self.assertEqual(keep["coadd_mean"][0].shape, (11, 11))
        self.assertTrue(np.allclose(keep["coadd_mean"][0], np.full((11, 11), 4.5)))

        # The second stamp should have zeros where it is off the edge.
        self.assertEqual(keep["coadd_mean"][1].shape, (11, 11))
        expected = np.zeros((11, 11))
        expected[4:, 4:] = 4.5
        self.assertTrue(np.allclose(keep["coadd_mean"][1], expected))

        # The second stamp should be different since it runs off the edge
        # at different points.
        self.assertEqual(keep["coadd_mean"][2].shape, (11, 11))
        self.assertGreater(len(np.unique(keep["coadd_mean"][2])), 2)

    def test_make_coadds_empty(self):
        """Check that we can make coadds when there are no results."""
        keep = Results()
        self.assertEqual(len(keep), 0)
        self.assertFalse("coadd_mean" in keep.colnames)

        append_coadds(keep, self.known_stack, ["mean"], 5)
        self.assertTrue("coadd_mean" in keep.colnames)

    def test_make_coadds_no_valid_times(self):
        """Check that we can make coadds when there are no valid times."""
        keep = Results.from_trajectories([self.trj])
        self.assertEqual(len(keep), 1)

        obs_valid = np.full((1, self.image_count), False)
        keep.update_obs_valid(obs_valid)

        append_coadds(keep, self.known_stack, ["mean", "median", "sum"], 5)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_median" in keep.colnames)
        self.assertTrue("coadd_sum" in keep.colnames)

    def test_get_coadds_and_filter_with_invalid(self):
        valid1 = [True] * self.image_count
        valid2 = [True] * self.image_count
        # Completely mess up some of the images.
        for i in [1, 4, 6, 7, 9]:
            self.ds.stack_py.sci[i][:, :] = 1000.0
            valid2[i] = False

        # Create the Results with nearly identical trajectories,
        # but different valid observations
        trj2 = Trajectory(self.trj.x, self.trj.y, self.trj.vx, self.trj.vy + 0.001, flux=250.0)
        keep = Results.from_trajectories([self.trj, trj2])
        keep.update_obs_valid(np.array([valid1, valid2]))

        # Make the stamps, check that there were saved, and check they are correct.
        append_coadds(keep, self.known_stack, ["mean", "median"], 5)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_median" in keep.colnames)
        self.assertFalse("coadd_mean_2015-04-18" in keep.colnames)
        self.assertFalse("coadd_mean_2015-04-19" in keep.colnames)
        self.assertFalse("coadd_mean_2015-04-20" in keep.colnames)
        self.assertEqual(len(keep), 2)

        self.assertEqual(keep["coadd_mean"][0].shape, (11, 11))
        self.assertTrue(np.allclose(keep["coadd_mean"][0], np.full((11, 11), 4.5)))

        self.assertEqual(keep["coadd_mean"][1].shape, (11, 11))
        self.assertTrue(np.allclose(keep["coadd_mean"][1], np.full((11, 11), 3.6)))

        # The median is 4.0 instead of 4.5 because pytorch's median takes the
        # lower of the two middle values (instead of the average) when there is
        # an even number of samples.
        self.assertEqual(keep["coadd_median"][0].shape, (11, 11))
        self.assertTrue(np.allclose(keep["coadd_median"][0], np.full((11, 11), 4.0)))

        self.assertEqual(keep["coadd_median"][1].shape, (11, 11))
        self.assertTrue(np.allclose(keep["coadd_median"][1], np.full((11, 11), 3.0)))

    def test_make_coadds_nightly(self):
        valid1 = [True] * self.image_count
        valid2 = [True] * self.image_count
        # Completely mess up some of the images.
        for i in [1, 4, 6, 7, 9]:
            self.ds.stack_py.sci[i][:, :] = 1000.0
            valid2[i] = False

        # Make the stamps for a single trajectory.
        keep = Results.from_trajectories([self.trj, self.trj])
        keep.update_obs_valid(np.array([valid1, valid2]))
        append_coadds(keep, self.known_stack, ["mean"], 1, nightly=True)

        # Check we have coadds for each night (and overall coadds)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_mean_2015-04-18" in keep.colnames)
        self.assertTrue("coadd_mean_2015-04-19" in keep.colnames)
        self.assertTrue("coadd_mean_2015-04-20" in keep.colnames)
        self.assertEqual(len(keep), 2)

        # Check the values are correct.
        self.assertTrue(np.allclose(keep["coadd_mean"][0], np.full((3, 3), 4.5)))
        self.assertTrue(np.allclose(keep["coadd_mean"][1], np.full((3, 3), 3.6)))
        self.assertTrue(np.allclose(keep["coadd_mean_2015-04-18"][0], np.full((3, 3), 1.5)))
        self.assertTrue(np.allclose(keep["coadd_mean_2015-04-19"][0], np.full((3, 3), 5.5)))
        self.assertTrue(np.allclose(keep["coadd_mean_2015-04-20"][0], np.full((3, 3), 8.5)))
        self.assertTrue(np.allclose(keep["coadd_mean_2015-04-18"][1], np.full((3, 3), 5.0 / 3.0)))
        self.assertTrue(np.allclose(keep["coadd_mean_2015-04-19"][1], np.full((3, 3), 5.0)))
        self.assertTrue(np.allclose(keep["coadd_mean_2015-04-20"][1], np.full((3, 3), 8.0)))

    def test_append_coadds(self):
        # Create trajectories to test: 0) known good, 1) completely wrong
        # 2) close to good, but offset], 3) just close enough, and
        # 4) another wrong one.
        trj_list = [
            self.trj,
            Trajectory(1, 1, 0.0, 0.0),
            Trajectory(self.trj.x + 2, self.trj.y + 2, self.trj.vx, self.trj.vy),
            Trajectory(self.trj.x + 1, self.trj.y + 1, self.trj.vx, self.trj.vy),
            Trajectory(10, 3, 0.1, -0.1),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertFalse("coadd_mean" in keep.colnames)
        self.assertFalse("coadd_median" in keep.colnames)
        self.assertFalse("coadd_weighted" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Adding nothing does nothing.
        append_coadds(keep, self.ds.stack_py, [], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertFalse("coadd_mean" in keep.colnames)
        self.assertFalse("coadd_median" in keep.colnames)
        self.assertFalse("coadd_weighted" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Adding "mean" and "median" does only those.
        append_coadds(keep, self.ds.stack_py, ["median", "mean"], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_median" in keep.colnames)
        self.assertFalse("coadd_weighted" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # We can add "weighted" later.
        append_coadds(keep, self.ds.stack_py, ["weighted"], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_median" in keep.colnames)
        self.assertTrue("coadd_weighted" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Check that all coadds are generated without filtering.
        for i in range(len(trj_list)):
            self.assertEqual(keep["coadd_mean"][i].shape, (7, 7))
            self.assertEqual(keep["coadd_median"][i].shape, (7, 7))

    def test_append_all_stamps(self):
        # Make a few results with different trajectories.
        trj_list = [
            Trajectory(8, 7, 2.0, 1.0),
            Trajectory(10, 22, -2.0, -1.0),
            Trajectory(8, 7, -2.0, -1.0),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("all_stamps" in keep.colnames)

        append_all_stamps(keep, self.ds.stack_py, 5)
        self.assertTrue("all_stamps" in keep.colnames)
        for i in range(len(keep)):
            stamps_array = keep["all_stamps"][i]
            self.assertEqual(stamps_array.shape[0], self.image_count)
            self.assertEqual(stamps_array.shape[1], 11)
            self.assertEqual(stamps_array.shape[2], 11)

        # Check that everything works if the results are empty.
        keep2 = Results.from_trajectories([])
        append_all_stamps(keep2, self.ds.stack_py, 5)
        self.assertTrue("all_stamps" in keep2.colnames)

    def test_append_all_stamps_empty(self):
        keep = Results()
        self.assertEqual(len(keep), 0)
        self.assertFalse("all_stamps" in keep.colnames)

        append_all_stamps(keep, self.ds.stack_py, 5)
        self.assertEqual(len(keep), 0)
        self.assertTrue("all_stamps" in keep.colnames)

    def test_filter_stamps_by_cnn(self):
        import torch

        torch.manual_seed(747474747)

        trj_list = [
            self.trj,
            Trajectory(self.trj.x, self.trj.y, 0.0, 0.0),
            Trajectory(self.trj.x + 2, self.trj.y + 2, self.trj.vx, self.trj.vy),
        ]
        keep = Results.from_trajectories(trj_list)
        append_coadds(keep, self.ds.stack_py, ["mean"], 5)

        filter_stamps_by_cnn(
            keep,
            None,
            coadd_type="mean",
            stamp_radius=3,
            coadd_radius=5,
        )

        # the test model was trained on totally random data
        assert keep.table["cnn_class"].data[2] == True
        filtered_results = keep.filter_rows(keep.table["cnn_class"])
        assert len(filtered_results) == 2

        # assert that the function fails
        # when `Results` doesn't have needed
        # coadd column.
        with self.assertRaises(ValueError):
            filter_stamps_by_cnn(
                keep,
                None,
                coadd_type="median",
                stamp_radius=3,
            )

        with self.assertRaises(ValueError):
            keep2 = Results.from_trajectories(trj_list)
            append_coadds(keep2, self.ds.stack_py, ["mean"], 1)
            filter_stamps_by_cnn(
                keep,
                None,
                coadd_type="mean",
                stamp_radius=3,
                coadd_radius=1,
            )


if __name__ == "__main__":
    unittest.main()
