import numpy as np
import pathlib
import unittest

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.image_utils import extract_sci_images_from_stack, extract_var_images_from_stack
from kbmod.filters.stamp_filters import *
from kbmod.results import Results
from kbmod.search import *


class test_stamp_filters(unittest.TestCase):
    def setUp(self):
        # Create a fake data set to use in the tests.
        self.image_count = 10
        self.fake_times = create_fake_times(self.image_count, 57130.2, 1, 0.01, 1)
        self.ds = FakeDataSet(
            25,  # width
            35,  # height
            self.fake_times,  # time stamps
            1.0,  # noise level
            0.5,  # psf value
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        self.trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        self.ds.insert_object(self.trj)

        # Create a Python version of the image stack.
        self.stack_py = ImageStackPy(
            times=self.fake_times,
            sci=extract_sci_images_from_stack(self.ds.stack),
            var=extract_var_images_from_stack(self.ds.stack),
        )

        current_dir = pathlib.Path(__file__).parent.resolve()
        self.model_path = pathlib.Path(current_dir, "data/test_model.keras")

    def test_make_coadds(self):
        # Create trajectories to test: 0) known good, 1) completely wrong
        # 2) close to good, but offset], and 3) just close enough.
        trj_list = [
            self.trj,
            Trajectory(1, 1, 0.0, 0.0),
            Trajectory(self.trj.x + 2, self.trj.y + 2, self.trj.vx, self.trj.vy),
            Trajectory(self.trj.x + 1, self.trj.y + 1, self.trj.vx, self.trj.vy),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("stamp" in keep.colnames)

        # Make the stamps.
        coadd_types = ["mean"]
        append_coadds(keep, self.ds.stack, coadd_types, 5)

        # We do not filter, so everything should be saved.
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertEqual(len(keep), 4)
        self.assertEqual(keep["coadd_mean"][0].shape, (11, 11))

        # Check that we get the same coadd values if we use a Python image stack.
        keep2 = Results.from_trajectories(trj_list)
        append_coadds(keep2, self.stack_py, coadd_types, 5)
        self.assertTrue("coadd_mean" in keep2.colnames)
        self.assertEqual(len(keep2), 4)
        self.assertEqual(keep2["coadd_mean"][0].shape, (11, 11))

        # Check that the coadd values are the same.
        for i in range(len(keep)):
            np.testing.assert_allclose(keep["coadd_mean"][i], keep2["coadd_mean"][i])

    def test_get_coadds_and_filter_with_invalid(self):
        valid1 = [True] * self.image_count
        valid2 = [True] * self.image_count
        # Completely mess up some of the images.
        for i in [1, 3, 6, 7, 9]:
            self.ds.stack.get_single_image(i).sci[:, :] = 1000.0
            valid2[i] = False

        # Create the Results with nearly identical trajectories,
        # but different valid observations
        trj2 = Trajectory(self.trj.x, self.trj.y, self.trj.vx, self.trj.vy + 0.001, flux=250.0)
        keep = Results.from_trajectories([self.trj, trj2])
        keep.update_obs_valid(np.array([valid1, valid2]))

        # Make the stamps and check that there were saved.
        append_coadds(keep, self.ds.stack, ["mean"], 5)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertEqual(len(keep), 2)

        # Check that we get the coadd values if we use a Python image stack.
        keep2 = Results.from_trajectories([self.trj, trj2])
        keep2.update_obs_valid(np.array([valid1, valid2]))
        append_coadds(keep2, self.stack_py, ["mean"], 5)
        self.assertTrue("mean" in keep2.colnames)
        self.assertEqual(len(keep2), 2)

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
        append_coadds(keep, self.ds.stack, [], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertFalse("coadd_mean" in keep.colnames)
        self.assertFalse("coadd_median" in keep.colnames)
        self.assertFalse("coadd_weighted" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Adding "mean" and "median" does only those.
        append_coadds(keep, self.ds.stack, ["median", "mean"], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_median" in keep.colnames)
        self.assertFalse("coadd_weighted" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # We can add "weighted" later.
        append_coadds(keep, self.ds.stack, ["weighted"], 3)
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

        append_all_stamps(keep, self.ds.stack, 5)
        self.assertTrue("all_stamps" in keep.colnames)
        for i in range(len(keep)):
            stamps_array = keep["all_stamps"][i]
            self.assertEqual(stamps_array.shape[0], self.image_count)
            self.assertEqual(stamps_array.shape[1], 11)
            self.assertEqual(stamps_array.shape[2], 11)

        # Check that everything works if the results are empty.
        keep2 = Results.from_trajectories([])
        append_all_stamps(keep2, self.ds.stack, 5)
        self.assertTrue("all_stamps" in keep2.colnames)

        # Check that everything works with a Python image stack.
        keep3 = Results.from_trajectories(trj_list)
        append_all_stamps(keep3, self.stack_py, 5)
        self.assertTrue("all_stamps" in keep3.colnames)
        for i in range(len(keep3)):
            stamps_array = keep3["all_stamps"][i]
            self.assertEqual(stamps_array.shape[0], self.image_count)
            self.assertEqual(stamps_array.shape[1], 11)
            self.assertEqual(stamps_array.shape[2], 11)

    def test_filter_stamps_by_cnn(self):
        trj_list = [
            self.trj,
            Trajectory(self.trj.x, self.trj.y, 0.0, 0.0),
            Trajectory(self.trj.x + 2, self.trj.y + 2, self.trj.vx, self.trj.vy),
        ]
        keep = Results.from_trajectories(trj_list)
        append_coadds(keep, self.ds.stack, ["mean"], 3)

        filter_stamps_by_cnn(
            keep,
            self.model_path,
            coadd_type="mean",
            stamp_radius=3,
        )

        # the test model was trained on totally random data
        assert keep.table["cnn_class"].data[2] == False
        filtered_results = keep.filter_rows(keep.table["cnn_class"])
        assert len(filtered_results) == 2

        # assert that the function fails
        # when `Results` doesn't have needed
        # coadd column.
        with self.assertRaises(ValueError):
            filter_stamps_by_cnn(
                keep,
                self.model_path,
                coadd_type="median",
                stamp_radius=3,
            )


if __name__ == "__main__":
    unittest.main()
