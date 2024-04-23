import unittest

import numpy as np

from kbmod.batch_search import BatchSearchManager
from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import add_fake_object, make_fake_layered_image, FakeDataSet
from kbmod.run_search import SearchRunner
from kbmod.search import *
from kbmod.trajectory_generator import KBMODV1Search


class test_search(unittest.TestCase):
    def setUp(self):
        # test pass thresholds
        self.pixel_error = 1
        self.velocity_error = 0.1
        self.flux_error = 0.15

        # image properties
        self.img_count = 20
        self.dim_y = 80
        self.dim_x = 60
        self.noise_level = 4.0
        self.variance = self.noise_level**2
        self.p = PSF(1.0)

        # object properties
        self.object_flux = 250.0
        self.start_x = 17
        self.start_y = 12
        self.vxel = 21.0
        self.vyel = 16.0

        # create a Trajectory for the object
        self.trj = Trajectory(
            x=self.start_x,
            y=self.start_y,
            vx=self.vxel,
            vy=self.vyel,
        )

        # Add convenience array of all true bools for the stamp tests.
        self.all_valid = [True] * self.img_count

        # search parameters
        self.angle_steps = 150
        self.velocity_steps = 150
        self.min_angle = 0.0
        self.max_angle = 1.5
        self.min_vel = 5.0
        self.max_vel = 40.0

        # Select one pixel to mask in every other image.
        self.masked_y = 5
        self.masked_x = 6

        # create image set with single moving object
        self.imlist = []
        for i in range(self.img_count):
            time = i / self.img_count
            im = make_fake_layered_image(
                self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, seed=i
            )
            add_fake_object(
                im,
                self.trj.get_x_pos(time),
                self.trj.get_y_pos(time),
                self.object_flux,
                self.p,
            )

            # Mask a pixel in half the images.
            if i % 2 == 0:
                mask = im.get_mask()
                mask.set_pixel(self.masked_y, self.masked_x, 1)
                im.apply_mask(1)

            self.imlist.append(im)
        self.stack = ImageStack(self.imlist)
        self.search = StackSearch(self.stack)

        # Set the filtering parameters.
        self.params = StampParameters()
        self.params.radius = 5
        self.params.do_filtering = True
        self.params.stamp_type = StampType.STAMP_MEAN
        self.params.center_thresh = 0.03
        self.params.peak_offset_x = 1.5
        self.params.peak_offset_y = 1.5
        self.params.m01_limit = 1.0
        self.params.m10_limit = 1.0
        self.params.m11_limit = 2.0
        self.params.m02_limit = 35.5
        self.params.m20_limit = 35.5

        self.trj_gen = KBMODV1Search(
            self.velocity_steps,
            self.min_vel,
            self.max_vel,
            self.angle_steps,
            self.min_angle,
            self.max_angle,
        )

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

        # Check invalid settings
        self.assertRaises(RuntimeError, self.search.get_results, -1, 5)
        self.assertRaises(RuntimeError, self.search.get_results, 0, 0)

    def test_load_and_filter_results_lh(self):
        time_list = [i / self.img_count for i in range(self.img_count)]
        fake_ds = FakeDataSet(
            self.dim_x,
            self.dim_y,
            time_list,
            noise_level=1.0,
            psf_val=0.5,
            use_seed=True,
        )

        # Create fake result trajectories with given initial likelihoods. The 1st is
        # filtered by max likelihood. The 4th and 5th are filtered by min likelihood.
        trjs = [
            Trajectory(20, 20, 0, 0, 500.0, 9000.0, self.img_count),
            Trajectory(30, 30, 0, 0, 100.0, 100.0, self.img_count),
            Trajectory(40, 40, 0, 0, 50.0, 50.0, self.img_count),
            Trajectory(50, 50, 0, 0, 1.0, 2.0, self.img_count),
            Trajectory(60, 60, 0, 0, 1.0, 1.0, self.img_count),
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
            "max_lh": 1000.0,
            "num_cores": 1,
            "num_obs": 5,
            "sigmaG_lims": [25, 75],
        }
        config.set_multiple(overrides)

        runner = SearchRunner()
        results = runner.load_and_filter_results(search, config)

        # Only two of the middle results should pass the filtering.
        self.assertEqual(len(results), 2)
        self.assertEqual(results["y"][0], 30)
        self.assertEqual(results["y"][1], 40)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_evaluate_single_trajectory(self):
        test_trj = Trajectory(
            x=self.start_x,
            y=self.start_y,
            vx=self.vxel,
            vy=self.vyel,
        )
        self.search.evaluate_single_trajectory(test_trj)

        # We found a valid result.
        self.assertGreater(test_trj.obs_count, 0)
        self.assertGreater(test_trj.flux, 0.0)
        self.assertGreater(test_trj.lh, 0.0)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_search_linear_trajectory(self):
        test_trj = self.search.search_linear_trajectory(
            self.start_x,
            self.start_y,
            self.vxel,
            self.vyel,
        )

        # We found a valid result.
        self.assertGreater(test_trj.obs_count, 0)
        self.assertGreater(test_trj.flux, 0.0)
        self.assertGreater(test_trj.lh, 0.0)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results(self):
        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, int(self.img_count / 2))

        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_extended_bounds(self):
        self.search.set_start_bounds_x(-10, self.dim_x + 10)
        self.search.set_start_bounds_y(-10, self.dim_y + 10)

        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, int(self.img_count / 2))

        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_reduced_bounds(self):
        self.search.set_start_bounds_x(5, self.dim_x - 5)
        self.search.set_start_bounds_y(5, self.dim_y - 5)

        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, int(self.img_count / 2))

        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    def test_invalid_start_bounds(self):
        self.assertRaises(RuntimeError, self.search.set_start_bounds_x, 6, 5)
        self.assertRaises(RuntimeError, self.search.set_start_bounds_y, -1, -5)

    def test_set_sigmag_config(self):
        self.search.enable_gpu_sigmag_filter([0.25, 0.75], 0.5, 1.0)
        self.assertRaises(RuntimeError, self.search.enable_gpu_sigmag_filter, [0.25], 0.5, 1.0)
        self.assertRaises(RuntimeError, self.search.enable_gpu_sigmag_filter, [0.75, 0.25], 0.5, 1.0)
        self.assertRaises(RuntimeError, self.search.enable_gpu_sigmag_filter, [-0.01, 0.75], 0.5, 1.0)
        self.assertRaises(RuntimeError, self.search.enable_gpu_sigmag_filter, [0.75, 1.10], 0.5, 1.0)
        self.assertRaises(RuntimeError, self.search.enable_gpu_sigmag_filter, [0.25, 0.75], -0.5, 1.0)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_off_chip(self):
        trj = Trajectory(x=-3, y=12, vx=25.0, vy=10.0)

        # Create images with this fake object.
        imlist = []
        for i in range(self.img_count):
            time = i / self.img_count
            im = make_fake_layered_image(
                self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, seed=i
            )
            add_fake_object(
                im,
                trj.get_x_index(time),
                trj.get_y_index(time),
                self.object_flux,
                self.p,
            )
            imlist.append(im)
        stack = ImageStack(imlist)
        search = StackSearch(stack)

        # Do the extended search.
        search.set_start_bounds_x(-10, self.dim_x + 10)
        search.set_start_bounds_y(-10, self.dim_y + 10)
        candidates = [trj for trj in self.trj_gen]
        search.search_all(candidates, int(self.img_count / 2))

        # Check the results.
        results = search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, trj.x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, trj.y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / trj.vx, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / trj.vy, 1, delta=self.velocity_error)

    def test_sci_viz_stamps(self):
        sci_stamps = StampCreator.get_stamps(self.search.get_imagestack(), self.trj, 2)
        self.assertEqual(len(sci_stamps), self.img_count)

        times = self.stack.build_zeroed_times()
        for i in range(self.img_count):
            self.assertEqual(sci_stamps[i].width, 5)
            self.assertEqual(sci_stamps[i].height, 5)

            # Compute the interpolated pixel value at the projected location.
            x = self.trj.get_x_index(times[i])
            y = self.trj.get_y_index(times[i])
            pixVal = self.imlist[i].get_science().get_pixel(y, x)
            if not pixel_value_valid(pixVal):
                pivVal = 0.0

            # Check that pixel value of the projected location equals
            # that of the base image.
            self.assertAlmostEqual(sci_stamps[i].get_pixel(2, 2), pixVal, delta=0.001)

    def test_stacked_sci(self):
        # Compute the stacked science from a single Trajectory.
        sci = StampCreator.get_summed_stamp(self.search.get_imagestack(), self.trj, 2, [])
        self.assertEqual(sci.width, 5)
        self.assertEqual(sci.height, 5)

        # Compute the true stacked pixel for the middle of the track.
        times = self.stack.build_zeroed_times()
        sum_middle = 0.0
        for i in range(self.img_count):
            t = times[i]
            x = self.trj.get_x_index(t)
            y = self.trj.get_y_index(t)
            pixVal = self.imlist[i].get_science().get_pixel(y, x)
            if not pixel_value_valid(pixVal):
                pivVal = 0.0
            sum_middle = sum_middle + pixVal

        # Check that the two different approaches for stack science
        # match the true value.
        self.assertAlmostEqual(sci.get_pixel(2, 2), sum_middle, delta=0.001)

    def test_median_stamps_trj(self):
        # Compute the stacked science from two trajectories (one with bad points).
        goodIdx = [[1] * self.img_count for _ in range(2)]
        goodIdx[1][1] = 0
        goodIdx[1][5] = 0
        goodIdx[1][9] = 0

        medianStamps0 = StampCreator.get_median_stamp(self.search.get_imagestack(), self.trj, 2, goodIdx[0])
        self.assertEqual(medianStamps0.width, 5)
        self.assertEqual(medianStamps0.height, 5)

        medianStamps1 = StampCreator.get_median_stamp(self.search.get_imagestack(), self.trj, 2, goodIdx[1])
        self.assertEqual(medianStamps1.width, 5)
        self.assertEqual(medianStamps1.height, 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.build_zeroed_times()
        pix_values0 = []
        pix_values1 = []
        for i in range(self.img_count):
            t = times[i]
            x = self.trj.get_x_index(t)
            y = self.trj.get_y_index(t)
            pixVal = self.imlist[i].get_science().get_pixel(y, x)
            if pixel_value_valid(pixVal) and goodIdx[0][i] == 1:
                pix_values0.append(pixVal)
            if pixel_value_valid(pixVal) and goodIdx[1][i] == 1:
                pix_values1.append(pixVal)
        self.assertEqual(len(pix_values0), self.img_count)
        self.assertEqual(len(pix_values1), self.img_count - 3)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values0), medianStamps0.get_pixel(2, 2), delta=1e-5)
        self.assertAlmostEqual(np.median(pix_values1), medianStamps1.get_pixel(2, 2), delta=1e-5)

    def test_median_stamps_no_data(self):
        # Create a Trajectory that goes through the masked pixels.
        trj = Trajectory(x=self.masked_x, y=self.masked_y, vx=0.0, vy=0.0)

        # Compute the stacked science from a single Trajectory.
        medianStamp = StampCreator.get_median_stamp(self.search.get_imagestack(), trj, 2, self.all_valid)
        self.assertEqual(medianStamp.width, 5)
        self.assertEqual(medianStamp.height, 5)

        # Compute the true median pixel for the middle of the track.
        pix_values = []
        for i in range(self.img_count):
            pixVal = self.imlist[i].get_science().get_pixel(self.masked_y, self.masked_x)
            if pixel_value_valid(pixVal):
                pix_values.append(pixVal)
        self.assertEqual(len(pix_values), self.img_count / 2)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values), medianStamp.get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_trj(self):
        # Compute the stacked science from two trajectories (one with bad points).
        goodIdx = [[1] * self.img_count for _ in range(2)]
        goodIdx[1][1] = 0
        goodIdx[1][5] = 0
        goodIdx[1][9] = 0

        meanStamp0 = StampCreator.get_mean_stamp(self.search.get_imagestack(), self.trj, 2, goodIdx[0])
        self.assertEqual(meanStamp0.width, 5)
        self.assertEqual(meanStamp0.height, 5)

        meanStamp1 = StampCreator.get_mean_stamp(self.search.get_imagestack(), self.trj, 2, goodIdx[1])
        self.assertEqual(meanStamp1.width, 5)
        self.assertEqual(meanStamp1.height, 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.build_zeroed_times()
        pix_sum0 = 0.0
        pix_sum1 = 0.0
        pix_count0 = 0.0
        pix_count1 = 0.0
        for i in range(self.img_count):
            t = times[i]
            x = self.trj.get_x_index(t)
            y = self.trj.get_y_index(t)
            pixVal = self.imlist[i].get_science().get_pixel(y, x)
            if pixel_value_valid(pixVal) and goodIdx[0][i] == 1:
                pix_sum0 += pixVal
                pix_count0 += 1
            if pixel_value_valid(pixVal) and goodIdx[1][i] == 1:
                pix_sum1 += pixVal
                pix_count1 += 1
        self.assertEqual(pix_count0, self.img_count)
        self.assertEqual(pix_count1, self.img_count - 3)

        # Check that we get the correct answer.
        self.assertAlmostEqual(pix_sum0 / pix_count0, meanStamp0.get_pixel(2, 2), delta=1e-5)
        self.assertAlmostEqual(pix_sum1 / pix_count1, meanStamp1.get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_no_data(self):
        # Create a Trajectory that goes through the masked pixels.
        trj = Trajectory(x=self.masked_x, y=self.masked_y, vx=0.0, vy=0.0)

        # Compute the stacked science from a single Trajectory
        meanStamp = StampCreator.get_mean_stamp(self.search.get_imagestack(), trj, 2, self.all_valid)
        self.assertEqual(meanStamp.width, 5)
        self.assertEqual(meanStamp.height, 5)

        # Compute the true median pixel for the middle of the track.
        pix_sum = 0.0
        pix_count = 0.0
        for i in range(self.img_count):
            pixVal = self.imlist[i].get_science().get_pixel(self.masked_y, self.masked_x)
            if pixel_value_valid(pixVal):
                pix_sum += pixVal
                pix_count += 1.0
        self.assertEqual(pix_count, self.img_count / 2.0)

        # Check that we get the correct answer.
        self.assertAlmostEqual(pix_sum / pix_count, meanStamp.get_pixel(2, 2), delta=1e-5)

    def test_filter_stamp(self):
        stamp_width = 2 * self.params.radius + 1

        # Test a stamp with nothing in it.
        stamp = RawImage(stamp_width, stamp_width)
        stamp.set_all(1.0)
        self.assertTrue(StampCreator.filter_stamp(stamp, self.params))

        # Test a stamp with a bright spot in the center.
        stamp.set_pixel(5, 5, 100.0)
        self.assertFalse(StampCreator.filter_stamp(stamp, self.params))

        # A little noise around the pixel does not hurt as long as the shape is
        # roughly Gaussian.
        stamp.set_pixel(4, 5, 15.0)
        stamp.set_pixel(5, 4, 10.0)
        stamp.set_pixel(6, 5, 10.0)
        stamp.set_pixel(5, 6, 20.0)
        self.assertFalse(StampCreator.filter_stamp(stamp, self.params))

        # A bright peak far from the center is bad.
        stamp.set_pixel(1, 1, 500.0)
        self.assertTrue(StampCreator.filter_stamp(stamp, self.params))
        stamp.set_pixel(1, 1, 1.0)

        # A non-Gaussian bright spot is also bad. Blur to the -x direction.
        stamp.set_pixel(4, 5, 50.0)
        stamp.set_pixel(3, 5, 50.0)
        stamp.set_pixel(2, 5, 60.0)
        stamp.set_pixel(4, 4, 45.0)
        stamp.set_pixel(3, 4, 45.0)
        stamp.set_pixel(2, 4, 55.0)
        stamp.set_pixel(4, 6, 55.0)
        stamp.set_pixel(3, 6, 55.0)
        stamp.set_pixel(2, 6, 65.0)
        self.assertTrue(StampCreator.filter_stamp(stamp, self.params))

        # A very dim peak at the center is invalid.
        stamp.set_all(1.0)
        stamp.set_pixel(5, 5, 1.0001)
        self.assertTrue(StampCreator.filter_stamp(stamp, self.params))

        # A slightly offset peak of sufficient brightness is okay.
        stamp.set_pixel(5, 5, 15.0)
        stamp.set_pixel(4, 5, 20.0)
        self.assertFalse(StampCreator.filter_stamp(stamp, self.params))

    def test_coadd_cpu_simple(self):
        # Create an image set with three images.
        imlist = []
        for i in range(3):
            # Set the first column to i, the middle column to i + 1, and the last column to 0.5.
            # In the very first image, make pixel (0, 2) NaN
            sci = np.array([[i, i + 1, 0.5], [i, i + 1, 0.5], [i, i + 1, 0.5]])
            if i == 0:
                sci[0][2] = np.nan

            var = np.full((3, 3), 0.1)

            # Mask out the column's first pixel twice and second pixel once.
            msk = np.zeros((3, 3))
            if i == 0:
                msk[0][1] = 1
                msk[1][1] = 1
            if i == 1:
                msk[0][1] = 1

            img = LayeredImage(
                RawImage(sci.astype(np.float32)),
                RawImage(var.astype(np.float32)),
                RawImage(msk.astype(np.float32)),
                self.p,
            )
            img.apply_mask(1)
            imlist.append(img)
        stack = ImageStack(imlist)
        search = StackSearch(stack)
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Basic Stamp parameters.
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False

        # Test summed.
        params.stamp_type = StampType.STAMP_SUM
        stamps = StampCreator.get_coadded_stamps(search.get_imagestack(), [trj], [all_valid], params, False)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 0), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 0), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 0), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 5.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 6.0)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 2), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 2), 1.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 2), 1.5)

        # Test mean.
        params.stamp_type = StampType.STAMP_MEAN
        stamps = StampCreator.get_coadded_stamps(search.get_imagestack(), [trj], [all_valid], params, False)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 0), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 0), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 0), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 2), 0.5)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 2), 0.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 2), 0.5)

        # Test median.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps = StampCreator.get_coadded_stamps(search.get_imagestack(), [trj], [all_valid], params, False)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 0), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 0), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 0), 1.0)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 2), 0.5)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 2), 0.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 2), 0.5)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu_simple(self):
        # Create an image set with three images.
        imlist = []
        for i in range(3):
            time = i
            im = make_fake_layered_image(3, 3, 0.1, 0.01, i, self.p, seed=i)

            # Overwrite the middle row to be i + 1.
            sci = im.get_science()
            for x in range(3):
                sci.set_pixel(x, 1, i + 1)

            # Mask out the row's first pixel twice and second pixel once.
            mask = im.get_mask()
            if i == 0:
                mask.set_pixel(0, 1, 1)
                mask.set_pixel(1, 1, 1)
            if i == 1:
                mask.set_pixel(0, 1, 1)
            im.apply_mask(1)

            imlist.append(im)
        stack = ImageStack(imlist)
        search = StackSearch(stack)
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Basic Stamp parameters.
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False

        # Test summed.
        params.stamp_type = StampType.STAMP_SUM
        stamps = StampCreator.get_coadded_stamps(search.get_imagestack(), [trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 5.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 6.0)

        # Test mean.
        params.stamp_type = StampType.STAMP_MEAN
        stamps = StampCreator.get_coadded_stamps(search.get_imagestack(), [trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

        # Test median.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps = StampCreator.get_coadded_stamps(search.get_imagestack(), [trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

    def test_coadd_cpu(self):
        params = StampParameters()
        params.radius = 3
        params.do_filtering = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj], [self.all_valid], params, False
        )
        self.assertEqual(summedStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj], [self.all_valid], params, False
        )
        self.assertEqual(meanStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEDIAN
        medianStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj], [self.all_valid], params, False
        )
        self.assertEqual(medianStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(medianStamps[0].height, 2 * params.radius + 1)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.build_zeroed_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                pix_sum = 0.0
                pix_count = 0.0
                pix_vals = []
                for i in range(self.img_count):
                    t = times[i]
                    x = self.trj.get_x_index(t) + x_offset
                    y = self.trj.get_y_index(t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(y, x)
                    if pixel_value_valid(pixVal):
                        pix_sum += pixVal
                        pix_count += 1.0
                        pix_vals.append(pixVal)

                # Check that we get the correct answers.
                self.assertAlmostEqual(pix_sum, summedStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3)
                self.assertAlmostEqual(
                    pix_sum / pix_count, meanStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3
                )
                self.assertAlmostEqual(
                    np.median(pix_vals), medianStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3
                )

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu(self):
        params = StampParameters()
        params.radius = 3
        params.do_filtering = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj], [self.all_valid], params, True
        )
        self.assertEqual(summedStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj], [self.all_valid], params, True
        )
        self.assertEqual(meanStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEDIAN
        medianStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj], [self.all_valid], params, True
        )
        self.assertEqual(medianStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(medianStamps[0].height, 2 * params.radius + 1)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.build_zeroed_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                pix_sum = 0.0
                pix_count = 0.0
                pix_vals = []
                for i in range(self.img_count):
                    t = times[i]
                    x = self.trj.get_x_index(t) + x_offset
                    y = self.trj.get_y_index(t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(y, x)
                    if pixel_value_valid(pixVal):
                        pix_sum += pixVal
                        pix_count += 1.0
                        pix_vals.append(pixVal)

                # Check that we get the correct answers.
                self.assertAlmostEqual(pix_sum, summedStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3)
                self.assertAlmostEqual(
                    pix_sum / pix_count, meanStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3
                )
                self.assertAlmostEqual(
                    np.median(pix_vals), medianStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3
                )

    def test_coadd_cpu_use_inds(self):
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        inds = [[True] * self.img_count, [True] * self.img_count]
        inds[0][5] = False
        inds[1][3] = False
        inds[1][6] = False
        inds[1][7] = False
        inds[1][11] = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        meanStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj, self.trj], inds, params, False
        )

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.build_zeroed_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                sum_0 = 0.0
                sum_1 = 0.0
                count_0 = 0.0
                count_1 = 0.0
                for i in range(self.img_count):
                    t = times[i]
                    x = self.trj.get_x_index(t) + x_offset
                    y = self.trj.get_y_index(t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(y, x)

                    if pixel_value_valid(pixVal) and inds[0][i] > 0:
                        sum_0 += pixVal
                        count_0 += 1.0

                    if pixel_value_valid(pixVal) and inds[1][i] > 0:
                        sum_1 += pixVal
                        count_1 += 1.0

                # Check that we get the correct answers.
                self.assertAlmostEqual(count_0, 19.0)
                self.assertAlmostEqual(count_1, 16.0)
                self.assertAlmostEqual(sum_0 / count_0, meanStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3)
                self.assertAlmostEqual(sum_1 / count_1, meanStamps[1].get_pixel(stamp_y, stamp_x), delta=1e-3)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu_use_inds(self):
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        inds = [[True] * self.img_count, [True] * self.img_count]
        inds[0][5] = False
        inds[1][3] = False
        inds[1][6] = False
        inds[1][7] = False
        inds[1][11] = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        meanStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj, self.trj], inds, params, True
        )

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.build_zeroed_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                sum_0 = 0.0
                sum_1 = 0.0
                count_0 = 0.0
                count_1 = 0.0
                for i in range(self.img_count):
                    t = times[i]
                    x = self.trj.get_x_index(t) + x_offset
                    y = self.trj.get_y_index(t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(y, x)

                    if pixel_value_valid(pixVal) and inds[0][i] > 0:
                        sum_0 += pixVal
                        count_0 += 1.0

                    if pixel_value_valid(pixVal) and inds[1][i] > 0:
                        sum_1 += pixVal
                        count_1 += 1.0

                # Check that we get the correct answers.
                self.assertAlmostEqual(count_0, 19.0)
                self.assertAlmostEqual(count_1, 16.0)
                self.assertAlmostEqual(sum_0 / count_0, meanStamps[0].get_pixel(stamp_y, stamp_x), delta=1e-3)
                self.assertAlmostEqual(sum_1 / count_1, meanStamps[1].get_pixel(stamp_y, stamp_x), delta=1e-3)

    def test_coadd_filter_cpu(self):
        # Create a second Trajectory that isn't any good.
        trj2 = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Create a third Trajectory that is close to good, but offset.
        trj3 = Trajectory(
            x=self.trj.x + 2,
            y=self.trj.y + 2,
            vx=self.trj.vx,
            vy=self.trj.vy,
        )

        # Create a fourth Trajectory that is close enough
        trj4 = Trajectory(
            x=self.trj.x + 1,
            y=self.trj.y + 1,
            vx=self.trj.vx,
            vy=self.trj.vy,
        )

        # Compute the stacked science from a single Trajectory.
        all_valid_vect = [(self.all_valid) for i in range(4)]
        meanStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj, trj2, trj3, trj4], all_valid_vect, self.params, True
        )

        # The first and last are unfiltered
        self.assertEqual(meanStamps[0].width, 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[0].height, 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].width, 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].height, 2 * self.params.radius + 1)

        # The second and third are filtered.
        self.assertEqual(meanStamps[1].width, 1)
        self.assertEqual(meanStamps[1].height, 1)
        self.assertEqual(meanStamps[2].width, 1)
        self.assertEqual(meanStamps[2].height, 1)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_filter_gpu(self):
        # Create a second Trajectory that isn't any good.
        trj2 = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Create a third Trajectory that is close to good, but offset.
        trj3 = Trajectory(
            x=self.trj.x + 2,
            y=self.trj.y + 2,
            vx=self.trj.vx,
            vy=self.trj.vy,
        )

        # Create a fourth Trajectory that is close enough
        trj4 = Trajectory(
            x=self.trj.x + 1,
            y=self.trj.y + 1,
            vx=self.trj.vx,
            vy=self.trj.vy,
        )

        # Compute the stacked science from a single Trajectory.
        all_valid_vect = [(self.all_valid) for i in range(4)]
        meanStamps = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), [self.trj, trj2, trj3, trj4], all_valid_vect, self.params, True
        )

        # The first and last are unfiltered
        self.assertEqual(meanStamps[0].width, 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[0].height, 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].width, 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].height, 2 * self.params.radius + 1)

        # The second and third are filtered.
        self.assertEqual(meanStamps[1].width, 1)
        self.assertEqual(meanStamps[1].height, 1)
        self.assertEqual(meanStamps[2].width, 1)
        self.assertEqual(meanStamps[2].height, 1)

    @staticmethod
    def result_hash(res):
        return hash((res.x, res.y, res.vx, res.vy, res.lh, res.obs_count))

    def test_search_batch(self):
        width = 50
        height = 50
        results_per_pixel = 8
        min_observations = 2

        # Simple average PSF
        psf_data = np.zeros((5, 5), dtype=np.single)
        psf_data[1:4, 1:4] = 0.1111111
        p = PSF(psf_data)

        # Create a stack with 10 20x20 images with random noise and times ranging from 0 to 1
        count = 10
        imlist = [make_fake_layered_image(width, height, 5.0, 25.0, n / count, p) for n in range(count)]
        stack = ImageStack(imlist)

        for i in range(count):
            im = stack.get_single_image(i)
            time = stack.get_zeroed_time(i)
            add_fake_object(im, 5.0 + (time * 8.0), 35.0 + (time * 0.0), 25000.0)

        search = StackSearch(stack)

        # Sample generator
        gen = KBMODV1Search(
            10, 5, 15, 10, -0.1, 0.1
        )  # velocity_steps, min_vel, max_vel, angle_steps, min_ang, max_ang,
        candidates = [trj for trj in gen]

        # Peform complete in-memory search
        search.search_all(candidates, min_observations)
        total_results = width * height * results_per_pixel
        # Need to filter as the fields are undefined otherwise
        results = [
            result
            for result in search.get_results(0, total_results)
            if result.lh > -1 and result.obs_count >= min_observations
        ]

        with BatchSearchManager(StackSearch(stack), candidates, min_observations) as batch_search:
            batch_results = []
            for i in range(0, width, 5):
                batch_search.set_start_bounds_x(i, i + 5)
                for j in range(0, height, 5):
                    batch_search.set_start_bounds_y(j, j + 5)
                    batch_results.extend(batch_search.search_single_batch())

            # Need to filter as the fields are undefined otherwise
            batch_results = [
                result for result in batch_results if result.lh > -1 and result.obs_count >= min_observations
            ]

            # Check that the results are the same.
            results_hash_set = {test_search.result_hash(result) for result in results}
            batch_results_hash_set = {test_search.result_hash(result) for result in batch_results}

            for res_hash in results_hash_set:
                self.assertTrue(res_hash in batch_results_hash_set)


if __name__ == "__main__":
    unittest.main()
