import unittest

import numpy as np

from kbmod.fake_data_creator import add_fake_object
from kbmod.search import *


class test_search(unittest.TestCase):
    def setUp(self):
        # test pass thresholds
        self.pixel_error = 0
        self.velocity_error = 0.05
        self.flux_error = 0.15

        # image properties
        self.imCount = 20
        self.dim_x = 80
        self.dim_y = 60
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
        self.trj = Trajectory()
        self.trj.x = self.start_x
        self.trj.y = self.start_y
        self.trj.vx = self.vxel
        self.trj.vy = self.vyel

        # Add convenience array of all true bools for the stamp tests.
        self.all_valid = [True] * self.imCount

        # search parameters
        self.angle_steps = 150
        self.velocity_steps = 150
        self.min_angle = 0.0
        self.max_angle = 1.5
        self.min_vel = 5.0
        self.max_vel = 40.0

        # Select one pixel to mask in every other image.
        self.masked_x = 5
        self.masked_y = 6

        # create image set with single moving object
        self.imlist = []
        for i in range(self.imCount):
            time = i / self.imCount
            im = LayeredImage(
                str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i
            )
            add_fake_object(
                im,
                self.start_x + time * self.vxel + 0.5,
                self.start_y + time * self.vyel + 0.5,
                self.object_flux,
                self.p,
            )

            # Mask a pixel in half the images.
            if i % 2 == 0:
                mask = im.get_mask()
                mask.set_pixel(self.masked_x, self.masked_y, 1)
                im.set_mask(mask)
                im.apply_mask_flags(1, [])

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
        self.params.m01_limit = 0.6
        self.params.m10_limit = 0.6
        self.params.m11_limit = 2.0
        self.params.m02_limit = 35.5
        self.params.m20_limit = 35.5

    def test_psiphi(self):
        p = PSF(0.00001)

        # Image1 has a single object.
        image1 = LayeredImage("test1", 5, 10, 2.0, 4.0, 1.0, p)
        add_fake_object(image1, 3.5, 2.5, 400.0, p)

        # Image2 has a single object and a masked pixel.
        image2 = LayeredImage("test2", 5, 10, 2.0, 4.0, 2.0, p)
        add_fake_object(image2, 2.5, 4.5, 400.0, p)

        mask = image2.get_mask()
        mask.set_pixel(4, 9, 1)
        image2.set_mask(mask)
        image2.apply_mask_flags(1, [])

        # Create a stack from the two objects.
        stack = ImageStack([image1, image2])
        search = StackSearch(stack)

        # Generate psi and phi.
        search.prepare_psi_phi()
        psi = search.get_psi_images()
        phi = search.get_phi_images()

        # Test phi and psi for image1.
        sci = image1.get_science()
        var = image1.get_variance()
        for x in range(5):
            for y in range(10):
                self.assertAlmostEqual(
                    psi[0].get_pixel(x, y), sci.get_pixel(x, y) / var.get_pixel(x, y), delta=1e-6
                )
                self.assertAlmostEqual(phi[0].get_pixel(x, y), 1.0 / var.get_pixel(x, y), delta=1e-6)

        # Test phi and psi for image2.
        sci = image2.get_science()
        var = image2.get_variance()
        for x in range(5):
            for y in range(10):
                if x == 4 and y == 9:
                    self.assertFalse(psi[1].pixel_has_data(x, y))
                    self.assertFalse(phi[1].pixel_has_data(x, y))
                else:
                    self.assertAlmostEqual(
                        psi[1].get_pixel(x, y), sci.get_pixel(x, y) / var.get_pixel(x, y), delta=1e-6
                    )
                    self.assertAlmostEqual(phi[1].get_pixel(x, y), 1.0 / var.get_pixel(x, y), delta=1e-6)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results(self):
        self.search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

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

        self.search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

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

        self.search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_off_chip(self):
        trj = Trajectory()
        trj.x = -3
        trj.y = 12
        trj.vx = 25.0
        trj.vy = 10.0

        # Create images with this fake object.
        imlist = []
        for i in range(self.imCount):
            time = i / self.imCount
            im = LayeredImage(
                str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i
            )
            add_fake_object(
                im,
                trj.x + time * trj.vx + 0.5,
                trj.y + time * trj.vy + 0.5,
                self.object_flux,
                self.p,
            )
            imlist.append(im)
        stack = ImageStack(imlist)
        search = StackSearch(stack)

        # Do the extended search.
        search.set_start_bounds_x(-10, self.dim_x + 10)
        search.set_start_bounds_y(-10, self.dim_y + 10)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

        # Check the results.
        results = search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, trj.x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, trj.y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / trj.vx, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / trj.vy, 1, delta=self.velocity_error)

    def test_sci_viz_stamps(self):
        sci_stamps = self.search.get_stamps(self.trj, 2)
        self.assertEqual(len(sci_stamps), self.imCount)

        times = self.stack.get_times()
        for i in range(self.imCount):
            self.assertEqual(sci_stamps[i].get_width(), 5)
            self.assertEqual(sci_stamps[i].get_height(), 5)

            # Compute the interpolated pixel value at the projected location.
            t = times[i]
            x = float(self.trj.x) + self.trj.vx * t
            y = float(self.trj.y) + self.trj.vy * t
            pixVal = self.imlist[i].get_science().get_pixel_interp(x, y)
            if pixVal == KB_NO_DATA:
                pivVal = 0.0

            # Check that pixel value of the projected location equals
            # that of the base image.
            self.assertAlmostEqual(sci_stamps[i].get_pixel(2, 2), pixVal, delta=0.001)

    def test_stacked_sci(self):
        # Compute the stacked science from a single Trajectory.
        sci = self.search.get_summed_stamp(self.trj, 2, [])
        self.assertEqual(sci.get_width(), 5)
        self.assertEqual(sci.get_height(), 5)

        # Compute the true stacked pixel for the middle of the track.
        times = self.stack.get_times()
        sum_middle = 0.0
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.vx * t)
            y = int(self.trj.y + self.trj.vy * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal == KB_NO_DATA:
                pivVal = 0.0
            sum_middle = sum_middle + pixVal

        # Check that the two different approaches for stack science
        # match the true value.
        self.assertAlmostEqual(sci.get_pixel(2, 2), sum_middle, delta=0.001)

    def test_median_stamps_trj(self):
        # Compute the stacked science from two trajectories (one with bad points).
        goodIdx = [[1] * self.imCount for _ in range(2)]
        goodIdx[1][1] = 0
        goodIdx[1][5] = 0
        goodIdx[1][9] = 0

        medianStamps0 = self.search.get_median_stamp(self.trj, 2, goodIdx[0])
        self.assertEqual(medianStamps0.get_width(), 5)
        self.assertEqual(medianStamps0.get_height(), 5)

        medianStamps1 = self.search.get_median_stamp(self.trj, 2, goodIdx[1])
        self.assertEqual(medianStamps1.get_width(), 5)
        self.assertEqual(medianStamps1.get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.get_times()
        pix_values0 = []
        pix_values1 = []
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.vx * t)
            y = int(self.trj.y + self.trj.vy * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal != KB_NO_DATA and goodIdx[0][i] == 1:
                pix_values0.append(pixVal)
            if pixVal != KB_NO_DATA and goodIdx[1][i] == 1:
                pix_values1.append(pixVal)
        self.assertEqual(len(pix_values0), self.imCount)
        self.assertEqual(len(pix_values1), self.imCount - 3)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values0), medianStamps0.get_pixel(2, 2), delta=1e-5)
        self.assertAlmostEqual(np.median(pix_values1), medianStamps1.get_pixel(2, 2), delta=1e-5)

    def test_median_stamps_no_data(self):
        # Create a Trajectory that goes through the masked pixels.
        trj = Trajectory()
        trj.x = self.masked_x
        trj.y = self.masked_y
        trj.vx = 0
        trj.vy = 0

        # Compute the stacked science from a single Trajectory.
        medianStamp = self.search.get_median_stamp(trj, 2, self.all_valid)
        self.assertEqual(medianStamp.get_width(), 5)
        self.assertEqual(medianStamp.get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        pix_values = []
        for i in range(self.imCount):
            pixVal = self.imlist[i].get_science().get_pixel(self.masked_x, self.masked_y)
            if pixVal != KB_NO_DATA:
                pix_values.append(pixVal)
        self.assertEqual(len(pix_values), self.imCount / 2)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values), medianStamp.get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_trj(self):
        # Compute the stacked science from two trajectories (one with bad points).
        goodIdx = [[1] * self.imCount for _ in range(2)]
        goodIdx[1][1] = 0
        goodIdx[1][5] = 0
        goodIdx[1][9] = 0

        meanStamp0 = self.search.get_mean_stamp(self.trj, 2, goodIdx[0])
        self.assertEqual(meanStamp0.get_width(), 5)
        self.assertEqual(meanStamp0.get_height(), 5)

        meanStamp1 = self.search.get_mean_stamp(self.trj, 2, goodIdx[1])
        self.assertEqual(meanStamp1.get_width(), 5)
        self.assertEqual(meanStamp1.get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.get_times()
        pix_sum0 = 0.0
        pix_sum1 = 0.0
        pix_count0 = 0.0
        pix_count1 = 0.0
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.vx * t)
            y = int(self.trj.y + self.trj.vy * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal != KB_NO_DATA and goodIdx[0][i] == 1:
                pix_sum0 += pixVal
                pix_count0 += 1
            if pixVal != KB_NO_DATA and goodIdx[1][i] == 1:
                pix_sum1 += pixVal
                pix_count1 += 1
        self.assertEqual(pix_count0, self.imCount)
        self.assertEqual(pix_count1, self.imCount - 3)

        # Check that we get the correct answer.
        self.assertAlmostEqual(pix_sum0 / pix_count0, meanStamp0.get_pixel(2, 2), delta=1e-5)
        self.assertAlmostEqual(pix_sum1 / pix_count1, meanStamp1.get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_no_data(self):
        # Create a Trajectory that goes through the masked pixels.
        trj = Trajectory()
        trj.x = self.masked_x
        trj.y = self.masked_y
        trj.vx = 0
        trj.vy = 0

        # Compute the stacked science from a single Trajectory.
        meanStamp = self.search.get_mean_stamp(trj, 2, self.all_valid)
        self.assertEqual(meanStamp.get_width(), 5)
        self.assertEqual(meanStamp.get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        pix_sum = 0.0
        pix_count = 0.0
        for i in range(self.imCount):
            pixVal = self.imlist[i].get_science().get_pixel(self.masked_x, self.masked_y)
            if pixVal != KB_NO_DATA:
                pix_sum += pixVal
                pix_count += 1.0
        self.assertEqual(pix_count, self.imCount / 2.0)

        # Check that we get the correct answer.
        self.assertAlmostEqual(pix_sum / pix_count, meanStamp.get_pixel(2, 2), delta=1e-5)

    def test_filter_stamp(self):
        stamp_width = 2 * self.params.radius + 1

        # Test a stamp with nothing in it.
        stamp = RawImage(stamp_width, stamp_width)
        stamp.set_all(1.0)
        self.assertTrue(self.search.filter_stamp(stamp, self.params))

        # Test a stamp with a bright spot in the center.
        stamp.set_pixel(5, 5, 100.0)
        self.assertFalse(self.search.filter_stamp(stamp, self.params))

        # A little noise around the pixel does not hurt as long as the shape is
        # roughly Gaussian.
        stamp.set_pixel(4, 5, 15.0)
        stamp.set_pixel(5, 4, 10.0)
        stamp.set_pixel(6, 5, 10.0)
        stamp.set_pixel(5, 6, 20.0)
        self.assertFalse(self.search.filter_stamp(stamp, self.params))

        # A bright peak far from the center is bad.
        stamp.set_pixel(1, 1, 500.0)
        self.assertTrue(self.search.filter_stamp(stamp, self.params))
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
        self.assertTrue(self.search.filter_stamp(stamp, self.params))

        # A very dim peak at the center is invalid.
        stamp.set_all(1.0)
        stamp.set_pixel(5, 5, 1.0001)
        self.assertTrue(self.search.filter_stamp(stamp, self.params))

        # A slightly offset peak of sufficient brightness is okay.
        stamp.set_pixel(5, 5, 15.0)
        stamp.set_pixel(4, 5, 20.0)
        self.assertFalse(self.search.filter_stamp(stamp, self.params))

    def test_coadd_cpu_simple(self):
        # Create an image set with three images.
        imlist = []
        for i in range(3):
            time = i
            im = LayeredImage(str(i), 3, 3, 0.1, 0.01, i, self.p, i)

            # Overwrite the middle row to be i + 1.
            sci = im.get_science()
            for x in range(3):
                sci.set_pixel(x, 1, i + 1)
            im.set_science(sci)

            # Mask out the row's first pixel twice and second pixel once.
            mask = im.get_mask()
            if i == 0:
                mask.set_pixel(0, 1, 1)
                mask.set_pixel(1, 1, 1)
            if i == 1:
                mask.set_pixel(0, 1, 1)
            im.set_mask(mask)
            im.apply_mask_flags(1, [])

            imlist.append(im)
        stack = ImageStack(imlist)
        search = StackSearch(stack)
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory()
        trj.x = 1
        trj.y = 1
        trj.vx = 0
        trj.vy = 0

        # Basic Stamp parameters.
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False

        # Test summed.
        params.stamp_type = StampType.STAMP_SUM
        stamps = search.get_coadded_stamps([trj], [all_valid], params, False)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 5.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 6.0)

        # Test mean.
        params.stamp_type = StampType.STAMP_MEAN
        stamps = search.get_coadded_stamps([trj], [all_valid], params, False)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

        # Test median.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps = search.get_coadded_stamps([trj], [all_valid], params, False)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu_simple(self):
        # Create an image set with three images.
        imlist = []
        for i in range(3):
            time = i
            im = LayeredImage(str(i), 3, 3, 0.1, 0.01, i, self.p, i)

            # Overwrite the middle row to be i + 1.
            sci = im.get_science()
            for x in range(3):
                sci.set_pixel(x, 1, i + 1)
            im.set_science(sci)

            # Mask out the row's first pixel twice and second pixel once.
            mask = im.get_mask()
            if i == 0:
                mask.set_pixel(0, 1, 1)
                mask.set_pixel(1, 1, 1)
            if i == 1:
                mask.set_pixel(0, 1, 1)
            im.set_mask(mask)
            im.apply_mask_flags(1, [])

            imlist.append(im)
        stack = ImageStack(imlist)
        search = StackSearch(stack)
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory()
        trj.x = 1
        trj.y = 1
        trj.vx = 0
        trj.vy = 0

        # Basic Stamp parameters.
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False

        # Test summed.
        params.stamp_type = StampType.STAMP_SUM
        stamps = search.get_coadded_stamps([trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 5.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 6.0)

        # Test mean.
        params.stamp_type = StampType.STAMP_MEAN
        stamps = search.get_coadded_stamps([trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

        # Test median.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps = search.get_coadded_stamps([trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

    def test_coadd_cpu(self):
        params = StampParameters()
        params.radius = 3
        params.do_filtering = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = self.search.get_coadded_stamps([self.trj], [self.all_valid], params, False)
        self.assertEqual(summedStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].get_height(), 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = self.search.get_coadded_stamps([self.trj], [self.all_valid], params, False)
        self.assertEqual(meanStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].get_height(), 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEDIAN
        medianStamps = self.search.get_coadded_stamps([self.trj], [self.all_valid], params, False)
        self.assertEqual(medianStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(medianStamps[0].get_height(), 2 * params.radius + 1)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.get_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                pix_sum = 0.0
                pix_count = 0.0
                pix_vals = []
                for i in range(self.imCount):
                    t = times[i]
                    x = int(self.trj.x + self.trj.vx * t) + x_offset
                    y = int(self.trj.y + self.trj.vy * t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(x, y)
                    if pixVal != KB_NO_DATA:
                        pix_sum += pixVal
                        pix_count += 1.0
                        pix_vals.append(pixVal)

                # Check that we get the correct answers.
                self.assertAlmostEqual(pix_sum, summedStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3)
                self.assertAlmostEqual(
                    pix_sum / pix_count, meanStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3
                )
                self.assertAlmostEqual(
                    np.median(pix_vals), medianStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3
                )

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu(self):
        params = StampParameters()
        params.radius = 3
        params.do_filtering = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = self.search.get_coadded_stamps([self.trj], [self.all_valid], params, True)
        self.assertEqual(summedStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].get_height(), 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = self.search.get_coadded_stamps([self.trj], [self.all_valid], params, True)
        self.assertEqual(meanStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].get_height(), 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEDIAN
        medianStamps = self.search.get_coadded_stamps([self.trj], [self.all_valid], params, True)
        self.assertEqual(medianStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(medianStamps[0].get_height(), 2 * params.radius + 1)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.get_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                pix_sum = 0.0
                pix_count = 0.0
                pix_vals = []
                for i in range(self.imCount):
                    t = times[i]
                    x = int(self.trj.x + self.trj.vx * t) + x_offset
                    y = int(self.trj.y + self.trj.vy * t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(x, y)
                    if pixVal != KB_NO_DATA:
                        pix_sum += pixVal
                        pix_count += 1.0
                        pix_vals.append(pixVal)

                # Check that we get the correct answers.
                self.assertAlmostEqual(pix_sum, summedStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3)
                self.assertAlmostEqual(
                    pix_sum / pix_count, meanStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3
                )
                self.assertAlmostEqual(
                    np.median(pix_vals), medianStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3
                )

    def test_coadd_cpu_use_inds(self):
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        inds = [[True] * self.imCount, [True] * self.imCount]
        inds[0][5] = False
        inds[1][3] = False
        inds[1][6] = False
        inds[1][7] = False
        inds[1][11] = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        meanStamps = self.search.get_coadded_stamps([self.trj, self.trj], inds, params, False)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.get_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                sum_0 = 0.0
                sum_1 = 0.0
                count_0 = 0.0
                count_1 = 0.0
                for i in range(self.imCount):
                    t = times[i]
                    x = int(self.trj.x + self.trj.vx * t) + x_offset
                    y = int(self.trj.y + self.trj.vy * t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(x, y)

                    if pixVal != KB_NO_DATA and inds[0][i] > 0:
                        sum_0 += pixVal
                        count_0 += 1.0

                    if pixVal != KB_NO_DATA and inds[1][i] > 0:
                        sum_1 += pixVal
                        count_1 += 1.0

                # Check that we get the correct answers.
                self.assertAlmostEqual(count_0, 19.0)
                self.assertAlmostEqual(count_1, 16.0)
                self.assertAlmostEqual(sum_0 / count_0, meanStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3)
                self.assertAlmostEqual(sum_1 / count_1, meanStamps[1].get_pixel(stamp_x, stamp_y), delta=1e-3)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu_use_inds(self):
        params = StampParameters()
        params.radius = 1
        params.do_filtering = False
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        inds = [[True] * self.imCount, [True] * self.imCount]
        inds[0][5] = False
        inds[1][3] = False
        inds[1][6] = False
        inds[1][7] = False
        inds[1][11] = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        meanStamps = self.search.get_coadded_stamps([self.trj, self.trj], inds, params, True)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.get_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                sum_0 = 0.0
                sum_1 = 0.0
                count_0 = 0.0
                count_1 = 0.0
                for i in range(self.imCount):
                    t = times[i]
                    x = int(self.trj.x + self.trj.vx * t) + x_offset
                    y = int(self.trj.y + self.trj.vy * t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(x, y)

                    if pixVal != KB_NO_DATA and inds[0][i] > 0:
                        sum_0 += pixVal
                        count_0 += 1.0

                    if pixVal != KB_NO_DATA and inds[1][i] > 0:
                        sum_1 += pixVal
                        count_1 += 1.0

                # Check that we get the correct answers.
                self.assertAlmostEqual(count_0, 19.0)
                self.assertAlmostEqual(count_1, 16.0)
                self.assertAlmostEqual(sum_0 / count_0, meanStamps[0].get_pixel(stamp_x, stamp_y), delta=1e-3)
                self.assertAlmostEqual(sum_1 / count_1, meanStamps[1].get_pixel(stamp_x, stamp_y), delta=1e-3)

    def test_coadd_filter_cpu(self):
        # Create a second Trajectory that isn't any good.
        trj2 = Trajectory()
        trj2.x = 1
        trj2.y = 1
        trj2.vx = 0
        trj2.vy = 0

        # Create a third Trajectory that is close to good, but offset.
        trj3 = Trajectory()
        trj3.x = self.trj.x + 2
        trj3.y = self.trj.y + 2
        trj3.vx = self.trj.vx
        trj3.vy = self.trj.vy

        # Create a fourth Trajectory that is close enough
        trj4 = Trajectory()
        trj4.x = self.trj.x + 1
        trj4.y = self.trj.y + 1
        trj4.vx = self.trj.vx
        trj4.vy = self.trj.vy

        # Compute the stacked science from a single Trajectory.
        all_valid_vect = [(self.all_valid) for i in range(4)]
        meanStamps = self.search.get_coadded_stamps(
            [self.trj, trj2, trj3, trj4], all_valid_vect, self.params, False
        )

        # The first and last are unfiltered
        self.assertEqual(meanStamps[0].get_width(), 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[0].get_height(), 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].get_width(), 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].get_height(), 2 * self.params.radius + 1)

        # The second and third are filtered.
        self.assertEqual(meanStamps[1].get_width(), 1)
        self.assertEqual(meanStamps[1].get_height(), 1)
        self.assertEqual(meanStamps[2].get_width(), 1)
        self.assertEqual(meanStamps[2].get_height(), 1)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_filter_gpu(self):
        # Create a second Trajectory that isn't any good.
        trj2 = Trajectory()
        trj2.x = 1
        trj2.y = 1
        trj2.vx = 0
        trj2.vy = 0

        # Create a third Trajectory that is close to good, but offset.
        trj3 = Trajectory()
        trj3.x = self.trj.x + 2
        trj3.y = self.trj.y + 2
        trj3.vx = self.trj.vx
        trj3.vy = self.trj.vy

        # Create a fourth Trajectory that is close enough
        trj4 = Trajectory()
        trj4.x = self.trj.x + 1
        trj4.y = self.trj.y + 1
        trj4.vx = self.trj.vx
        trj4.vy = self.trj.vy

        # Compute the stacked science from a single Trajectory.
        all_valid_vect = [(self.all_valid) for i in range(4)]
        meanStamps = self.search.get_coadded_stamps(
            [self.trj, trj2, trj3, trj4], all_valid_vect, self.params, True
        )

        # The first and last are unfiltered
        self.assertEqual(meanStamps[0].get_width(), 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[0].get_height(), 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].get_width(), 2 * self.params.radius + 1)
        self.assertEqual(meanStamps[3].get_height(), 2 * self.params.radius + 1)

        # The second and third are filtered.
        self.assertEqual(meanStamps[1].get_width(), 1)
        self.assertEqual(meanStamps[1].get_height(), 1)
        self.assertEqual(meanStamps[2].get_width(), 1)
        self.assertEqual(meanStamps[2].get_height(), 1)


if __name__ == "__main__":
    unittest.main()
