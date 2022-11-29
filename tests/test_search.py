import unittest

import numpy as np

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
        self.p = psf(1.0)

        # object properties
        self.object_flux = 250.0
        self.start_x = 17
        self.start_y = 12
        self.x_vel = 21.0
        self.y_vel = 16.0

        # create a trajectory for the object
        self.trj = trajectory()
        self.trj.x = self.start_x
        self.trj.y = self.start_y
        self.trj.x_v = self.x_vel
        self.trj.y_v = self.y_vel

        # create a trajectory result for computing stamps.
        self.trj_res = trj_result(self.trj, self.imCount)

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
            im = layered_image(str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i)
            im.add_object(
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
            )

            # Mask a pixel in half the images.
            if i % 2 == 0:
                mask = im.get_mask()
                mask.set_pixel(self.masked_x, self.masked_y, 1)
                im.set_mask(mask)
                im.apply_mask_flags(1, [])

            self.imlist.append(im)
        self.stack = image_stack(self.imlist)
        self.search = stack_search(self.stack)

    def test_psiphi(self):
        p = psf(0.00001)

        # Image1 has a single object.
        image1 = layered_image("test1", 5, 10, 2.0, 4.0, 1.0, p)
        image1.add_object(3.5, 2.5, 400.0)

        # Image2 has a single object and a masked pixel.
        image2 = layered_image("test2", 5, 10, 2.0, 4.0, 2.0, p)
        image2.add_object(2.5, 4.5, 400.0)
        mask = image2.get_mask()
        mask.set_pixel(4, 9, 1)
        image2.set_mask(mask)
        image2.apply_mask_flags(1, [])

        # Create a stack from the two objects.
        stack = image_stack([image1, image2])
        search = stack_search(stack)

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
        self.assertAlmostEqual(best.x_v / self.x_vel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.y_v / self.y_vel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    def test_sci_viz_stamps(self):
        sci_stamps = self.search.science_viz_stamps(self.trj, 2)
        self.assertEqual(len(sci_stamps), self.imCount)

        times = self.stack.get_times()
        for i in range(self.imCount):
            self.assertEqual(sci_stamps[i].get_width(), 5)
            self.assertEqual(sci_stamps[i].get_height(), 5)

            # Compute the interpolated pixel value at the projected location.
            t = times[i]
            x = float(self.trj.x) + self.trj.x_v * t
            y = float(self.trj.y) + self.trj.y_v * t
            pixVal = self.imlist[i].get_science().get_pixel_interp(x, y)
            if pixVal == KB_NO_DATA:
                pivVal = 0.0

            # Check that pixel value of the projected location equals
            # that of the base image.
            self.assertAlmostEqual(sci_stamps[i].get_pixel(2, 2), pixVal, delta=0.001)

    def test_stacked_sci(self):
        # Compute the stacked science from a single trajectory.
        sci = self.search.summed_sci_stamp(self.trj_res, 2, True)
        self.assertEqual(sci.get_width(), 5)
        self.assertEqual(sci.get_height(), 5)

        # Compute a vector of stacked sciences from a vector of trajectories.
        sci_vect = self.search.summed_sci_stamps([self.trj_res], 2)[0]
        self.assertEqual(sci_vect.get_width(), 5)
        self.assertEqual(sci_vect.get_height(), 5)

        # Compute the true stacked pixel for the middle of the track.
        times = self.stack.get_times()
        sum_middle = 0.0
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.x_v * t)
            y = int(self.trj.y + self.trj.y_v * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal == KB_NO_DATA:
                pivVal = 0.0
            sum_middle = sum_middle + pixVal

        # Check that the two different approaches for stack science
        # match the true value.
        self.assertAlmostEqual(sci.get_pixel(2, 2), sum_middle, delta=0.001)
        self.assertAlmostEqual(sci_vect.get_pixel(2, 2), sum_middle, delta=0.001)

    def test_median_stamps_trj(self):
        # Compute the stacked science from two trajectories (one with bad points).
        goodIdx = [[1] * self.imCount for _ in range(2)]
        goodIdx[1][1] = 0
        goodIdx[1][5] = 0
        goodIdx[1][9] = 0
        trj_res0 = trj_result(self.trj, goodIdx[0])
        trj_res1 = trj_result(self.trj, goodIdx[1])
        
        medianStamps = self.search.median_sci_stamps([trj_res0, trj_res1], 2)
        self.assertEqual(medianStamps[0].get_width(), 5)
        self.assertEqual(medianStamps[0].get_height(), 5)
        self.assertEqual(medianStamps[1].get_width(), 5)
        self.assertEqual(medianStamps[1].get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.get_times()
        pix_values0 = []
        pix_values1 = []
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.x_v * t)
            y = int(self.trj.y + self.trj.y_v * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal != KB_NO_DATA and goodIdx[0][i] == 1:
                pix_values0.append(pixVal)
            if pixVal != KB_NO_DATA and goodIdx[1][i] == 1:
                pix_values1.append(pixVal)
        self.assertEqual(len(pix_values0), self.imCount)
        self.assertEqual(len(pix_values1), self.imCount - 3)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values0), medianStamps[0].get_pixel(2, 2), delta=1e-5)
        self.assertAlmostEqual(np.median(pix_values1), medianStamps[1].get_pixel(2, 2), delta=1e-5)

    def test_median_stamps_no_data(self):
        # Create a trajectory that goes through the masked pixels.
        trj = trajectory()
        trj.x = self.masked_x
        trj.y = self.masked_y
        trj.x_v = 0
        trj.y_v = 0
        trj_res = trj_result(trj, self.imCount)

        # Compute the stacked science from a single trajectory.
        medianStamps = self.search.median_sci_stamps([trj_res], 2)
        self.assertEqual(medianStamps[0].get_width(), 5)
        self.assertEqual(medianStamps[0].get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        pix_values = []
        for i in range(self.imCount):
            pixVal = self.imlist[i].get_science().get_pixel(self.masked_x, self.masked_y)
            if pixVal != KB_NO_DATA:
                pix_values.append(pixVal)
        self.assertEqual(len(pix_values), self.imCount / 2)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values), medianStamps[0].get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_trj(self):
        # Compute the stacked science from two trajectories (one with bad points).
        goodIdx = [[1] * self.imCount for _ in range(2)]
        goodIdx[1][1] = 0
        goodIdx[1][5] = 0
        goodIdx[1][9] = 0
        trj_res0 = trj_result(self.trj, goodIdx[0])
        trj_res1 = trj_result(self.trj, goodIdx[1])
        
        meanStamps = self.search.mean_sci_stamps([trj_res0, trj_res1], 2)
        self.assertEqual(meanStamps[0].get_width(), 5)
        self.assertEqual(meanStamps[0].get_height(), 5)
        self.assertEqual(meanStamps[1].get_width(), 5)
        self.assertEqual(meanStamps[1].get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.get_times()
        pix_sum0 = 0.0
        pix_sum1 = 0.0
        pix_count0 = 0.0
        pix_count1 = 0.0
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.x_v * t)
            y = int(self.trj.y + self.trj.y_v * t)
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
        self.assertAlmostEqual(pix_sum0 / pix_count0, meanStamps[0].get_pixel(2, 2), delta=1e-5)
        self.assertAlmostEqual(pix_sum1 / pix_count1, meanStamps[1].get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_no_data(self):
        # Create a trajectory that goes through the masked pixels.
        trj = trajectory()
        trj.x = self.masked_x
        trj.y = self.masked_y
        trj.x_v = 0
        trj.y_v = 0
        trj_res = trj_result(trj, self.imCount)

        # Compute the stacked science from a single trajectory.
        meanStamps = self.search.mean_sci_stamps([trj_res], 2)
        self.assertEqual(meanStamps[0].get_width(), 5)
        self.assertEqual(meanStamps[0].get_height(), 5)

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
        self.assertAlmostEqual(pix_sum / pix_count, meanStamps[0].get_pixel(2, 2), delta=1e-5)

    def test_coadd_gpu(self):
        params = stamp_parameters()
        params.radius = 3
        params.do_filtering = False
        
        # Compute the stacked science (summed and mean) from a single trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = self.search.gpu_coadded_stamps([self.trj], params)
        self.assertEqual(summedStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].get_height(), 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = self.search.gpu_coadded_stamps([self.trj], params)
        self.assertEqual(meanStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].get_height(), 2 * params.radius + 1)

        # Compute the true summed and mean pixels for all of the pixels in the stamp.
        times = self.stack.get_times()
        for stamp_x in range(2 * params.radius + 1):
            for stamp_y in range(2 * params.radius + 1):
                x_offset = stamp_x - params.radius
                y_offset = stamp_y - params.radius

                pix_sum = 0.0
                pix_count = 0.0
                for i in range(self.imCount):
                    t = times[i]
                    x = int(self.trj.x + self.trj.x_v * t) + x_offset
                    y = int(self.trj.y + self.trj.y_v * t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(x, y)                    
                    if pixVal != KB_NO_DATA:
                        pix_sum += pixVal
                        pix_count += 1.0

                # Check that we get the correct answers.
                self.assertAlmostEqual(pix_sum, summedStamps[0].get_pixel(stamp_x, stamp_y),
                                       delta=1e-3)
                self.assertAlmostEqual(pix_sum / pix_count,
                                       meanStamps[0].get_pixel(stamp_x, stamp_y),
                                       delta=1e-3)

    def test_coadd_gpu_use_inds(self):
        params = stamp_parameters()
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

        # Compute the stacked science (summed and mean) from a single trajectory.
        meanStamps = self.search.gpu_coadded_stamps([self.trj, self.trj], inds, params)

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
                    x = int(self.trj.x + self.trj.x_v * t) + x_offset
                    y = int(self.trj.y + self.trj.y_v * t) + y_offset
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
                self.assertAlmostEqual(sum_0 / count_0,
                                       meanStamps[0].get_pixel(stamp_x, stamp_y),
                                       delta=1e-3)
                self.assertAlmostEqual(sum_1 / count_1,
                                       meanStamps[1].get_pixel(stamp_x, stamp_y),
                                       delta=1e-3)

    def test_coadd_gpu_trj_result(self):
        params = stamp_parameters()
        params.radius = 1
        params.do_filtering = False
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        trj_result_1 = trj_result(self.trj, self.imCount)
        trj_result_1.set_index_valid(5, False)
        
        trj_result_2 = trj_result(self.trj, self.imCount)
        trj_result_2.set_index_valid(4, False)
        trj_result_2.set_index_valid(7, False)
        trj_result_2.set_index_valid(13, False)

        # Compute the stacked science (summed and mean) from a single trajectory.
        meanStamps = self.search.gpu_coadded_stamps([trj_result_1, trj_result_2], params)

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
                    x = int(self.trj.x + self.trj.x_v * t) + x_offset
                    y = int(self.trj.y + self.trj.y_v * t) + y_offset
                    pixVal = self.imlist[i].get_science().get_pixel(x, y)  

                    if pixVal != KB_NO_DATA and i != 5:
                        sum_0 += pixVal
                        count_0 += 1.0

                    if pixVal != KB_NO_DATA and i != 4 and i != 7 and i != 13:
                        sum_1 += pixVal
                        count_1 += 1.0

                # Check that we get the correct answers.
                self.assertAlmostEqual(count_0, 19.0)
                self.assertAlmostEqual(count_1, 17.0)
                self.assertAlmostEqual(sum_0 / count_0,
                                       meanStamps[0].get_pixel(stamp_x, stamp_y),
                                       delta=1e-3)
                self.assertAlmostEqual(sum_1 / count_1,
                                       meanStamps[1].get_pixel(stamp_x, stamp_y),
                                       delta=1e-3)

    def test_coadd_filter_gpu(self):
        # Create a second trajectory that isn't any good.
        trj2 = trajectory()
        trj2.x = 1
        trj2.y = 1
        trj2.x_v = 0
        trj2.y_v = 0

        # Create a third trajectory that is close to good, but offset.
        trj3 = trajectory()
        trj3.x = self.trj.x + 2
        trj3.y = self.trj.y + 2
        trj3.x_v = self.trj.x_v
        trj3.y_v = self.trj.y_v

        # Create a fourth trajectory that is close enough
        trj4 = trajectory()
        trj4.x = self.trj.x + 1
        trj4.y = self.trj.y + 1
        trj4.x_v = self.trj.x_v
        trj4.y_v = self.trj.y_v

        # Set the filtering parameters.
        params = stamp_parameters()
        params.radius = 5
        params.do_filtering = True
        params.stamp_type = StampType.STAMP_MEAN
        params.center_thresh = 0.03
        params.peak_offset_x = 1.5
        params.peak_offset_y = 1.5
        params.m01 = 0.6
        params.m10 = 0.6
        params.m11 = 2.0
        params.m02 = 35.5
        params.m20 = 35.5

        # Compute the stacked science from a single trajectory.
        meanStamps = self.search.gpu_coadded_stamps([self.trj, trj2, trj3, trj4], params)

        # The first and last are unfiltered
        self.assertEqual(meanStamps[0].get_width(), 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].get_height(), 2 * params.radius + 1)
        self.assertEqual(meanStamps[3].get_width(), 2 * params.radius + 1)
        self.assertEqual(meanStamps[3].get_height(), 2 * params.radius + 1)

        # The second and third are filtered.
        self.assertEqual(meanStamps[1].get_width(), 1)
        self.assertEqual(meanStamps[1].get_height(), 1)
        self.assertEqual(meanStamps[2].get_width(), 1)
        self.assertEqual(meanStamps[2].get_height(), 1)
        
if __name__ == "__main__":
    unittest.main()
