import unittest

import numpy as np

from kbmod import *


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
            im = layered_image(str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i + 5)
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
        self.search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

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
        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.x_v / self.x_vel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.y_v / self.y_vel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    def test_sci_stamps(self):
        sci_stamps = self.search.sci_stamps(self.trj, 2)
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
        sci = self.search.stacked_sci(self.trj, 2)
        self.assertEqual(sci.get_width(), 5)
        self.assertEqual(sci.get_height(), 5)

        # Compute a vector of stacked sciences from a vector of trajectories.
        sci_vect = self.search.summed_sci([self.trj], 2)[0]
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
        # Compute the stacked science from a single trajectory.
        goodIdx = [[1] * self.imCount]
        medianStamps = self.search.median_stamps([self.trj], goodIdx, 2)
        self.assertEqual(medianStamps[0].get_width(), 5)
        self.assertEqual(medianStamps[0].get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.get_times()
        pix_values = []
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.x_v * t)
            y = int(self.trj.y + self.trj.y_v * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal != KB_NO_DATA:
                pix_values.append(pixVal)
        self.assertEqual(len(pix_values), self.imCount)

        # Check that we get the correct answer.
        self.assertAlmostEqual(np.median(pix_values), medianStamps[0].get_pixel(2, 2), delta=1e-5)

    def test_median_stamps_no_data(self):
        # Create a trajectory that goes through the masked pixels.
        trj = trajectory()
        trj.x = self.masked_x
        trj.y = self.masked_y
        trj.x_v = 0
        trj.y_v = 0

        # Compute the stacked science from a single trajectory.
        goodIdx = [[1] * self.imCount]
        medianStamps = self.search.median_stamps([trj], goodIdx, 2)
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
        # Compute the stacked science from a single trajectory.
        goodIdx = [[1] * self.imCount]
        meanStamps = self.search.mean_stamps([self.trj], goodIdx, 2)
        self.assertEqual(meanStamps[0].get_width(), 5)
        self.assertEqual(meanStamps[0].get_height(), 5)

        # Compute the true median pixel for the middle of the track.
        times = self.stack.get_times()
        pix_sum = 0.0
        pix_count = 0.0
        for i in range(self.imCount):
            t = times[i]
            x = int(self.trj.x + self.trj.x_v * t)
            y = int(self.trj.y + self.trj.y_v * t)
            pixVal = self.imlist[i].get_science().get_pixel(x, y)
            if pixVal != KB_NO_DATA:
                pix_sum += pixVal
                pix_count += 1
        self.assertEqual(pix_count, self.imCount)

        # Check that we get the correct answer.
        self.assertAlmostEqual(pix_sum / pix_count, meanStamps[0].get_pixel(2, 2), delta=1e-5)

    def test_mean_stamps_no_data(self):
        # Create a trajectory that goes through the masked pixels.
        trj = trajectory()
        trj.x = self.masked_x
        trj.y = self.masked_y
        trj.x_v = 0
        trj.y_v = 0

        # Compute the stacked science from a single trajectory.
        goodIdx = [[1] * self.imCount]
        meanStamps = self.search.mean_stamps([trj], goodIdx, 2)
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


if __name__ == "__main__":
    unittest.main()
