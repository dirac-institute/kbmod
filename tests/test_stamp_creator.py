import numpy as np
import unittest

from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import (
    add_fake_object,
    create_fake_times,
    make_fake_layered_image,
    FakeDataSet,
)
from kbmod.search import (
    HAS_GPU,
    ImageStack,
    LayeredImage,
    pixel_value_valid,
    RawImage,
    StampParameters,
    StampType,
    Trajectory,
    get_stamps,
    get_median_stamp,
    get_mean_stamp,
    get_summed_stamp,
    get_coadded_stamps,
    get_variance_weighted_stamp,
    create_stamps,
    create_stamps_xy,
    create_variance_stamps,
)


class test_stamp_creator(unittest.TestCase):
    def setUp(self):
        # image properties
        self.img_count = 20
        self.dim_y = 80
        self.dim_x = 60
        self.noise_level = 4.0
        self.variance = self.noise_level**2
        self.p = PSF.make_gaussian_kernel(1.0)

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

        # Set the filtering parameters.
        self.params = StampParameters()
        self.params.radius = 5
        self.params.stamp_type = StampType.STAMP_MEAN

        # Create a second (smaller) fake data set to use in some of the tests.
        self.img_count2 = 10
        self.fake_times = create_fake_times(self.img_count2, 57130.2, 1, 0.01, 1)
        self.ds = FakeDataSet(
            25,  # width
            35,  # height
            self.fake_times,  # time stamps
            1.0,  # noise level
            0.5,  # psf value
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        self.trj2 = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        self.ds.insert_object(self.trj)

    def test_create_stamps(self):
        stamps = create_stamps(self.ds.stack, self.trj2, 1, True, [])
        self.assertEqual(len(stamps), self.img_count2)
        for i in range(self.img_count2):
            self.assertEqual(stamps[i].image.shape, (3, 3))

            pix_val = self.ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(pix_val):
                self.assertTrue(np.isnan(stamps[i].get_pixel(1, 1)))
            else:
                self.assertAlmostEqual(pix_val, stamps[i].get_pixel(1, 1))

        # Check that we can set use_indices to produce only some stamps.
        use_times = [False, True, False, True, True, False, False, False, True, False]
        stamps = create_stamps(self.ds.stack, self.trj2, 1, True, use_times)
        self.assertEqual(len(stamps), np.count_nonzero(use_times))

        stamp_count = 0
        for i in range(self.img_count2):
            if use_times[i]:
                self.assertEqual(stamps[stamp_count].image.shape, (3, 3))

                pix_val = self.ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
                if np.isnan(pix_val):
                    self.assertTrue(np.isnan(stamps[stamp_count].get_pixel(1, 1)))
                else:
                    self.assertAlmostEqual(pix_val, stamps[stamp_count].get_pixel(1, 1))

                stamp_count += 1

    def test_create_stamps_xy(self):
        zeroed_times = np.array(self.ds.stack.build_zeroed_times())
        xvals = (self.trj2.x + self.trj2.vx * zeroed_times + 0.5).astype(int)
        yvals = (self.trj2.y + self.trj2.vy * zeroed_times + 0.5).astype(int)
        stamps = create_stamps_xy(self.ds.stack, 1, xvals, yvals, [])
        self.assertEqual(len(stamps), self.img_count2)
        for i in range(self.img_count2):
            self.assertEqual(stamps[i].image.shape, (3, 3))

            pix_val = self.ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(pix_val):
                pix_val = 0.0
            self.assertAlmostEqual(pix_val, stamps[i].get_pixel(1, 1))

        # Check that we can set use_indices to produce only some stamps.
        use_inds = np.array([1, 2, 3, 5, 6])
        xvals = (self.trj2.x + self.trj2.vx * zeroed_times[use_inds] + 0.5).astype(int)
        yvals = (self.trj2.y + self.trj2.vy * zeroed_times[use_inds] + 0.5).astype(int)
        stamps = create_stamps_xy(self.ds.stack, 1, xvals, yvals, use_inds)
        self.assertEqual(len(stamps), len(use_inds))

        for stamp_i, image_i in enumerate(use_inds):
            stamp = stamps[stamp_i]

            self.assertEqual(stamp.image.shape, (3, 3))
            pix_val = (
                self.ds.stack.get_single_image(image_i)
                .get_science()
                .get_pixel(
                    7 + image_i,
                    8 + 2 * image_i,
                )
            )
            if np.isnan(pix_val):
                pix_val = 0.0
            self.assertAlmostEqual(pix_val, stamp.get_pixel(1, 1))

    def test_create_variance_stamps(self):
        test_trj = Trajectory(8, 7, 1.0, 2.0)
        stamps = create_variance_stamps(self.ds.stack, self.trj2, 1, [])
        self.assertEqual(len(stamps), self.img_count2)
        for i in range(self.img_count2):
            self.assertEqual(stamps[i].image.shape, (3, 3))

            pix_val = self.ds.stack.get_single_image(i).get_variance().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(pix_val):
                self.assertTrue(np.isnan(stamps[i].get_pixel(1, 1)))
            else:
                self.assertAlmostEqual(pix_val, stamps[i].get_pixel(1, 1))

        # Check that we can set use_indices to produce only some stamps.
        use_times = [False, True, False, True, True, False, False, False, True, False]
        stamps = create_variance_stamps(self.ds.stack, self.trj2, 1, use_times)
        self.assertEqual(len(stamps), np.count_nonzero(use_times))

        stamp_count = 0
        for i in range(self.img_count2):
            if use_times[i]:
                self.assertEqual(stamps[stamp_count].image.shape, (3, 3))

                pix_val = self.ds.stack.get_single_image(i).get_variance().get_pixel(7 + i, 8 + 2 * i)
                if np.isnan(pix_val):
                    self.assertTrue(np.isnan(stamps[stamp_count].get_pixel(1, 1)))
                else:
                    self.assertAlmostEqual(pix_val, stamps[stamp_count].get_pixel(1, 1))

                stamp_count += 1

    def test_get_variance_weighted_stamp(self):
        sci1 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.single)
        var1 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.single)
        msk1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.single)
        layer1 = LayeredImage(sci1, var1, msk1, PSF.make_gaussian_kernel(1e-12), 0.0)
        layer1.apply_mask(0xFFFFFF)

        sci2 = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.single)
        var2 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.single)
        msk2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.single)
        layer2 = LayeredImage(sci2, var2, msk2, PSF.make_gaussian_kernel(1e-12), 0.0)
        layer2.apply_mask(0xFFFFFF)

        stack = ImageStack([layer1, layer2])

        # Unmoving point in the center. Result should be (1.0 / 1.0 + 2.0 / 0.5) / (1.0 / 1.0 + 1.0 / 0.5)
        stamp = get_variance_weighted_stamp(stack, Trajectory(1, 1, 0.0, 0.0), 0, [])
        self.assertEqual(stamp.image.shape, (1, 1))
        self.assertAlmostEqual(stamp.get_pixel(0, 0), 5.0 / 3.0)

        # Unmoving point in the top corner. Should ignore the point in the second image.
        stamp = get_variance_weighted_stamp(stack, Trajectory(0, 0, 0.0, 0.0), 0, [])
        self.assertEqual(stamp.image.shape, (1, 1))
        self.assertAlmostEqual(stamp.get_pixel(0, 0), 1.0)

        # Unmoving point in the bottom corner. Should ignore the point in the first image.
        stamp = get_variance_weighted_stamp(stack, Trajectory(2, 2, 0.0, 0.0), 0, [])
        self.assertEqual(stamp.image.shape, (1, 1))
        self.assertAlmostEqual(stamp.get_pixel(0, 0), 2.0)

    def test_sci_viz_stamps(self):
        sci_stamps = get_stamps(self.stack, self.trj, 2)
        self.assertEqual(len(sci_stamps), self.img_count)

        times = self.stack.build_zeroed_times()
        for i in range(self.img_count):
            self.assertEqual(sci_stamps[i].width, 5)
            self.assertEqual(sci_stamps[i].height, 5)

            # Compute the pixel value at the projected location.
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
        sci = get_summed_stamp(self.stack, self.trj, 2, [])
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

        medianStamps0 = get_median_stamp(self.stack, self.trj, 2, goodIdx[0])
        self.assertEqual(medianStamps0.width, 5)
        self.assertEqual(medianStamps0.height, 5)

        medianStamps1 = get_median_stamp(self.stack, self.trj, 2, goodIdx[1])
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
        medianStamp = get_median_stamp(self.stack, trj, 2, self.all_valid)
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

        meanStamp0 = get_mean_stamp(self.stack, self.trj, 2, goodIdx[0])
        self.assertEqual(meanStamp0.width, 5)
        self.assertEqual(meanStamp0.height, 5)

        meanStamp1 = get_mean_stamp(self.stack, self.trj, 2, goodIdx[1])
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
        meanStamp = get_mean_stamp(self.stack, trj, 2, self.all_valid)
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
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Basic Stamp parameters.
        params = StampParameters()
        params.radius = 1

        # Test summed.
        params.stamp_type = StampType.STAMP_SUM
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, False)
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
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, False)
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
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, False)
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
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Basic Stamp parameters.
        params = StampParameters()
        params.radius = 1

        # Test summed.
        params.stamp_type = StampType.STAMP_SUM
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 5.0)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 6.0)

        # Test mean.
        params.stamp_type = StampType.STAMP_MEAN
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

        # Test median.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, True)
        self.assertAlmostEqual(stamps[0].get_pixel(0, 1), 3.0)
        self.assertAlmostEqual(stamps[0].get_pixel(1, 1), 2.5)
        self.assertAlmostEqual(stamps[0].get_pixel(2, 1), 2.0)

    def test_coadd_cpu(self):
        params = StampParameters()
        params.radius = 3

        # Compute the stacked science (summed and mean) from a single Trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = get_coadded_stamps(self.stack, [self.trj], [self.all_valid], params, False)
        self.assertEqual(summedStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = get_coadded_stamps(self.stack, [self.trj], [self.all_valid], params, False)
        self.assertEqual(meanStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEDIAN
        medianStamps = get_coadded_stamps(self.stack, [self.trj], [self.all_valid], params, False)
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

        # Compute the stacked science (summed and mean) from a single Trajectory.
        params.stamp_type = StampType.STAMP_SUM
        summedStamps = get_coadded_stamps(self.stack, [self.trj], [self.all_valid], params, True)
        self.assertEqual(summedStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(summedStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEAN
        meanStamps = get_coadded_stamps(self.stack, [self.trj], [self.all_valid], params, True)
        self.assertEqual(meanStamps[0].width, 2 * params.radius + 1)
        self.assertEqual(meanStamps[0].height, 2 * params.radius + 1)

        params.stamp_type = StampType.STAMP_MEDIAN
        medianStamps = get_coadded_stamps(self.stack, [self.trj], [self.all_valid], params, True)
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
    def test_coadd_gpu_large(self):
        """Check that the coadds are generated with a larger radius."""
        # Create an image set with three images.
        num_times = 50
        imlist = []
        for i in range(num_times):
            imlist.append(make_fake_layered_image(200, 300, 0.5, 0.05, i, self.p, seed=i))
        stack = ImageStack(imlist)
        all_valid = [True] * num_times

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=100, y=100, vx=0.0, vy=0.0)

        # Test a few different values of radius.
        for radius in [10, 20, 30]:
            params = StampParameters()
            params.radius = radius

            # Test that we get valid stamp values for all pixels (not all zeros).
            params.stamp_type = StampType.STAMP_MEAN
            stamps = get_coadded_stamps(stack, [trj], [all_valid], params, True)
            num_zeros = 0
            for i in range(2 * params.radius + 1):
                for j in range(2 * params.radius + 1):
                    pix_val = stamps[0].get_pixel(i, j)
                    if pix_val == 0.0:
                        num_zeros += 1

            self.assertLess(num_zeros, 5)

    def test_coadd_cpu_fallback(self):
        # Create an image set with three images.
        imlist = []
        for i in range(3):
            imlist.append(make_fake_layered_image(3000, 3000, 0.1, 0.01, i, self.p, seed=i))
        stack = ImageStack(imlist)
        all_valid = [True, True, True]  # convenience array

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=1000, y=1000, vx=0.0, vy=0.0)

        # Use a radius that is too large for the GPU.
        params = StampParameters()
        params.radius = 500

        # Test that we get valid stamp values for all pixels.
        params.stamp_type = StampType.STAMP_MEAN
        stamps = get_coadded_stamps(stack, [trj], [all_valid], params, True)
        for i in range(2 * params.radius + 1):
            for j in range(2 * params.radius + 1):
                pix_val = stamps[0].get_pixel(i, j)
                self.assertLess(pix_val, 1000.0)
                self.assertGreater(pix_val, -1000.0)

    def test_coadd_cpu_use_inds(self):
        params = StampParameters()
        params.radius = 1
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        inds = [[True] * self.img_count, [True] * self.img_count]
        inds[0][5] = False
        inds[1][3] = False
        inds[1][6] = False
        inds[1][7] = False
        inds[1][11] = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        meanStamps = get_coadded_stamps(self.stack, [self.trj, self.trj], inds, params, False)

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
        params.stamp_type = StampType.STAMP_MEAN

        # Mark a few of the observations as "do not use"
        inds = [[True] * self.img_count, [True] * self.img_count]
        inds[0][5] = False
        inds[1][3] = False
        inds[1][6] = False
        inds[1][7] = False
        inds[1][11] = False

        # Compute the stacked science (summed and mean) from a single Trajectory.
        meanStamps = get_coadded_stamps(self.stack, [self.trj, self.trj], inds, params, True)

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


if __name__ == "__main__":
    unittest.main()
