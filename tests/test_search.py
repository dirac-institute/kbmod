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

        # Check that we have the expected number of results using the default
        # of 8 results per pixel searched.
        expected_size = 8 * self.dim_x * self.dim_y
        self.assertEqual(self.search.compute_max_results(), expected_size)
        results = self.search.get_results(0, 10 * expected_size)
        self.assertEqual(len(results), expected_size)

        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_extended_bounds(self):
        self.search.set_results_per_pixel(5)
        self.search.set_start_bounds_x(-10, self.dim_x + 10)
        self.search.set_start_bounds_y(-10, self.dim_y + 10)

        num_results = self.search.compute_max_results()
        expected_num_results = (self.dim_x + 20) * (self.dim_y + 20) * 5
        self.assertEqual(num_results, expected_num_results)

        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, int(self.img_count / 2))

        # Check that we have the expected number of results
        expected_size = 5 * (self.dim_x + 20) * (self.dim_y + 20)
        results = self.search.get_results(0, 10 * expected_size)
        self.assertEqual(len(results), expected_size)

        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_reduced_bounds(self):
        self.search.set_results_per_pixel(10)
        self.search.set_start_bounds_x(5, self.dim_x - 5)
        self.search.set_start_bounds_y(5, self.dim_y - 5)

        num_results = self.search.compute_max_results()
        expected_num_results = (self.dim_x - 10) * (self.dim_y - 10) * 10
        self.assertEqual(num_results, expected_num_results)

        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, int(self.img_count / 2))

        # Check that we have the expected number of results
        expected_size = 10 * (self.dim_x - 10) * (self.dim_y - 10)
        results = self.search.get_results(0, 10 * expected_size)
        self.assertEqual(len(results), expected_size)

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

    @staticmethod
    def result_hash(res):
        return hash((res.x, res.y, res.vx, res.vy, res.lh, res.obs_count))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
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
