import unittest

import numpy as np

from kbmod.fake_data.fake_data_creator import image_stack_add_fake_object, make_fake_image_stack
from kbmod.core.psf import PSF
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

        # Create image stack with single moving object.
        self.times = np.array([i / self.img_count for i in range(self.img_count)])
        rng = np.random.default_rng(100)
        image_stack_py = make_fake_image_stack(
            self.dim_y,
            self.dim_x,
            self.times,
            noise_level=self.noise_level,
            psf_val=1.0,
            rng=rng,
        )

        image_stack_add_fake_object(
            image_stack_py,
            self.start_x,
            self.start_y,
            self.vxel,
            self.vyel,
            self.object_flux,
        )

        # Mask a pixel in every other image.
        for i in range(self.img_count):
            if i % 2 == 0:
                image_stack_py.sci[i][self.masked_y, self.masked_x] = np.nan
                image_stack_py.var[i][self.masked_y, self.masked_x] = np.nan

        self.search = StackSearch(
            image_stack_py.sci,
            image_stack_py.var,
            image_stack_py.psfs,
            image_stack_py.zeroed_times,
        )

        self.trj_gen = KBMODV1Search(
            self.velocity_steps,
            self.min_vel,
            self.max_vel,
            self.angle_steps,
            self.min_angle,
            self.max_angle,
        )

    def test_evaluate_single_trajectory(self):
        test_trj = Trajectory(
            x=self.start_x,
            y=self.start_y,
            vx=self.vxel,
            vy=self.vyel,
        )
        self.search.evaluate_single_trajectory(test_trj, False)

        # We found a valid result.
        self.assertGreater(test_trj.obs_count, 0)
        self.assertGreater(test_trj.flux, 0.0)
        self.assertGreater(test_trj.lh, 0.0)

    def test_search_linear_trajectory(self):
        test_trj = self.search.search_linear_trajectory(
            self.start_x,
            self.start_y,
            self.vxel,
            self.vyel,
            False,
        )

        # We found a valid result.
        self.assertGreater(test_trj.obs_count, 0)
        self.assertGreater(test_trj.flux, 0.0)
        self.assertGreater(test_trj.lh, 0.0)

    def test_results_cpu(self):
        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, False)

        # Check that we have the at most the expected number of results (using the default
        # of 8 results per pixel searched). We can have fewer since initial filtering
        # is done during the search.
        expected_size = 8 * self.dim_x * self.dim_y
        self.assertEqual(self.search.compute_max_results(), expected_size)
        results = self.search.get_results(0, 10 * expected_size)
        self.assertLessEqual(len(results), expected_size)
        self.assertGreater(len(results), 0)

        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results_gpu(self):
        candidates = [trj for trj in self.trj_gen]
        self.search.search_all(candidates, True)

        # Check that we have the at most the expected number of results (using the default
        # of 8 results per pixel searched). We can have fewer since initial filtering
        # is done during the search.
        expected_size = 8 * self.dim_x * self.dim_y
        self.assertEqual(self.search.compute_max_results(), expected_size)
        results = self.search.get_results(0, 10 * expected_size)
        self.assertLessEqual(len(results), expected_size)
        self.assertGreater(len(results), 0)

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
        self.search.search_all(candidates, True)

        # Check that we have the at most the expected number of results (using the default
        # of 8 results per pixel searched). We can have fewer since initial filtering
        # is done during the search.
        expected_size = 5 * (self.dim_x + 20) * (self.dim_y + 20)
        results = self.search.get_results(0, 10 * expected_size)
        self.assertLessEqual(len(results), expected_size)
        self.assertGreater(len(results), 0)

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
        self.search.search_all(candidates, True)

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
        times = np.array([i / self.img_count for i in range(self.img_count)])
        rng = np.random.default_rng(100)
        image_stack_py = make_fake_image_stack(
            self.dim_y,
            self.dim_x,
            times,
            noise_level=self.noise_level,
            psf_val=1.0,
            rng=rng,
        )
        image_stack_add_fake_object(image_stack_py, -3, 12, 25.0, 10.0, self.object_flux)

        search = StackSearch(
            image_stack_py.sci,
            image_stack_py.var,
            image_stack_py.psfs,
            image_stack_py.zeroed_times,
        )

        # Do the extended search.
        search.set_start_bounds_x(-10, self.dim_x + 10)
        search.set_start_bounds_y(-10, self.dim_y + 10)
        candidates = [trj for trj in self.trj_gen]
        search.search_all(candidates, True)

        # Check the results.
        results = search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, trj.x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, trj.y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / trj.vx, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / trj.vy, 1, delta=self.velocity_error)

    def test_stack_search_set_min_obs(self):
        self.search.set_min_obs(1)  # Okay
        self.search.set_min_obs(self.img_count)  # Okay
        with self.assertRaises(RuntimeError):
            self.search.set_min_obs(-1)
        with self.assertRaises(RuntimeError):
            self.search.set_min_obs(self.img_count + 1)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_search_too_many_images(self):
        # Create a very large image stack.
        width = 10
        height = 20
        num_times = 1_000
        times = np.array([i / num_times for i in range(num_times)])
        image_stack_py = make_fake_image_stack(
            height,
            width,
            times,
            noise_level=0.5,
            psf_val=1.0,
        )
        self.assertEqual(len(image_stack_py), num_times)

        # Create the search stack and try to evaluate.
        search = StackSearch(
            image_stack_py.sci,
            image_stack_py.var,
            image_stack_py.psfs,
            image_stack_py.zeroed_times,
        )
        test_trj = Trajectory(x=0, y=0, vx=0.0, vy=0.0)
        self.assertRaises(RuntimeError, search.search_all, [test_trj], True)
        self.assertRaises(RuntimeError, search.evaluate_single_trajectory, test_trj, True)


if __name__ == "__main__":
    unittest.main()
