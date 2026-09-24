"""Check image-count boundaries with a moving source, a mask, and a flux outlier."""

import unittest

import numpy as np

from kbmod.search import (
    MAX_NUM_IMAGES,
    MAX_NUM_IMAGE_TIMES,
    StackSearch,
    Trajectory,
    kb_has_gpu,
)

# Include the old limit, the number of threads per block, and the new limit.
IMAGE_COUNTS = (150, 200, 256, 385, 447, 448)


def make_search(num_images, num_bytes=4, masked=True):
    times = np.arange(num_images, dtype=float) / 64.0

    # Scale the times so the trajectories stay on the images. This is needed for
    # for the MAX_NUM_IMAGE_TIMES + 1 test.
    while np.max(times) > 10:
        times /= 10.0

    sci = np.zeros((num_images, 5, 17), dtype=np.float32)
    var = np.ones_like(sci)
    psfs = np.ones((num_images, 1, 1), dtype=np.float32)
    for i, time in enumerate(times):
        x = int(np.floor(1 + 2.0 * time + 0.5))
        y = int(np.floor(1 + 0.25 * time + 0.5))
        sci[i, y, x] = 100.0 if i == num_images - 1 else 10.0
        if masked and i == num_images - 2:
            sci[i, y, x] = np.nan
            var[i, y, x] = np.nan

    search = StackSearch(sci, var, psfs, times, num_bytes)
    # Only one thread searches a pixel. All threads must still load times and
    # synchronize before returning for out-of-bounds search positions.
    search.set_start_bounds_x(1, 2)
    search.set_start_bounds_y(1, 2)
    search.set_results_per_pixel(1)
    search.set_min_obs(num_images - int(masked))
    return search


def candidate():
    return Trajectory(x=1, y=1, vx=2.0, vy=0.25)


class TestImageLimitCPU(unittest.TestCase):
    def test_exported_gpu_limit(self):
        # Keep the intended cap explicit so a build with the wrong value fails.
        self.assertIsInstance(MAX_NUM_IMAGES, int)
        self.assertEqual(MAX_NUM_IMAGES, 448)
        self.assertIsInstance(MAX_NUM_IMAGE_TIMES, int)
        self.assertEqual(MAX_NUM_IMAGE_TIMES, 2000)

    def test_unfiltered_reference(self):
        for n in (*IMAGE_COUNTS, 449):
            with self.subTest(num_images=n):
                search = make_search(n)
                search.search_all([candidate()], False)
                results = search.get_results(0, 1)
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertEqual(result.obs_count, n - 1)
                self.assertAlmostEqual(result.flux, (10.0 * (n - 2) + 100.0) / (n - 1), places=4)
                self.assertAlmostEqual(result.lh, (10.0 * (n - 2) + 100.0) / np.sqrt(n - 1), places=3)


@unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
class TestImageLimitGPU(unittest.TestCase):
    def check_search(self, n, filtered, num_bytes=4, masked=True):
        search = make_search(n, num_bytes, masked)
        # Reference values include the same encoding as the device will read.
        curves = np.asarray(search.get_all_psi_phi_curves([candidate()]))[0]
        psi, phi = curves[:n], curves[n:]
        if filtered:
            # All unmasked source values are identical except the final outlier.
            # Its removal is independent of the sigma-G percentile convention.
            psi, phi = psi[:-1], phi[:-1]
            search.enable_gpu_sigmag_filter([0.25, 0.75], 0.7413, 0.0)
        expected_flux = np.sum(psi) / np.sum(phi)
        expected_lh = np.sum(psi) / np.sqrt(np.sum(phi))

        # This invokes the host version of the CUDA evaluator and also checks
        # the public API's inclusive limit.
        single = candidate()
        search.evaluate_single_trajectory(single, True)

        # Unlike evaluate_single_trajectory, search_all launches the GPU kernel
        # and exercises shared time loading beyond the first 256 threads.
        search.search_all([candidate()], True)
        results = search.get_results(0, 1)
        self.assertEqual(len(results), 1)
        for result in (single, results[0]):
            self.assertEqual((result.x, result.y, result.vx, result.vy), (1, 1, 2.0, 0.25))
            # Sigma-G changes flux and likelihood, but obs_count intentionally
            # remains the count before clipping (after excluding masked pixels).
            self.assertEqual(result.obs_count, n - int(masked))
            np.testing.assert_allclose(result.flux, expected_flux, rtol=1e-4)
            np.testing.assert_allclose(result.lh, expected_lh, rtol=1e-4)

    def test_search_boundaries(self):
        for n in IMAGE_COUNTS:
            for filtered in (False, True):
                with self.subTest(num_images=n, filtered=filtered):
                    self.check_search(n, filtered)

    def test_encoded_search_at_limit(self):
        for num_bytes in (1, 2):
            for filtered in (False, True):
                with self.subTest(num_bytes=num_bytes, filtered=filtered):
                    self.check_search(448, filtered, num_bytes)

    def test_all_images_valid_at_limit(self):
        # Fill every slot of the evaluator's local arrays, including sigma-G's
        # sorting arrays; the masked boundary tests use one fewer slot.
        for filtered in (False, True):
            with self.subTest(filtered=filtered):
                self.check_search(448, filtered, masked=False)

    def test_reject_over_limit(self):
        search = make_search(449)
        search.enable_gpu_sigmag_filter([0.25, 0.75], 0.7413, 0.0)
        with self.assertRaisesRegex(RuntimeError, "Too many images to evaluate on GPU. Max = 448"):
            search.evaluate_single_trajectory(candidate(), True)
        with self.assertRaisesRegex(RuntimeError, "Number of images exceeds GPU maximum 448"):
            search.search_all([candidate()], True)

    def test_over_limit_no_gpu(self):
        # We can ignore the NUM_IMAGES limit since if we shut off GPU filtering.
        search = make_search(MAX_NUM_IMAGES + 10)
        search.search_all([candidate()], True)
        results = search.get_results(0, 1)
        self.assertEqual(len(results), 1)

        # We still fail if we exceed the MAX_NUM_IMAGE_TIMES limit on the GPU.
        search = make_search(MAX_NUM_IMAGE_TIMES + 1)
        with self.assertRaisesRegex(RuntimeError, "Number of images exceeds GPU maximum 2000"):
            search.search_all([candidate()], True)


if __name__ == "__main__":
    unittest.main()
