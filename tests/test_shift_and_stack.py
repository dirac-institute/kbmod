import math
import numpy as np
import unittest

from kbmod.core.shift_and_stack import generate_psi_phi_images


class test_shift_and_stack(unittest.TestCase):
    def test_generate_psi_phi_images_no_psf(self):
        """Test that we can create psi and phi images with a no-op psf."""
        width = 10
        height = 20
        psf_kernel = np.array([[1.0]])  # No-op kernel
        sci = np.array([np.arange(width) for _ in range(height)], dtype=np.single)
        var = np.array([0.1 * (h + 1) * np.ones(width) for h in range(height)], dtype=np.single)

        # Mask a few of the values.
        for y, x in [(3, 4), (15, 3), (1, 1)]:
            sci[y, x] = np.nan
            var[y, x] = np.nan

        psi, phi = generate_psi_phi_images(sci, var, psf_kernel)
        for y in range(height):
            for x in range(width):
                if np.isnan(sci[y, x]):
                    self.assertTrue(np.isnan(psi[y, x]))
                    self.assertTrue(np.isnan(phi[y, x]))
                else:
                    self.assertAlmostEqual(psi[y, x], x / (0.1 * (y + 1)), places=5)
                    self.assertAlmostEqual(phi[y, x], 1.0 / (0.1 * (y + 1)), places=5)

    def test_generate_psi_phi_images_given_psf(self):
        """Test that we can create psi and phi images with a given psf."""
        sci = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, np.nan, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=np.single,
        )
        var = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, np.nan, 0.2],
                [0.1, 0.1, 0.1, 0.1],
            ],
            dtype=np.single,
        )
        psf_kernel = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.6, 0.1],
                [0.0, 0.1, 0.0],
            ]
        )

        # Compare to manually computed psi and phi values.
        psi_expected = np.array(
            [
                [3.75, 11.66666, 20.0, 29.375],
                [25.0, 30.0, np.nan, 43.75],
                [73.75, 82.77777, 100.0, 99.375],
            ],
            dtype=np.single,
        )
        phi_expected = np.array(
            [
                [3.9473684, 3.9487179, 4.0, 3.94736842],
                [2.1025641, 2.1025641, np.nan, 2.10526316],
                [3.9473684, 3.9487179, 4.0, 3.94736842],
            ],
            dtype=np.single,
        )
        psi, phi = generate_psi_phi_images(sci, var, psf_kernel)

        self.assertTrue(np.allclose(psi, psi_expected, rtol=0.001, atol=0.001, equal_nan=True))
        self.assertTrue(np.allclose(phi, phi_expected, rtol=0.001, atol=0.001, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
