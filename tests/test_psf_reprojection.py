"""Tests for PSF measurement utilities and the generated-source harness.

These cover the machinery the later phases' acceptance gates depend on. If the
moment or Jacobian code is wrong, every downstream threshold is measuring the
wrong thing, so the tests here check against analytic truth rather than against
recorded outputs.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kbmod.psf_reprojection import (
    encircled_energy_radius,
    local_wcs_jacobian,
    measure_moments,
    predict_geometric_covariance,
    summarize_shape,
    transform_covariance,
)
from kbmod.reprojection_config import CONSERVE_FLUX_CONFIG, LEGACY_CONFIG
from utils.reproject_harness import (
    elliptical_gaussian_source,
    gaussian_source,
    make_tan_wcs,
    moffat_source,
    render_source,
    run_trial,
)

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class test_moments(unittest.TestCase):
    def test_recovers_gaussian_truth(self):
        """Moments of an analytic Gaussian match its parameters."""
        shape = (61, 61)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        sigma_x, sigma_y = 3.0, 2.0
        image = np.exp(-((xx - 30.0) ** 2 / (2 * sigma_x**2) + (yy - 25.0) ** 2 / (2 * sigma_y**2)))

        moments = measure_moments(image)
        self.assertAlmostEqual(moments.centroid_x, 30.0, places=6)
        self.assertAlmostEqual(moments.centroid_y, 25.0, places=6)
        self.assertAlmostEqual(moments.covariance[0, 0], sigma_x**2, places=4)
        self.assertAlmostEqual(moments.covariance[1, 1], sigma_y**2, places=4)
        self.assertAlmostEqual(moments.covariance[0, 1], 0.0, places=6)

    def test_asymmetric_source_is_not_transposed(self):
        """An x/y swap or transpose must be detectable.

        A circular source cannot catch this class of bug, which is why the test
        matrix requires asymmetric profiles.
        """
        shape = (61, 61)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        # Deliberately wider in x than y, and offset differently in each axis.
        image = np.exp(-((xx - 34.0) ** 2 / (2 * 4.0**2) + (yy - 22.0) ** 2 / (2 * 1.5**2)))

        moments = measure_moments(image)
        self.assertGreater(moments.covariance[0, 0], moments.covariance[1, 1])
        self.assertAlmostEqual(moments.centroid_x, 34.0, places=5)
        self.assertAlmostEqual(moments.centroid_y, 22.0, places=5)
        self.assertNotAlmostEqual(moments.centroid_x, moments.centroid_y, places=1)

    def test_ignores_nans(self):
        """NaN pixels are excluded rather than poisoning the result."""
        shape = (41, 41)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        image = np.exp(-((xx - 20.0) ** 2 + (yy - 20.0) ** 2) / (2 * 2.0**2))
        image[0, :] = np.nan  # a clipped edge, far from the source

        moments = measure_moments(image)
        self.assertTrue(np.isfinite(moments.flux))
        self.assertAlmostEqual(moments.centroid_x, 20.0, places=5)

    def test_empty_image_raises(self):
        with self.assertRaises(ValueError):
            measure_moments(np.zeros((5, 5)))

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            measure_moments(np.zeros((3, 3, 3)))


class test_shape_summary(unittest.TestCase):
    def test_matches_gaussian_fwhm(self):
        sigma_major, sigma_minor = 3.0, 1.5
        covariance = np.diag([sigma_major**2, sigma_minor**2])
        shape = summarize_shape(covariance)
        self.assertAlmostEqual(shape.fwhm_major, _FWHM_PER_SIGMA * sigma_major, places=8)
        self.assertAlmostEqual(shape.fwhm_minor, _FWHM_PER_SIGMA * sigma_minor, places=8)
        self.assertAlmostEqual(shape.position_angle, 0.0, places=6)

    def test_position_angle_of_rotated_ellipse(self):
        """A 30-degree rotation of an elongated source is recovered."""
        angle = np.deg2rad(30.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        covariance = rotation @ np.diag([9.0, 1.0]) @ rotation.T
        self.assertAlmostEqual(summarize_shape(covariance).position_angle, 30.0, places=5)

    def test_rejects_non_psd(self):
        with self.assertRaises(ValueError):
            summarize_shape(np.array([[1.0, 5.0], [5.0, 1.0]]))


class test_encircled_energy(unittest.TestCase):
    def test_gaussian_half_light_radius(self):
        """The 50% encircled-energy radius of a Gaussian is 1.1774 sigma."""
        # Measured with a well-resolved source. Encircled energy is computed on
        # a pixel grid, so the radius is quantized: the error against the
        # analytic value falls 5.0% -> 1.1% -> 0.16% -> 0.09% for sigma of 2, 4,
        # 8, 32 px. sigma=8 puts that floor well below the tolerance asserted.
        shape = (113, 113)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        sigma = 8.0
        center = 56.0
        image = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
        radius = encircled_energy_radius(image, center, center, 0.5)
        self.assertAlmostEqual(radius / sigma, np.sqrt(2.0 * np.log(2.0)), places=2)

    def test_moffat_wings_exceed_gaussian(self):
        """Heavier wings need more support at high enclosed fractions.

        This is why a fixed 3-sigma kernel support is not valid for realistic
        PSFs.
        """
        shape = (121, 121)
        gauss = render_source(shape, 60.0, 60.0, gaussian_source(5.0))
        moffat = render_source(shape, 60.0, 60.0, moffat_source(5.0, beta=2.5))
        gauss_r = encircled_energy_radius(gauss, 60.0, 60.0, 0.95)
        moffat_r = encircled_energy_radius(moffat, 60.0, 60.0, 0.95)
        self.assertGreater(moffat_r, gauss_r)

    def test_invalid_fraction(self):
        image = np.ones((5, 5))
        with self.assertRaises(ValueError):
            encircled_energy_radius(image, 2.0, 2.0, 0.0)


class test_wcs_jacobian(unittest.TestCase):
    def test_pixel_scale_recovered(self):
        """The Jacobian determinant recovers the pixel area in arcsec^2."""
        scale_deg = 5.55e-5
        wcs = make_tan_wcs((150.0, 2.0), -scale_deg, (41, 41))
        jacobian = local_wcs_jacobian(wcs, 20.0, 20.0)
        expected_arcsec = scale_deg * 3600.0
        self.assertAlmostEqual(np.sqrt(abs(np.linalg.det(jacobian))), expected_arcsec, places=4)

    def test_rotation_preserves_area(self):
        """Rotation changes the matrix but not its determinant."""
        base = make_tan_wcs((150.0, 2.0), -5.55e-5, (41, 41))
        rotated = make_tan_wcs((150.0, 2.0), -5.55e-5, (41, 41), rot_deg=35.0)
        det_base = abs(np.linalg.det(local_wcs_jacobian(base, 20.0, 20.0)))
        det_rot = abs(np.linalg.det(local_wcs_jacobian(rotated, 20.0, 20.0)))
        # Compared relatively: the Jacobian comes from numerical
        # differentiation, so the residual is finite-difference noise (~1e-8
        # relative), not a property of the rotation.
        np.testing.assert_allclose(det_rot, det_base, rtol=1e-6)

    def test_scale_change_alters_area(self):
        base = make_tan_wcs((150.0, 2.0), -5.55e-5, (41, 41))
        scaled = make_tan_wcs((150.0, 2.0), -5.55e-5 * 1.3, (41, 41))
        det_base = abs(np.linalg.det(local_wcs_jacobian(base, 20.0, 20.0)))
        det_scaled = abs(np.linalg.det(local_wcs_jacobian(scaled, 20.0, 20.0)))
        self.assertAlmostEqual(det_scaled / det_base, 1.3**2, places=6)

    def test_angular_shape_is_frame_independent(self):
        """The same source measured in two frames has one angular shape.

        Pixel widths differ between frames by construction; angular widths must
        not. This is the check that separates a pixel-scale change from blur.
        """
        scale = 1.5
        errors = {}
        for fwhm in (12.0, 24.0):
            size = int(fwhm * 14) | 1
            shape = (size, size)
            center = float(size // 2)
            fine = make_tan_wcs((150.0, 2.0), -5.55e-5, shape)
            coarse = make_tan_wcs((150.0, 2.0), -5.55e-5 * scale, shape)

            fine_image = render_source(
                shape, center, center, elliptical_gaussian_source(fwhm, fwhm / 2, 25.0)
            )
            # The same angular source, rendered natively in the coarser frame.
            coarse_image = render_source(
                shape,
                center,
                center,
                elliptical_gaussian_source(fwhm / scale, fwhm / (2 * scale), 25.0),
            )

            fine_moments = measure_moments(fine_image)
            coarse_moments = measure_moments(coarse_image)

            fine_angular = summarize_shape(
                transform_covariance(fine_moments.covariance, local_wcs_jacobian(fine, center, center))
            )
            coarse_angular = summarize_shape(
                transform_covariance(coarse_moments.covariance, local_wcs_jacobian(coarse, center, center))
            )

            # Pixel widths differ by construction ...
            self.assertGreater(fine_moments.covariance[0, 0], coarse_moments.covariance[0, 0])
            # ... but the angular widths agree.
            ratio = fine_angular.fwhm_major / coarse_angular.fwhm_major
            self.assertAlmostEqual(ratio, 1.0, places=2)
            errors[fwhm] = abs(ratio - 1.0)

        # The residual is pixel discretization, so it must shrink as the source
        # is better resolved. A constant offset would instead indicate a real
        # bias in the Jacobian conversion.
        self.assertLess(errors[24.0], errors[12.0])


class test_harness(unittest.TestCase):
    """The generated-source harness itself, which later gates depend on."""

    def setUp(self):
        self.shape = (81, 81)
        self.original_wcs = make_tan_wcs((150.0, 2.0), -5.55e-5, self.shape)

    def test_render_respects_subpixel_position(self):
        """Rendering integrates over the pixel rather than sampling its center."""
        for phase in (0.0, 0.25, 0.5):
            with self.subTest(phase=phase):
                image = render_source(self.shape, 40.0 + phase, 40.0, gaussian_source(4.0))
                moments = measure_moments(image)
                self.assertAlmostEqual(moments.centroid_x, 40.0 + phase, places=3)
                self.assertAlmostEqual(moments.centroid_y, 40.0, places=3)

    def test_identity_reprojection_preserves_flux_and_position(self):
        result = run_trial(
            gaussian_source(4.0),
            self.original_wcs,
            self.original_wcs,
            40.3,
            39.7,
            config=CONSERVE_FLUX_CONFIG,
        )
        self.assertAlmostEqual(result.flux_ratio, 1.0, places=2)
        self.assertLess(result.centroid_error, 0.05)
        self.assertFalse(result.clipped)

    def test_conserve_flux_preserves_point_source_flux_under_rescaling(self):
        """The flux-mode audit, as an assertion.

        Under a pixel-scale change the flux-conserving operator recovers the
        injected flux, while the legacy surface-brightness operator changes it
        by the pixel-area ratio. This is the evidence behind the recommended
        production preset.
        """
        for scale in (0.77, 1.30):
            with self.subTest(scale=scale):
                common_wcs = make_tan_wcs((150.0, 2.0), -5.55e-5 * scale, self.shape)

                conserved = run_trial(
                    gaussian_source(4.0),
                    self.original_wcs,
                    common_wcs,
                    40.0,
                    40.0,
                    config=CONSERVE_FLUX_CONFIG,
                )
                legacy = run_trial(
                    gaussian_source(4.0),
                    self.original_wcs,
                    common_wcs,
                    40.0,
                    40.0,
                    config=LEGACY_CONFIG,
                )

                self.assertAlmostEqual(conserved.flux_ratio, 1.0, places=2)
                self.assertAlmostEqual(legacy.flux_ratio, 1.0 / scale**2, places=2)

    def test_broadening_separates_geometry_from_interpolation(self):
        """A pixel-scale change is not blur, and the harness must say so.

        Reprojecting to finer pixels widens the source in pixel units by the
        scale ratio. Only the residual above that geometric prediction is
        interpolation.
        """
        scale = 0.8
        common_wcs = make_tan_wcs((150.0, 2.0), -5.55e-5 * scale, self.shape)
        result = run_trial(
            gaussian_source(5.24),
            self.original_wcs,
            common_wcs,
            40.0,
            40.0,
            config=CONSERVE_FLUX_CONFIG,
        )

        pixel_ratio = result.common_shape_pixels.fwhm_major / result.native_shape_pixels.fwhm_major
        geometric_ratio = result.geometric_prediction.fwhm_major / result.native_shape_pixels.fwhm_major
        interpolation_ratio = result.common_shape_pixels.fwhm_major / result.geometric_prediction.fwhm_major

        # Most of the apparent broadening is the pixel-scale change ...
        self.assertAlmostEqual(geometric_ratio, 1.0 / scale, places=2)
        # ... and interpolation contributes a much smaller residual.
        self.assertLess(interpolation_ratio, 1.10)
        self.assertGreater(interpolation_ratio, 1.0)
        self.assertGreater(pixel_ratio, interpolation_ratio)

    def test_rotation_does_not_change_angular_width(self):
        """A pure rotation must not change the source's angular size."""
        rotated_wcs = make_tan_wcs((150.0, 2.0), -5.55e-5, self.shape, rot_deg=30.0)
        result = run_trial(
            gaussian_source(4.0),
            self.original_wcs,
            rotated_wcs,
            40.0,
            40.0,
            config=CONSERVE_FLUX_CONFIG,
        )
        ratio = result.common_shape_angular.fwhm_major / result.native_shape_angular.fwhm_major
        # Only the interpolation floor, no geometric term.
        self.assertLess(abs(ratio - 1.0), 0.10)

    def test_moffat_and_elliptical_profiles_run(self):
        """The non-Gaussian profiles the test matrix requires are usable."""
        for name, profile in (
            ("moffat", moffat_source(4.0)),
            ("elliptical", elliptical_gaussian_source(6.0, 3.0, 40.0)),
        ):
            with self.subTest(profile=name):
                result = run_trial(
                    profile,
                    self.original_wcs,
                    self.original_wcs,
                    40.0,
                    40.0,
                    config=CONSERVE_FLUX_CONFIG,
                )
                self.assertGreater(result.flux_ratio, 0.9)
                self.assertLess(result.centroid_error, 0.05)

    def test_elliptical_source_keeps_its_orientation(self):
        """Position angle survives an identity reprojection."""
        result = run_trial(
            elliptical_gaussian_source(7.0, 3.0, 35.0),
            self.original_wcs,
            self.original_wcs,
            40.0,
            40.0,
            config=CONSERVE_FLUX_CONFIG,
        )
        self.assertAlmostEqual(
            result.common_shape_pixels.position_angle,
            result.native_shape_pixels.position_angle,
            places=0,
        )
        self.assertGreater(result.common_shape_pixels.fwhm_major, result.common_shape_pixels.fwhm_minor)


if __name__ == "__main__":
    unittest.main()
