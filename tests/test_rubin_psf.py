"""Tests for rendering native Rubin PSF models.

The Rubin API has two rendering entry points with different centering
semantics, and confusing them produces a kernel that looks entirely reasonable
while biasing every position measured with it. Most of these tests exist to pin
that distinction and the placement metadata that goes with it.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kbmod.psf_reprojection import measure_moments, summarize_shape
from kbmod.standardizers.rubin_psf import (
    NativePsfStamp,
    RubinPsfError,
    average_position,
    render_rubin_psf,
)
from utils import BrokenRubinPsf, MockPsfImage, MockRubinPsf, Point2D, lsstGeom

GEOM_PATCH = {"lsst.geom": lsstGeom}


class StubPsf:
    """A PSF model returning a caller-supplied array, for edge cases."""

    def __init__(self, array, x0=0, y0=0):
        self.array = np.asarray(array, dtype=np.float64)
        self.x0, self.y0 = x0, y0

    def computeKernelImage(self, position):
        return MockPsfImage(self.array, self.x0, self.y0)

    computeImage = computeKernelImage


class NoArrayPsf:
    def computeKernelImage(self, position):
        return object()

    computeImage = computeKernelImage


@mock.patch.dict("sys.modules", GEOM_PATCH)
class test_render_rubin_psf(unittest.TestCase):
    def setUp(self):
        self.psf = MockRubinPsf(detector_width=2048, detector_height=4096)
        self.exposure = mock.Mock()
        self.exposure.psf = self.psf

    def test_kernel_mode_is_centered(self):
        """Kernel mode centers the PSF, as convolution requires."""
        stamp = render_rubin_psf(self.exposure, 1024.3, 2048.7, mode="kernel")

        center = (stamp.width - 1) / 2.0
        self.assertAlmostEqual(stamp.centroid_x, center, places=3)
        self.assertAlmostEqual(stamp.centroid_y, center, places=3)
        self.assertEqual(stamp.provenance, "computeKernelImage")
        # Rubin kernel images are origin-centered.
        self.assertEqual(stamp.origin_x, -(stamp.width // 2))
        self.assertEqual(stamp.origin_y, -(stamp.width // 2))

    def test_image_mode_retains_pixel_phase(self):
        """Image mode keeps the fractional position, kernel mode does not.

        This is the distinction that makes the two modes non-interchangeable.
        """
        image_stamp = render_rubin_psf(self.exposure, 1024.3, 2048.7, mode="image")
        kernel_stamp = render_rubin_psf(self.exposure, 1024.3, 2048.7, mode="kernel")

        center = (image_stamp.width - 1) / 2.0
        self.assertAlmostEqual(image_stamp.centroid_x - center, 0.3, places=2)
        self.assertAlmostEqual(image_stamp.centroid_y - center, 0.7, places=2)
        self.assertAlmostEqual(kernel_stamp.centroid_x, center, places=3)
        self.assertEqual(image_stamp.provenance, "computeImage")

    def test_image_mode_placement_recovers_the_evaluation_point(self):
        """Origin plus in-stamp centroid returns the requested position.

        Losing the bounding-box origin is the classic way to introduce a
        one-pixel error here, so the round trip is asserted explicitly.
        """
        for x, y in ((1024.0, 2048.0), (1024.3, 2048.7), (500.9, 900.1)):
            with self.subTest(x=x, y=y):
                stamp = render_rubin_psf(self.exposure, x, y, mode="image")
                self.assertAlmostEqual(stamp.offset_x, x, places=2)
                self.assertAlmostEqual(stamp.offset_y, y, places=2)

    def test_normalization_and_native_sum(self):
        stamp = render_rubin_psf(self.exposure, 1024.0, 2048.0)
        self.assertAlmostEqual(float(stamp.array.sum()), 1.0, places=6)
        self.assertEqual(stamp.array.dtype, np.float32)
        # The mock renders a normalized stamp, so the pre-normalization sum is
        # recorded as ~1 rather than being discarded.
        self.assertAlmostEqual(stamp.native_sum, 1.0, places=6)

    def test_non_unit_input_sum_is_recorded_and_normalized(self):
        """A model that does not sum to 1 is normalized, and the sum is kept."""
        raw = np.ones((5, 5), dtype=np.float64) * 4.0
        stamp = render_rubin_psf(StubPsf(raw), 0.0, 0.0)
        self.assertAlmostEqual(stamp.native_sum, 100.0, places=6)
        self.assertAlmostEqual(float(stamp.array.sum()), 1.0, places=6)

    def test_spatial_variation_is_observable(self):
        """Evaluating at different positions gives different kernels."""
        near = render_rubin_psf(self.exposure, 0.0, 0.0)
        far = render_rubin_psf(self.exposure, 2000.0, 0.0)
        near_shape = summarize_shape(measure_moments(near.array).covariance)
        far_shape = summarize_shape(measure_moments(far.array).covariance)
        self.assertGreater(far_shape.fwhm_major, near_shape.fwhm_major * 1.05)

    def test_asymmetry_survives(self):
        """The model's ellipticity and orientation are preserved."""
        stamp = render_rubin_psf(self.exposure, 1024.0, 3000.0)
        shape = summarize_shape(measure_moments(stamp.array).covariance)
        self.assertGreater(shape.fwhm_major / shape.fwhm_minor, 1.2)
        expected_angle = self.psf.angle_at(1024.0, 3000.0)
        self.assertAlmostEqual(shape.position_angle, expected_angle, places=0)

    def test_diagnostics_are_optional(self):
        plain = render_rubin_psf(self.exposure, 1024.0, 2048.0)
        self.assertEqual(plain.diagnostics, {})

        detailed = render_rubin_psf(self.exposure, 1024.0, 2048.0, diagnostics=True)
        self.assertIn("fwhm_major", detailed.diagnostics)
        self.assertIn("position_angle", detailed.diagnostics)

    def test_accepts_bare_model_or_exposure(self):
        """Both an Exposure and a bare psf component are accepted."""
        from_exposure = render_rubin_psf(self.exposure, 1024.0, 2048.0)
        from_model = render_rubin_psf(self.psf, 1024.0, 2048.0)
        np.testing.assert_array_equal(from_exposure.array, from_model.array)

    def test_average_position(self):
        self.assertEqual(average_position(self.exposure), (1024.0, 2048.0))
        self.assertEqual(average_position(self.psf), (1024.0, 2048.0))
        self.assertIsNone(average_position(mock.Mock(spec=[])))
        # A model whose getAveragePosition raises reports None rather than
        # propagating, so the caller can fall back deliberately.
        self.assertIsNone(average_position(BrokenRubinPsf()))

    def test_guessed_origin_is_labeled_and_refused_in_image_mode(self):
        """Placement must be known, not assumed, when the stamp will be reprojected.

        Kernel mode only needs a centered array, so a guessed origin is
        tolerated and labeled. Image mode exists to preserve exact placement, so
        a guess is refused: reprojecting from it would yield a confidently
        misplaced effective PSF.
        """

        class NoPlacementImage:
            def __init__(self, array):
                self.array = array

        class NoPlacementPsf:
            def __init__(self, array):
                self._array = array

            def computeKernelImage(self, position):
                return NoPlacementImage(self._array)

            computeImage = computeKernelImage

        array = np.ones((5, 5))
        model = NoPlacementPsf(array)

        kernel_stamp = render = render_rubin_psf(model, 10.0, 20.0, mode="kernel")
        self.assertEqual(kernel_stamp.origin_source, "guessed")

        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(model, 10.0, 20.0, mode="image")
        self.assertIn("placement", str(context.exception))

    def test_origin_source_is_bbox_for_a_normal_model(self):
        stamp = render_rubin_psf(self.exposure, 1024.0, 2048.0, mode="image")
        self.assertEqual(stamp.origin_source, "bbox")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            render_rubin_psf(self.exposure, 0.0, 0.0, mode="sideways")


@mock.patch.dict("sys.modules", GEOM_PATCH)
class test_render_rubin_psf_errors(unittest.TestCase):
    """Every one of these must raise rather than return a plausible kernel."""

    def test_missing_model(self):
        exposure = mock.Mock()
        exposure.psf = None
        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(exposure, 0.0, 0.0)
        self.assertIn("psf_fallback_std", str(context.exception))

    def test_model_without_method(self):
        exposure = mock.Mock()
        exposure.psf = mock.Mock(spec=["computeImage"])
        with self.assertRaises(RubinPsfError):
            render_rubin_psf(exposure, 0.0, 0.0, mode="kernel")

    def test_model_raises(self):
        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(BrokenRubinPsf("sensor fault"), 0.0, 0.0)
        self.assertIn("sensor fault", str(context.exception))

    def test_result_without_array(self):
        with self.assertRaises(RubinPsfError):
            render_rubin_psf(NoArrayPsf(), 0.0, 0.0)

    def test_non_finite_values(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                array = np.ones((5, 5))
                array[2, 2] = bad
                with self.assertRaises(RubinPsfError) as context:
                    render_rubin_psf(StubPsf(array), 0.0, 0.0)
                self.assertIn("non-finite", str(context.exception))

    def test_even_width_is_refused(self):
        """Padding an even stamp would move the centroid half a pixel."""
        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(StubPsf(np.ones((4, 4))), 0.0, 0.0)
        self.assertIn("even width", str(context.exception))

    def test_non_square(self):
        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(StubPsf(np.ones((3, 5))), 0.0, 0.0)
        self.assertIn("square", str(context.exception))

    def test_not_2d(self):
        with self.assertRaises(RubinPsfError):
            render_rubin_psf(StubPsf(np.ones((3, 3, 3))), 0.0, 0.0)

    def test_all_zero(self):
        with self.assertRaises(RubinPsfError):
            render_rubin_psf(StubPsf(np.zeros((5, 5))), 0.0, 0.0)

    def test_materially_negative_is_refused(self):
        """A real signed lobe is refused, not flattened.

        Clipping it would change the kernel's normalization and its meaning,
        producing a nonnegative kernel that no longer describes the response.
        """
        array = np.ones((5, 5))
        array[0, 0] = -0.5
        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(StubPsf(array), 0.0, 0.0)
        self.assertIn("negative", str(context.exception))

    def test_tiny_negative_is_clipped_and_reported(self):
        """Interpolation noise at the 1e-12 level is clipped, and recorded."""
        array = np.ones((5, 5))
        array[0, 0] = -1e-12
        stamp = render_rubin_psf(StubPsf(array), 0.0, 0.0)
        self.assertGreater(stamp.clipped_negative_sum, 0.0)
        self.assertTrue(np.all(stamp.array >= 0.0))
        self.assertAlmostEqual(float(stamp.array.sum()), 1.0, places=6)


class test_without_lsst(unittest.TestCase):
    """Behavior when the Science Pipelines are not installed."""

    def test_module_imports_without_lsst(self):
        """Importing KBMOD must never require the Rubin stack."""
        self.assertNotIn("lsst.geom", sys.modules)
        import kbmod.standardizers.rubin_psf  # noqa: F401

        self.assertNotIn("lsst.geom", sys.modules)

    def test_render_reports_missing_lsst_clearly(self):
        psf = MockRubinPsf()
        with self.assertRaises(RubinPsfError) as context:
            render_rubin_psf(psf, 0.0, 0.0)
        message = str(context.exception)
        self.assertIn("lsst.geom", message)
        self.assertIn("psf_fallback_std", message)


if __name__ == "__main__":
    unittest.main()
