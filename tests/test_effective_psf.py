"""Tests for effective PSF generation.

The effective PSF is the native model after the same resampling the science
image went through. The strongest statement available is that generating it must
reproduce what the science path does to a real isolated source, so most of these
tests compare the two directly -- with negative controls, because a comparison
that cannot fail proves nothing.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kbmod.psf_reprojection import (
    DEFAULT_MAX_SUPPORT,
    EffectivePsfError,
    make_effective_psf,
    measure_moments,
)
from kbmod.reprojection import (
    _combine_constituent_psfs,
    _normalized_cross_correlation,
    _effective_psf_for_image,
)
from kbmod.reprojection import reproject_image
from kbmod.reprojection_config import CONSERVE_FLUX_CONFIG, LEGACY_CONFIG
from utils.reproject_harness import (
    elliptical_gaussian_source,
    gaussian_source,
    make_tan_wcs,
    moffat_source,
    render_source,
)

SHAPE = (241, 241)
SCALE = -5.55e-5
CRVAL = (150.0, 2.0)
CENTER = 120


def _wcs(scale=1.0, rot=0.0):
    return make_tan_wcs(crval=CRVAL, pixel_scale_deg=SCALE * scale, shape=SHAPE, rot_deg=rot)


class test_effective_psf_matches_science_path(unittest.TestCase):
    """The generated kernel must equal what reprojection does to a real source."""

    def setUp(self):
        self.original_wcs = _wcs()

    def _compare(
        self,
        common_wcs,
        profile,
        phase=(0.0, 0.0),
        half=40,
        config=CONSERVE_FLUX_CONFIG,
        psf_config=None,
        method="cutout",
    ):
        """Return (residual rms, max/peak, centroid error, flux error, EffectivePsf)."""
        px, py = phase
        source = render_source(SHAPE, CENTER + px, CENTER + py, profile)
        science, _ = reproject_image(source, self.original_wcs, common_wcs, config=config)
        science = np.where(np.isfinite(science), science, 0.0)

        stamp = render_source((2 * half + 1, 2 * half + 1), half + px, half + py, profile)
        effective = make_effective_psf(
            stamp,
            (CENTER - half, CENTER - half),
            self.original_wcs,
            common_wcs,
            config=psf_config or config,
            method=method,
        )

        moments = measure_moments(science)
        radius = effective.support_radius
        y0 = int(round(moments.centroid_y)) - radius
        x0 = int(round(moments.centroid_x)) - radius
        patch = science[y0 : y0 + 2 * radius + 1, x0 : x0 + 2 * radius + 1]
        enclosed = float(patch.sum())
        patch_moments = measure_moments(patch)
        normalized = patch / enclosed

        residual = effective.kernel - normalized
        return dict(
            rms=float(np.sqrt(np.mean(residual**2))),
            peak=float(np.abs(residual).max() / effective.kernel.max()),
            centroid=float(
                np.hypot(
                    effective.centroid_x - (x0 + patch_moments.centroid_x),
                    effective.centroid_y - (y0 + patch_moments.centroid_y),
                )
            ),
            flux=abs(effective.sum_before_normalization - enclosed) / enclosed,
            effective=effective,
        )

    def test_matches_across_transforms_profiles_and_phases(self):
        """The Phase 2 acceptance gates, on generated data.

        rms residual < 1e-4, max residual < 1e-3 of peak, centroid < 0.05 px,
        and the pre-normalization sum within 0.2% of the reprojected flux.
        """
        transforms = {
            "identity": _wcs(),
            "rotate_30": _wcs(rot=30.0),
            "scale_0.85_rot_25": _wcs(scale=0.85, rot=25.0),
            "scale_1.30": _wcs(scale=1.30),
        }
        profiles = {
            "gaussian": gaussian_source(4.0),
            "moffat": moffat_source(4.0, beta=2.5),
            "elliptical": elliptical_gaussian_source(6.0, 3.0, 35.0),
        }
        phases = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.1, 0.9)]

        for transform_name, common_wcs in transforms.items():
            for profile_name, profile in profiles.items():
                for phase in phases:
                    with self.subTest(transform=transform_name, profile=profile_name, phase=phase):
                        result = self._compare(common_wcs, profile, phase=phase)
                        self.assertLess(result["rms"], 1e-4)
                        self.assertLess(result["peak"], 1e-3)
                        self.assertLess(result["centroid"], 0.05)
                        self.assertLess(result["flux"], 0.002)

    def test_undersized_native_stamp_degrades_the_result(self):
        """The caller must supply an adequate native stamp; this is not checked for them.

        `lost_fraction` reports flux lost outside the *support*, not flux the
        caller already truncated when rendering the *native stamp*. A stamp that
        clips a Moffat's wings therefore produces a quietly wrong effective PSF
        with `lost_fraction` near zero. The error shrinks as the stamp grows,
        which is what identifies it as stamp truncation rather than a bug in the
        resampling.
        """
        common_wcs = _wcs(scale=0.85, rot=25.0)
        profile = moffat_source(4.0, beta=2.5)

        errors = {}
        for half in (12, 20, 40):
            errors[half] = self._compare(common_wcs, profile, half=half)["peak"]

        # A tight stamp misses the gate by an order of magnitude ...
        self.assertGreater(errors[12], 1e-3)
        # ... and the error falls away as the stamp is widened.
        self.assertLess(errors[20], errors[12])
        self.assertLess(errors[40], errors[20])
        self.assertLess(errors[40], 1e-3)

    def test_wrong_configuration_is_detected(self):
        """Negative control: the comparison must be able to fail.

        Generating the PSF with a different operator than the science image used
        has to show up, or the agreement above proves nothing.
        """
        common_wcs = _wcs(scale=0.85, rot=25.0)
        mismatched = self._compare(
            common_wcs,
            gaussian_source(4.0),
            config=CONSERVE_FLUX_CONFIG,
            psf_config=CONSERVE_FLUX_CONFIG.evolve(kernel_width=2.6, sample_region_width=6.0),
        )
        self.assertGreater(mismatched["peak"], 1e-3)

    def test_wrong_placement_is_detected(self):
        """Negative control: a misplaced stamp must move the centroid."""
        common_wcs = _wcs(scale=0.85, rot=25.0)
        half = 40
        stamp = render_source((2 * half + 1, 2 * half + 1), half, half, gaussian_source(4.0))

        correct = make_effective_psf(
            stamp,
            (CENTER - half, CENTER - half),
            self.original_wcs,
            common_wcs,
            config=CONSERVE_FLUX_CONFIG,
        )
        shifted = make_effective_psf(
            stamp,
            (CENTER - half + 3, CENTER - half),
            self.original_wcs,
            common_wcs,
            config=CONSERVE_FLUX_CONFIG,
        )
        moved = np.hypot(correct.centroid_x - shifted.centroid_x, correct.centroid_y - shifted.centroid_y)
        self.assertGreater(moved, 1.0)

    def test_full_and_cutout_agree(self):
        """The optimized path must match the full-frame reference."""
        common_wcs = _wcs(scale=0.85, rot=25.0)
        for name, profile in (
            ("gaussian", gaussian_source(4.0)),
            ("moffat", moffat_source(4.0, beta=2.5)),
            ("elliptical", elliptical_gaussian_source(6.0, 3.0, 35.0)),
        ):
            with self.subTest(profile=name):
                full = self._compare(common_wcs, profile, method="full")["effective"]
                cutout = self._compare(common_wcs, profile, method="cutout")["effective"]
                self.assertEqual(full.kernel.shape, cutout.kernel.shape)
                np.testing.assert_allclose(cutout.kernel, full.kernel, rtol=1e-5, atol=1e-7)
                self.assertAlmostEqual(full.centroid_x, cutout.centroid_x, places=6)
                self.assertAlmostEqual(full.centroid_y, cutout.centroid_y, places=6)


class test_support(unittest.TestCase):
    def setUp(self):
        self.original_wcs = _wcs()
        self.common_wcs = _wcs(scale=0.85, rot=25.0)

    def _make(self, profile, half=40, **kwargs):
        stamp = render_source((2 * half + 1, 2 * half + 1), half, half, profile)
        return make_effective_psf(
            stamp,
            (CENTER - half, CENTER - half),
            self.original_wcs,
            self.common_wcs,
            config=CONSERVE_FLUX_CONFIG,
            **kwargs,
        )

    def test_kernel_is_odd_square_normalized_and_finite(self):
        for name, profile in (("gaussian", gaussian_source(4.0)), ("moffat", moffat_source(4.0))):
            with self.subTest(profile=name):
                effective = self._make(profile)
                self.assertEqual(effective.kernel.shape[0], effective.kernel.shape[1])
                self.assertEqual(effective.kernel.shape[0] % 2, 1)
                self.assertAlmostEqual(float(effective.kernel.sum()), 1.0, places=6)
                self.assertTrue(np.all(np.isfinite(effective.kernel)))
                self.assertEqual(effective.kernel.shape[0], 2 * effective.support_radius + 1)

    def test_support_adapts_to_the_profile(self):
        """A wider source needs more support than a narrow one."""
        narrow = self._make(gaussian_source(2.5))
        wide = self._make(gaussian_source(6.0))
        self.assertGreater(wide.support_radius, narrow.support_radius)

    def test_heavy_wings_report_truncation_rather_than_hiding_it(self):
        """A Moffat cannot converge within an affordable support, and says so.

        The kernel is still usable; what matters is that the lost flux is
        reported instead of being normalized away.
        """
        effective = self._make(moffat_source(4.0, beta=2.0))
        self.assertLessEqual(effective.support_radius, DEFAULT_MAX_SUPPORT)
        if effective.warnings:
            self.assertGreater(effective.lost_fraction, 0.0)
            self.assertTrue(any("lost_fraction" in w for w in effective.warnings))

    def test_max_support_is_respected(self):
        effective = self._make(gaussian_source(6.0), max_support=5)
        self.assertLessEqual(effective.support_radius, 5)
        self.assertGreater(effective.lost_fraction, 0.0)

    def test_provenance_records_the_operator(self):
        effective = self._make(gaussian_source(4.0))
        self.assertEqual(effective.provenance["preset"], "conserve_flux")
        self.assertEqual(effective.provenance["config_hash"], CONSERVE_FLUX_CONFIG.hexdigest)
        self.assertIn("reproject_version", effective.provenance)
        self.assertEqual(effective.provenance["method"], "cutout")


class test_effective_psf_errors(unittest.TestCase):
    def setUp(self):
        self.original_wcs = _wcs()
        self.common_wcs = _wcs(scale=0.85, rot=25.0)

    def test_stamp_too_close_to_the_edge(self):
        """A stamp needing padding beyond the frame must raise, not be clipped."""
        half = 20
        stamp = render_source((2 * half + 1, 2 * half + 1), half, half, gaussian_source(3.0))
        with self.assertRaises(EffectivePsfError) as context:
            make_effective_psf(stamp, (0, 0), self.original_wcs, self.common_wcs, config=LEGACY_CONFIG)
        self.assertIn("edge", str(context.exception))

    def test_position_outside_the_common_frame(self):
        """A stamp mapping outside the output frame raises."""
        half = 10
        stamp = render_source((2 * half + 1, 2 * half + 1), half, half, gaussian_source(3.0))
        far = make_tan_wcs(crval=(CRVAL[0] + 5.0, CRVAL[1] + 5.0), pixel_scale_deg=SCALE, shape=SHAPE)
        with self.assertRaises(EffectivePsfError):
            make_effective_psf(stamp, (CENTER, CENTER), self.original_wcs, far, config=LEGACY_CONFIG)

    def test_invalid_stamps(self):
        good = render_source((21, 21), 10, 10, gaussian_source(3.0))
        cases = {
            "not 2d": np.ones((3, 3, 3)),
            "non finite": np.where(np.arange(441).reshape(21, 21) == 0, np.nan, good),
            "all zero": np.zeros((21, 21)),
        }
        for name, stamp in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(EffectivePsfError):
                    make_effective_psf(
                        stamp,
                        (CENTER - 10, CENTER - 10),
                        self.original_wcs,
                        self.common_wcs,
                        config=LEGACY_CONFIG,
                    )

    def test_invalid_method(self):
        stamp = render_source((21, 21), 10, 10, gaussian_source(3.0))
        with self.assertRaises(ValueError):
            make_effective_psf(
                stamp, (CENTER - 10, CENTER - 10), self.original_wcs, self.common_wcs, method="magic"
            )


class test_mosaic_uniformity_guardrail(unittest.TestCase):
    """One kernel per time is only defensible when the constituents agree."""

    def setUp(self):
        self.original_wcs = _wcs()
        self.common_wcs = _wcs(scale=0.85, rot=25.0)

    def _effective(self, profile, half=40):
        stamp = render_source((2 * half + 1, 2 * half + 1), half, half, profile)
        return make_effective_psf(
            stamp,
            (CENTER - half, CENTER - half),
            self.original_wcs,
            self.common_wcs,
            config=CONSERVE_FLUX_CONFIG,
        )

    def test_matching_constituents_are_accepted(self):
        first = self._effective(gaussian_source(4.0))
        second = self._effective(gaussian_source(4.0))
        chosen = _combine_constituent_psfs([first, second], 60000.0, [0, 1])
        np.testing.assert_array_equal(chosen.kernel, first.kernel)

    def test_differing_constituents_are_refused(self):
        """Searching part of a mosaic with another detector's PSF must not happen quietly."""
        narrow = self._effective(gaussian_source(2.5))
        wide = self._effective(gaussian_source(7.0))
        with self.assertRaises(ValueError) as context:
            _combine_constituent_psfs([narrow, wide], 60000.0, [4, 9])
        message = str(context.exception)
        self.assertIn("60000.0", message)
        self.assertIn("index 9", message)
        self.assertIn("cross-correlation", message)

    def test_single_constituent_needs_no_agreement(self):
        only = self._effective(gaussian_source(4.0))
        self.assertIs(_combine_constituent_psfs([only], 60000.0, [3]), only)

    def test_cross_correlation_handles_different_sizes(self):
        small = np.zeros((5, 5))
        small[2, 2] = 1.0
        large = np.zeros((9, 9))
        large[4, 4] = 1.0
        self.assertAlmostEqual(_normalized_cross_correlation(small, large), 1.0, places=10)
        offset = np.zeros((5, 5))
        offset[1, 1] = 1.0
        self.assertLess(_normalized_cross_correlation(small, offset), 0.5)


if __name__ == "__main__":
    unittest.main()
