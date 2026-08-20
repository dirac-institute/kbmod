"""Tests for the explicit adaptive reprojection configuration.

These are deliberately "canary" tests. KBMOD's scientific results depend on
`reproject_adaptive` options that were previously inherited from the installed
`reproject` release, and those defaults have changed across releases. The tests
here fail loudly when the library's behavior drifts, rather than letting a
silent change alter recovered fluxes.
"""

import inspect
import unittest

import numpy as np
import reproject
from astropy.wcs import WCS

from kbmod.reprojection import reproject_image
from kbmod.reprojection_config import (
    CONSERVE_FLUX_CONFIG,
    LEGACY_CONFIG,
    PRESETS,
    AdaptiveReprojectionConfig,
)

# `reproject_adaptive` parameters that do not affect the numerical result. They
# are call-site plumbing (what to reproject, where to put it, how to schedule the
# work), so the configuration deliberately does not model them. Anything in the
# signature but absent from both this set and the config is a new option KBMOD
# would be inheriting silently.
_PLUMBING_PARAMS = {
    "input_data",
    "output_projection",
    "shape_out",
    "hdu_in",
    "output_array",
    "output_footprint",
    "return_footprint",
    "block_size",
    "parallel",
    "return_type",
    "dask_method",
}

# Options where KBMOD deliberately differs from the library default, with the
# value KBMOD requires. Restoring a library default here would change results.
_INTENTIONAL_DIVERGENCES = {
    "bad_value_mode": "ignore",
    "roundtrip_coords": False,
}


def make_tan_wcs(crval, cd_scale, shape, rot_deg=0.0):
    """Build a deterministic TAN WCS for tests."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    rot = np.deg2rad(rot_deg)
    wcs.wcs.cd = cd_scale * np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    wcs.array_shape = shape
    return wcs


class test_reprojection_config(unittest.TestCase):
    def test_models_every_numerical_option(self):
        """Every numerically relevant `reproject_adaptive` option is modeled.

        If `reproject` gains an option that changes results, KBMOD would inherit
        it silently. This fails when that happens, so the option can be reviewed
        and either modeled or explicitly classified as plumbing.
        """
        signature = inspect.signature(reproject.reproject_adaptive)
        modeled = set(LEGACY_CONFIG.as_kwargs())
        unaccounted = set(signature.parameters) - modeled - _PLUMBING_PARAMS

        self.assertEqual(
            unaccounted,
            set(),
            f"reproject {reproject.__version__} exposes option(s) {sorted(unaccounted)} that KBMOD "
            "neither configures nor classifies as plumbing. Decide whether each affects the "
            "numerical result, then add it to AdaptiveReprojectionConfig or _PLUMBING_PARAMS.",
        )

    def test_config_options_all_exist_in_library(self):
        """Every modeled option is really a `reproject_adaptive` parameter."""
        signature = inspect.signature(reproject.reproject_adaptive)
        stale = set(LEGACY_CONFIG.as_kwargs()) - set(signature.parameters)
        self.assertEqual(
            stale,
            set(),
            f"AdaptiveReprojectionConfig models option(s) {sorted(stale)} that reproject "
            f"{reproject.__version__} does not accept.",
        )

    def test_legacy_matches_library_defaults_except_known_divergences(self):
        """Legacy preset tracks library defaults except where KBMOD overrides them.

        This is the drift detector. A library default changing under us is
        exactly the failure mode that motivated making these options explicit.
        """
        signature = inspect.signature(reproject.reproject_adaptive)
        legacy = LEGACY_CONFIG.as_kwargs()

        for name, value in legacy.items():
            with self.subTest(option=name):
                library_default = signature.parameters[name].default
                if name in _INTENTIONAL_DIVERGENCES:
                    self.assertEqual(
                        value,
                        _INTENTIONAL_DIVERGENCES[name],
                        f"KBMOD's intentional override of '{name}' changed.",
                    )
                    self.assertNotEqual(
                        value,
                        library_default,
                        f"'{name}' is recorded as an intentional divergence but now matches the "
                        f"reproject {reproject.__version__} default. Update _INTENTIONAL_DIVERGENCES.",
                    )
                else:
                    self.assertEqual(
                        value,
                        library_default,
                        f"reproject {reproject.__version__} changed its default for '{name}' "
                        f"({library_default!r}) away from the value KBMOD has been using "
                        f"({value!r}). Results will shift unless this is reviewed.",
                    )

    def test_legacy_preserves_historical_call(self):
        """The two options KBMOD always passed explicitly keep their values."""
        self.assertEqual(LEGACY_CONFIG.bad_value_mode, "ignore")
        self.assertFalse(LEGACY_CONFIG.roundtrip_coords)
        self.assertFalse(LEGACY_CONFIG.conserve_flux)

    def test_presets_and_provenance(self):
        """Presets are identifiable and provenance records the library version."""
        self.assertEqual(LEGACY_CONFIG.preset_name, "legacy")
        self.assertEqual(CONSERVE_FLUX_CONFIG.preset_name, "conserve_flux")
        self.assertEqual(LEGACY_CONFIG.evolve(kernel_width=2.0).preset_name, "custom")
        self.assertIn("legacy", PRESETS)

        provenance = LEGACY_CONFIG.provenance
        self.assertEqual(provenance["reproject_version"], reproject.__version__)
        self.assertEqual(provenance["config_hash"], LEGACY_CONFIG.hexdigest)
        self.assertEqual(provenance["bad_value_mode"], "ignore")

        # The hash must separate configurations that produce different numbers.
        self.assertNotEqual(LEGACY_CONFIG.hexdigest, CONSERVE_FLUX_CONFIG.hexdigest)
        self.assertEqual(LEGACY_CONFIG.hexdigest, AdaptiveReprojectionConfig().hexdigest)

    def test_frozen(self):
        """The configuration cannot be mutated in place."""
        with self.assertRaises(Exception):
            LEGACY_CONFIG.conserve_flux = True

    def test_numerical_canary(self):
        """Pin the legacy operator's output on a fixed input.

        A signature-level check cannot catch a change inside the resampling
        implementation. These numbers were recorded with reproject 0.21.0. If
        this fails while the signature tests pass, the library's numerics moved:
        re-measure deliberately and record why, rather than loosening tolerance.
        """
        shape = (21, 21)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        image = np.exp(-((xx - 10.3) ** 2 + (yy - 9.7) ** 2) / (2 * 1.5**2)).astype(np.float32)

        original_wcs = make_tan_wcs((10.0, 20.0), -1e-4, shape)
        common_wcs = make_tan_wcs((10.0, 20.0), -1e-4, shape, rot_deg=15.0)

        result, footprint = reproject_image(image, original_wcs, common_wcs, config=LEGACY_CONFIG)

        self.assertAlmostEqual(float(np.nansum(result)), 14.14274883, places=5)
        self.assertEqual(int(np.isnan(result).sum()), 120)
        self.assertEqual(int(footprint.sum()), 321)

        expected_center = np.array(
            [
                [0.5942073, 0.77860403, 0.69678706],
                [0.6264974, 0.81606895, 0.73512053],
                [0.45095086, 0.5915088, 0.5287998],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result[9:12, 9:12], expected_center, rtol=1e-6)

    def test_conserve_flux_applies_pixel_area_jacobian(self):
        """`conserve_flux` rescales by the pixel-area ratio, and only then.

        Two properties matter for the flux-mode decision. Under a pure rotation
        the presets must agree, because no pixel area changes. Under a pure
        scale change they must differ by exactly the area ratio.
        """
        shape = (21, 21)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        image = np.exp(-((xx - 10.0) ** 2 + (yy - 10.0) ** 2) / (2 * 1.5**2)).astype(np.float32)
        original_wcs = make_tan_wcs((10.0, 20.0), -1e-4, shape)

        # Pure rotation: identical pixel area, so the presets must agree exactly.
        rotated_wcs = make_tan_wcs((10.0, 20.0), -1e-4, shape, rot_deg=30.0)
        legacy_rot, _ = reproject_image(image, original_wcs, rotated_wcs, config=LEGACY_CONFIG)
        conserve_rot, _ = reproject_image(image, original_wcs, rotated_wcs, config=CONSERVE_FLUX_CONFIG)
        np.testing.assert_allclose(legacy_rot, conserve_rot, rtol=1e-6)

        # Pure scale change: outputs differ by the pixel-area ratio.
        scale = 1.3
        scaled_wcs = make_tan_wcs((10.0, 20.0), -1e-4 * scale, shape)
        legacy_scaled, _ = reproject_image(image, original_wcs, scaled_wcs, config=LEGACY_CONFIG)
        conserve_scaled, _ = reproject_image(image, original_wcs, scaled_wcs, config=CONSERVE_FLUX_CONFIG)

        usable = np.isfinite(legacy_scaled) & (np.abs(legacy_scaled) > 1e-6)
        self.assertGreater(usable.sum(), 10, "not enough unclipped pixels to compare")
        ratio = conserve_scaled[usable] / legacy_scaled[usable]
        np.testing.assert_allclose(ratio, scale**2, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
