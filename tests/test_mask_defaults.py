import importlib.util
import sys
import types
import unittest
from unittest import mock

from astropy.io import fits
import numpy as np

from kbmod.standardizers import (
    ButlerStandardizer,
    ButlerStandardizerConfig,
    KBMODV0_5,
    KBMODV0_5Config,
    KBMODV1,
    KBMODV1Config,
)
from kbmod.standardizers.mask_defaults import IC_BUILD_MASK_FLAGS, LEGACY_FITS_MASK_BITS


class TestMaskDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the CLI without a Rubin installation; mask conversion stays real.
        lsst = types.ModuleType("lsst")
        lsst.daf = types.ModuleType("lsst.daf")
        lsst.daf.butler = types.ModuleType("lsst.daf.butler")
        lsst.daf.butler.Butler = mock.Mock()
        modules = {"lsst": lsst, "lsst.daf": lsst.daf, "lsst.daf.butler": lsst.daf.butler}
        spec = importlib.util.find_spec("kbmod_cmdline.kbmod_build_ic")
        cls.driver = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(cls.driver)

    def build_config(self, *options):
        argv = ["kbmod_build_ic", "repo", "image", "--collections", "collection", *options]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(self.driver, "ingest_collection") as ingest,
        ):
            self.driver.main()
        return ingest.call_args.args[3]

    def butler_mask(self, config):
        # Deliberately use different bit positions from the legacy FITS map.
        planes = {
            "BAD": 0,
            "CR": 2,
            "INTRP": 4,
            "UNMASKEDNAN": 7,
            "DETECTED_NEGATIVE": 9,
            "SPIKE": 1,
            "INJECTED": 3,
            "INJECTED_TEMPLATE": 6,
            "DETECTED": 12,
        }
        pixels = np.array([[1 << bit for bit in planes.values()] + [0]])
        std = ButlerStandardizer.__new__(ButlerStandardizer)
        std.config = config
        std._naxis2, std._naxis1 = pixels.shape
        std.exp = types.SimpleNamespace(
            mask=types.SimpleNamespace(array=pixels, getMaskPlaneDict=lambda: planes)
        )
        return std.standardizeMaskImage()[0].ravel()

    def legacy_mask(self, std_class, config, pixels):
        std = std_class.__new__(std_class)
        std.config = config
        std.hdulist = fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(), fits.ImageHDU(np.array([pixels]), name="MASK")]
        )
        self.addCleanup(std.hdulist.close)
        return std._standardizeMask().ravel()

    def test_butler_defaults_preserved(self):
        config = ButlerStandardizerConfig(grow_mask=False)
        np.testing.assert_array_equal(
            self.butler_mask(config),
            [True, True, False, False, False, False, False, False, False, False],
        )

    def test_legacy_defaults_preserved(self):
        # Pixel i + 1 holds legacy bit i; the first pixel has no flags.
        pixels = [0] + [1 << bit for bit in range(16)]
        expected = np.zeros(17, dtype=bool)
        expected[[1, 5, 8, 9, 16]] = True  # BAD, EDGE, SUSPECT, NO_DATA, UNMASKEDNAN
        for std_class in (KBMODV0_5, KBMODV1):
            with self.subTest(standardizer=std_class.name):
                config = std_class.configClass(grow_mask=False)
                np.testing.assert_array_equal(self.legacy_mask(std_class, config, pixels), expected)

    def test_builder_mask_policy(self):
        # Negative detections/spikes are masked; injection flags alone are not.
        np.testing.assert_array_equal(
            self.butler_mask(self.build_config()),
            [True, True, True, True, True, True, False, False, False, False],
        )

    def test_builder_override_and_fresh_defaults(self):
        custom = self.build_config("--mask_flags", "INJECTED")
        expected = np.zeros(10, dtype=bool)
        expected[6] = True
        np.testing.assert_array_equal(self.butler_mask(custom), expected)

        first = self.build_config()
        first["mask_flags"].clear()
        second = self.build_config()
        self.assertEqual(second["mask_flags"], list(IC_BUILD_MASK_FLAGS))

    def test_standardizer_overrides(self):
        config = ButlerStandardizerConfig(grow_mask=False, mask_flags=["INTRP"])
        expected = np.zeros(10, dtype=bool)
        expected[2] = True
        np.testing.assert_array_equal(self.butler_mask(config), expected)
        config["mask_flags"] = []
        self.assertFalse(self.butler_mask(config).any())

        for std_class in (KBMODV0_5, KBMODV1):
            with self.subTest(standardizer=std_class.name):
                config = std_class.configClass(
                    grow_mask=False, mask_flags=["CUSTOM"], bit_flag_map={"CUSTOM": 1 << 17}
                )
                np.testing.assert_array_equal(
                    self.legacy_mask(std_class, config, [0, 1, 1 << 17]), [False, False, True]
                )

    def test_legacy_maps_are_independent(self):
        with mock.patch.dict(KBMODV1Config.bit_flag_map, {"BAD": 1 << 17}):
            self.assertEqual(KBMODV0_5Config.bit_flag_map["BAD"], 1)
            self.assertEqual(LEGACY_FITS_MASK_BITS["BAD"], 1)
        self.assertIsNot(KBMODV0_5Config.mask_flags, KBMODV1Config.mask_flags)


if __name__ == "__main__":
    unittest.main()
