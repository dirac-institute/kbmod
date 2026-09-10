import unittest
from unittest import mock

import numpy as np
from astropy.table import Table

from utils import MockButler, DatasetId, dafButler
from kbmod.snr_utils import SNR_COLUMNS, epoch_snr, predicted_snr, stack_snr


def make_table(mags_zp, sky_noise=1.0, psf_area=4.0, bands=None, visits=None, times=None):
    """Build a minimal ImageCollection-like table with per-visit depth."""
    n = len(mags_zp)
    return Table(
        {
            "mjd_mid": np.asarray(times if times is not None else np.arange(n, dtype=float)),
            "visit": np.asarray(visits if visits is not None else np.arange(n)),
            "detector": np.zeros(n, dtype=int),
            "band": np.asarray(bands if bands is not None else ["i"] * n),
            "zeroPoint": np.asarray(mags_zp, dtype=float),
            "skyNoise": np.full(n, sky_noise, dtype=float),
            "psfArea": np.full(n, psf_area, dtype=float),
        }
    )


class TestEpochSNR(unittest.TestCase):
    """The per-epoch photometry model."""

    def test_matches_closed_form(self):
        snr = epoch_snr(mag=25.0, zero_point=31.0, sky_noise=2.0, psf_area=9.0)
        expected = 10 ** (-0.4 * (25.0 - 31.0)) / (2.0 * 3.0)
        self.assertAlmostEqual(float(snr), expected)

    def test_source_at_zero_point_has_unit_flux(self):
        # mag == zeroPoint is flux 1, so SNR is just 1 / (skyNoise * sqrt(psfArea)).
        snr = epoch_snr(mag=31.0, zero_point=31.0, sky_noise=4.0, psf_area=25.0)
        self.assertAlmostEqual(float(snr), 1.0 / 20.0)

    def test_brighter_is_higher_snr(self):
        snrs = epoch_snr(np.array([24.0, 25.0, 26.0]), 31.0, 1.0, 4.0)
        self.assertTrue(np.all(np.diff(snrs) < 0))
        # 1 mag is a factor 10**0.4 in flux, and so in SNR.
        self.assertAlmostEqual(float(snrs[0] / snrs[1]), 10**0.4)

    def test_deeper_band_gives_more_snr(self):
        # z-band sky is ~2x noisier; the same source is ~2x weaker there.
        deep = epoch_snr(26.0, 31.0, sky_noise=1.0, psf_area=4.0)
        shallow = epoch_snr(26.0, 31.0, sky_noise=2.0, psf_area=4.0)
        self.assertAlmostEqual(float(deep / shallow), 2.0)


class TestStackSNR(unittest.TestCase):
    """Epochs combine in quadrature."""

    def test_quadrature(self):
        self.assertAlmostEqual(stack_snr([3.0, 4.0]), 5.0)

    def test_sqrt_n_growth(self):
        # N identical epochs stack to sqrt(N) times a single epoch.
        for n in (1, 4, 9, 16):
            self.assertAlmostEqual(stack_snr([2.0] * n), 2.0 * np.sqrt(n))

    def test_empty(self):
        self.assertEqual(stack_snr([]), 0.0)


class TestPredictedSNR(unittest.TestCase):
    """The ImageCollection-facing entry point."""

    def test_table_and_cumulative_agree(self):
        tbl = make_table([31.0] * 4)
        result = predicted_snr(tbl, mag=26.0)
        scalar = predicted_snr(tbl, mag=26.0, cumulative=True)

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result.meta["cumulative_snr"], scalar)
        self.assertAlmostEqual(stack_snr(result["snr"]), scalar)
        # Four identical epochs: exactly 2x a single epoch.
        self.assertAlmostEqual(scalar, 2.0 * float(result["snr"][0]))

    def test_running_cumulative_is_in_time_order(self):
        tbl = make_table([31.0] * 5, times=[4.0, 0.0, 3.0, 1.0, 2.0])
        result = predicted_snr(tbl, mag=26.0)

        self.assertTrue(np.all(np.diff(result["mjd_mid"]) > 0))
        # A steady source builds up monotonically, as sqrt(N).
        self.assertTrue(np.all(np.diff(result["cumulative_snr"]) > 0))
        single = float(result["snr"][0])
        expected = single * np.sqrt(np.arange(1, 6))
        np.testing.assert_allclose(result["cumulative_snr"], expected)
        self.assertAlmostEqual(result["cumulative_snr"][-1], result.meta["cumulative_snr"])

    def test_requires_depth_columns(self):
        tbl = make_table([31.0] * 2)
        tbl.remove_column("psfArea")
        with self.assertRaises(ValueError) as ctx:
            predicted_snr(tbl, mag=26.0)
        self.assertIn("psfArea", str(ctx.exception))

    def test_selection_by_visit_and_time(self):
        tbl = make_table([31.0] * 4, visits=[10, 11, 12, 13], times=[0.0, 1.0, 2.0, 3.0])

        by_visit = predicted_snr(tbl, mag=26.0, visits=[11, 13])
        self.assertEqual(sorted(by_visit["visit"].tolist()), [11, 13])

        by_time = predicted_snr(tbl, mag=26.0, times=[1.0, 2.0])
        self.assertEqual(sorted(by_time["visit"].tolist()), [11, 12])

        # Selecting fewer epochs can only lower the stacked SNR.
        self.assertLess(by_visit.meta["cumulative_snr"], predicted_snr(tbl, 26.0, cumulative=True))

    def test_no_matching_rows(self):
        tbl = make_table([31.0] * 3)
        self.assertEqual(predicted_snr(tbl, mag=26.0, visits=[999], cumulative=True), 0.0)

    def test_one_row_per_visit(self):
        # Two detectors of the same visit share an obstime; a source lands on
        # one of them, so the epoch must not be counted twice.
        tbl = make_table([31.0] * 4, visits=[10, 10, 11, 11], times=[0.0, 0.0, 1.0, 1.0])

        deduped = predicted_snr(tbl, mag=26.0, cumulative=True)
        doubled = predicted_snr(tbl, mag=26.0, cumulative=True, one_row_per_visit=False)

        self.assertEqual(len(predicted_snr(tbl, mag=26.0)), 2)
        self.assertAlmostEqual(doubled / deduped, np.sqrt(2.0))


class TestBandHandling(unittest.TestCase):
    """Magnitude is band-specific, and so is depth."""

    def test_per_band_mapping(self):
        tbl = make_table([31.0] * 3, bands=["i", "r", "z"])
        result = predicted_snr(tbl, mag={"i": 26.0, "r": 26.5, "z": 25.5})

        by_band = dict(zip(result["band"], result["mag"]))
        self.assertEqual(by_band, {"i": 26.0, "r": 26.5, "z": 25.5})

    def test_single_mag_rejected_across_bands(self):
        tbl = make_table([31.0] * 3, bands=["i", "r", "z"])
        with self.assertRaises(ValueError) as ctx:
            predicted_snr(tbl, mag=26.0)
        self.assertIn("band", str(ctx.exception).lower())

    def test_single_mag_allowed_within_one_band(self):
        tbl = make_table([31.0] * 3, bands=["i"] * 3)
        self.assertGreater(predicted_snr(tbl, mag=26.0, cumulative=True), 0.0)

    def test_missing_band_in_mapping(self):
        tbl = make_table([31.0] * 2, bands=["i", "z"])
        with self.assertRaises(ValueError) as ctx:
            predicted_snr(tbl, mag={"i": 26.0})
        self.assertIn("z", str(ctx.exception))

    def test_mean_mag_overstates_snr(self):
        """Regression pin: collapsing colours to a mean magnitude biases high.

        Recorded in the ops analysis - switching from a mean magnitude to the
        per-band magnitude lowered predicted SNR by ~5.5% and moved the
        genuine-object calibration from 0.85 to 0.90.
        """
        bands = ["i", "r", "z"]
        per_band = {"i": 26.55, "r": 26.82, "z": 26.44}
        mean_mag = float(np.mean(list(per_band.values())))

        # Realistic relative depth: z is the shallowest band.
        tbl = make_table([31.0] * 3, bands=bands)
        tbl["skyNoise"] = np.array([1.0, 1.2, 2.0])

        correct = predicted_snr(tbl, mag=per_band, cumulative=True)
        naive = predicted_snr(tbl, mag=np.full(3, mean_mag), cumulative=True)

        self.assertLess(correct, naive)
        self.assertLess(abs(correct / naive - 1.0), 0.15)

    def test_shallow_band_contributes_little(self):
        """z-band epochs add almost nothing next to i/r, as the depth predicts."""
        tbl = make_table([31.0] * 2, bands=["i", "z"])
        tbl["skyNoise"] = np.array([1.0, 2.0])

        result = predicted_snr(tbl, mag={"i": 26.5, "z": 26.5})
        i_snr, z_snr = result["snr"][result["band"] == "i"][0], result["snr"][result["band"] == "z"][0]
        self.assertAlmostEqual(float(i_snr / z_snr), 2.0)

        # Contributions add as squares, so 2x the SNR is 4x the contribution.
        i_share = i_snr**2 / (i_snr**2 + z_snr**2)
        self.assertAlmostEqual(float(i_share), 0.8)


@mock.patch.dict("sys.modules", {"lsst.daf.butler": dafButler})
class TestPredictedSNRFromImageCollection(unittest.TestCase):
    """The utility reads a real ImageCollection, not just a bare table."""

    def test_reads_butler_backed_collection(self):
        from kbmod import ImageCollection, Standardizer

        butler = MockButler("/far/far/away", zero_point=31.0, sky_noise=2.0, psf_area=9.0)
        stds = [Standardizer.get(DatasetId(i, fill_metadata=True), butler=butler) for i in (7, 8)]
        ic = ImageCollection.fromStandardizers(stds)

        for col in SNR_COLUMNS:
            self.assertIn(col, ic.data.colnames)

        result = predicted_snr(ic, mag=26.0)
        expected = 10 ** (-0.4 * (26.0 - 31.0)) / (2.0 * 3.0)
        np.testing.assert_allclose(result["snr"], expected)
        self.assertAlmostEqual(result.meta["cumulative_snr"], expected * np.sqrt(len(result)))


if __name__ == "__main__":
    unittest.main()
