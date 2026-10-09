from astropy.table import Table
import json
import numpy as np
import tempfile
import unittest
from uuid import uuid4

from pathlib import Path

from kbmod.core.image_stack_py import LayeredImagePy
from kbmod.image_collection import ImageCollection
from kbmod.core.psf import PSF
from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod.util_functions import (
    get_matched_obstimes,
    load_deccam_layered_image,
    make_manual_tracklets,
    mjd_to_day,
    unravel_results,
)


class test_util_functions(unittest.TestCase):
    def setUp(self):
        trj_list = []
        self.num_images = 10
        self.num_trjs = 11

        for i in range(self.num_trjs):
            trj = Trajectory(
                x=i,
                y=i + 0,
                vx=i - 2.0,
                vy=i + 5.0,
                flux=5.0 * i + 1.0,  # Flux must be > 1
                lh=100.0 + i,
                obs_count=self.num_images,
            )
            trj_list.append(trj)

        res = Results.from_trajectories(trj_list)

        # create an "image collection" with all required fields
        ic = Table()
        ic["mjd_mid"] = [60000 + i for i in range(self.num_images)]
        ic["band"] = ["g" for _ in range(self.num_images)]
        # Upstream instrumental zero points deliberately differ from output units.
        ic["zeroPoint"] = np.linspace(32.0, 32.4, self.num_images)
        ic["config"] = [json.dumps({"zero_point": 31.0}) for _ in range(self.num_images)]

        res.table.meta["mjd_mid"] = ic["mjd_mid"]
        obs_count = [self.num_images for _ in range(self.num_trjs)]
        res.table["obs_count"] = obs_count
        res.table["img_ra"] = [
            np.array([j + (i * 0.1) for j in range(self.num_images)]) for i in range(self.num_trjs)
        ]
        res.table["img_dec"] = [
            np.array([j + (i * 0.1) for j in range(self.num_images)]) for i in range(self.num_trjs)
        ]

        self.ic = ic
        self.res = res

    def test_get_matched_obstimes(self):
        obstimes = [1.0, 2.0, 3.0, 4.0, 6.0, 7.5, 9.0, 10.1]
        query_times = [-1.0, 0.999999, 1.001, 1.1, 1.999, 6.001, 7.499, 7.5, 10.099999, 10.10001, 20.0]
        matched_inds = get_matched_obstimes(obstimes, query_times, threshold=0.01)
        expected = [-1, 0, 0, -1, 1, 4, 5, 5, 7, 7, -1]
        self.assertTrue(np.array_equal(matched_inds, expected))

    def test_mjd_to_day(self):
        self.assertEqual(mjd_to_day(60481.04237269), "2024-06-20")
        self.assertEqual(mjd_to_day(60500), "2024-07-09")
        self.assertEqual(mjd_to_day(58100.5), "2017-12-13")

    def test_load_deccam_layered_image(self):
        base_path = Path(__file__).parent.parent
        img_path = base_path / "data" / "demo_image.fits"
        img = load_deccam_layered_image(img_path, PSF.make_gaussian_kernel(1.0))

        self.assertTrue(isinstance(img, LayeredImagePy))
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)

    def test_unravel_results(self):
        df = unravel_results(self.res, self.ic)
        self.assertEqual(len(df), (self.num_images * self.num_trjs))
        expected = np.repeat(31.0 - 2.5 * np.log10(self.res["flux"]), self.num_images)
        np.testing.assert_allclose(df["magnitude"], expected)

        obs_count = self.res.table["obs_count"]
        obs_count[int(self.num_images / 2)] = self.num_images - 1
        valids = [[True] * self.num_images for _ in range(self.num_trjs)]
        valids[int(self.num_images / 2)][-1] = False  # make one observation invalid
        res2 = self.res
        res2.table["obs_valid"] = valids
        res2.table["obs_count"] = obs_count

        df2 = unravel_results(res2, self.ic)
        self.assertEqual(len(df2), (self.num_images * self.num_trjs) - 1)

        df2 = unravel_results(self.res, self.ic, first_and_last=True)
        self.assertEqual(len(df2), self.num_trjs * 2)

    def test_unravel_results_configured_zero_point(self):
        self.ic["config"] = [json.dumps({"zero_point": 31.4})] * self.num_images
        self.ic.remove_column("zeroPoint")  # A summary is not needed for calibrated fluxes.
        df = unravel_results(self.res, self.ic)
        expected = np.repeat(31.4 - 2.5 * np.log10(self.res["flux"]), self.num_images)
        np.testing.assert_allclose(df["magnitude"], expected)

    def test_unravel_results_packed_collection_roundtrip(self):
        metadata = self.ic.copy()
        metadata["std_idx"] = np.arange(self.num_images)
        ic = ImageCollection(metadata, validate=False)
        ic.pack()
        self.assertNotIn("config", ic.data.colnames)
        self.assertNotIn("band", ic.data.colnames)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.ecsv"
            ic.data.write(path, format="ascii.ecsv")
            restored = ImageCollection(Table.read(path, format="ascii.ecsv"), validate=False)
            for collection in (ic, restored):
                columns = list(collection.data.colnames)
                original_meta = collection.data.meta.copy()
                df = unravel_results(self.res, collection)
                expected = np.repeat(31.0 - 2.5 * np.log10(self.res["flux"]), self.num_images)
                np.testing.assert_allclose(df["magnitude"], expected)
                self.assertTrue((df["band"] == "g").all())
                self.assertEqual(collection.data.colnames, columns)
                self.assertEqual(collection.data.meta, original_meta)
                self.assertTrue(collection.is_packed)

    def test_unravel_results_rejects_unknown_or_mixed_units(self):
        for config in (None, "{}", "not-json", '{"zero_point": null}', '{"zero_point": NaN}'):
            with self.subTest(config=config):
                metadata = self.ic.copy()
                if config is None:
                    metadata.remove_column("config")
                else:
                    metadata["config"] = [config] * self.num_images
                with self.assertRaises(ValueError):
                    unravel_results(self.res, metadata)
        self.ic["config"] = [json.dumps({"zero_point": 31.0 + (i % 2)}) for i in range(self.num_images)]
        with self.assertRaisesRegex(ValueError, "Mixed output zero points"):
            unravel_results(self.res, self.ic)

    def test_unravel_results_explicit_zero_point(self):
        self.ic.remove_column("config")
        # Preserve all historical positional arguments; the calibration override
        # is keyword-only and comes from known result units, never the summary.
        df = unravel_results(self.res, self.ic, "500", "test", True, zero_point=31.4)
        expected = np.repeat(31.4 - 2.5 * np.log10(self.res["flux"]), 2)
        np.testing.assert_allclose(df["magnitude"], expected)
        self.assertTrue((df["obscode"] == "500").all())
        self.assertEqual(df["id"].iloc[0], "test-0-0")
        for zp in (np.nan, np.inf, -np.inf):
            with self.subTest(zp=zp), self.assertRaises(ValueError):
                unravel_results(self.res, self.ic, zero_point=zp)

    def test_make_manual_tracklets(self):
        df = unravel_results(self.res, self.ic)
        tracklets, trk2det = make_manual_tracklets(df)

        self.assertEqual(len(tracklets), (self.num_trjs * (self.num_images - 1)))
        self.assertEqual(len(trk2det), self.num_trjs * ((self.num_images - 1) * 2))

        for trk_idx in range(self.num_trjs):
            for img_idx in range(self.num_images - 1):
                tracklet = tracklets.iloc[trk_idx * (self.num_images - 1) + img_idx]
                det1 = tracklet[["#Image1", "RA1", "Dec1"]]
                det2 = tracklet[["Image2", "RA2", "Dec2"]]

                o_idx = trk_idx * (self.num_images) + img_idx
                o_det1 = df.iloc[o_idx]
                o_det2 = df.iloc[o_idx + 1]

                self.assertEqual(det1["#Image1"], o_det1["mjd"])
                self.assertEqual(det1["RA1"], o_det1["ra"])
                self.assertEqual(det1["Dec1"], o_det1["dec"])

                self.assertEqual(det2["Image2"], o_det2["mjd"])
                self.assertEqual(det2["RA2"], o_det2["ra"])
                self.assertEqual(det2["Dec2"], o_det2["dec"])

    def test_make_manual_tracklets_without_uuid(self):
        test_res = self.res
        test_res.table.remove_column("uuid")
        df = unravel_results(test_res, self.ic)
        self.assertRaises(ValueError, make_manual_tracklets, df)


if __name__ == "__main__":
    unittest.main()
