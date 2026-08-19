"""Tests for kbmod.sorcha_injection.

These build a small synthetic Sorcha index and a small synthetic ImageCollection so
the whole selection/catalog path is exercised without touching the 3 TB of real
Sorcha output, the pointing database, or a Butler.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from kbmod.sorcha_injection.config import SorchaInjectionConfig
from kbmod.sorcha_injection.visits import ic_visit_table, sorcha_epoch_from_ic

# Geometry shared by the fake collection and the fake index.
CENTER_RA, CENTER_DEC = 150.0, -30.0
NX = NY = 512
PIXEL_SCALE = 0.2 / 3600.0  # deg/pixel
N_VISITS = 5
BASE_MJD = 60900.0
EXP_TIME = 30.0
VISIT_IDS = np.array([2025070100100 + i for i in range(N_VISITS)], dtype=np.int64)


def _make_wcs(crval1=CENTER_RA, crval2=CENTER_DEC, nx=NX, ny=NY):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [crval1, crval2]
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    w.wcs.cdelt = [-PIXEL_SCALE, PIXEL_SCALE]
    w.pixel_shape = (nx, ny)
    return w


def _wcs_json(w):
    """Serialise a WCS the way ImageCollection stores it (FITS header as JSON)."""
    hdr = w.to_header(relax=True)
    d = {k: hdr[k] for k in hdr}
    d["NAXIS1"], d["NAXIS2"] = w.pixel_shape
    return json.dumps(d)


class FakeIC:
    """Minimal stand-in for an ImageCollection: one detector per visit."""

    def __init__(self, guess_distance=None, global_wcs=None):
        w = _make_wcs()
        mjd_start = BASE_MJD + np.arange(N_VISITS) * 0.01
        mjd_mid = mjd_start + EXP_TIME / 2.0 / 86400.0
        cols = {
            "visit": VISIT_IDS,
            "detector": np.zeros(N_VISITS, dtype=np.int64),
            "mjd_start": mjd_start,
            "mjd_mid": mjd_mid,
            "exposureTime": np.full(N_VISITS, EXP_TIME),
            "band": np.array(["r"] * N_VISITS),
            "ra": np.full(N_VISITS, CENTER_RA),
            "dec": np.full(N_VISITS, CENTER_DEC),
            "pointing_ra": np.full(N_VISITS, CENTER_RA),
            "pointing_dec": np.full(N_VISITS, CENTER_DEC),
            "wcs": np.array([_wcs_json(w)] * N_VISITS),
        }
        if guess_distance is not None:
            cols["helio_guess_dist"] = np.full(N_VISITS, guess_distance)
        self.data = Table(cols)
        self._global_wcs = global_wcs

    def __len__(self):
        return len(self.data)

    def get_global_wcs(self):
        return self._global_wcs

    def get_observatory(self):
        from astropy.coordinates import EarthLocation
        import astropy.units as u

        return EarthLocation(lat=-30.2446 * u.deg, lon=-70.7494 * u.deg, height=2663 * u.m)


def _write_fake_index(path, n_objects=4, nside=64):
    """Write a small index with the real schema and partitioning."""
    import hpgeom
    import pyarrow as pa
    import pyarrow.parquet as pq

    w = _make_wcs()
    rows = []
    for j in range(n_objects):
        # Each object drifts a few pixels per visit, starting spread across the chip.
        x0, y0 = 100 + 60 * j, 100 + 40 * j
        for i in range(N_VISITS):
            sky = w.pixel_to_world(x0 + 3.0 * i, y0 + 2.0 * i)
            rows.append(
                {
                    "ObjID": f"TEST|{j:06d}",
                    "population": "cc",
                    "visit": int(VISIT_IDS[i]),
                    # Sorcha's epoch is the exposure start, expressed in TAI.
                    "fieldMJD_TAI": BASE_MJD + i * 0.01 + 37.0 / 86400.0,
                    "visit_time": 30.93,
                    "RA_deg": float(sky.ra.deg),
                    "Dec_deg": float(sky.dec.deg),
                    "optFilter": "r",
                    "trailedSourceMag": 22.0 + 0.5 * j,
                    "night": int(BASE_MJD),
                }
            )
    tbl = pa.Table.from_pylist(rows)
    hp = hpgeom.angle_to_pixel(
        nside,
        tbl["RA_deg"].to_numpy(zero_copy_only=False),
        tbl["Dec_deg"].to_numpy(zero_copy_only=False),
        nest=True,
        lonlat=True,
        degrees=True,
    )
    tbl = tbl.append_column("healpix", pa.array(hp.astype(np.int64), type=pa.int64()))
    pq.write_to_dataset(tbl, root_path=path, partition_cols=["population", "night"])
    meta = {
        "populations": ["cc"],
        "mag_max": 27.0,
        "nside": nside,
        "n_rows_kept": len(rows),
        "n_objects": n_objects,
        "n_shards": 1,
        "n_shards_skipped": 0,
    }
    with open(os.path.join(path, "_index_meta.json"), "w") as fh:
        json.dump(meta, fh)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SorchaInjectionConfig(index_path="/tmp/x")
        self.assertEqual(cfg.mag_range, (None, 27.0))
        self.assertTrue(cfg.require_on_detector)
        self.assertTrue(cfg.correct_epoch_to_mid_exposure)

    def test_rejects_unknown_population(self):
        with self.assertRaises(ValueError):
            SorchaInjectionConfig(index_path="/tmp/x", populations=["not_a_population"])

    def test_rejects_inverted_mag_range(self):
        with self.assertRaises(ValueError):
            SorchaInjectionConfig(index_path="/tmp/x", mag_range=(27.0, 19.0))

    def test_from_dict_random_source_returns_none(self):
        self.assertIsNone(SorchaInjectionConfig.from_dict({"source": "random"}))

    def test_from_dict_builds_sorcha_config(self):
        cfg = SorchaInjectionConfig.from_dict(
            {"source": "sorcha", "index_path": "/tmp/idx", "mag_range": [20.0, 25.0], "max_objs_per_patch": 7}
        )
        self.assertEqual(cfg.index_path, "/tmp/idx")
        self.assertEqual(cfg.mag_range, (20.0, 25.0))
        self.assertEqual(cfg.max_objs_per_patch, 7)

    def test_from_dict_rejects_unknown_key(self):
        with self.assertRaises(ValueError):
            SorchaInjectionConfig.from_dict({"index_path": "/tmp/idx", "bogus": 1})


class TestVisits(unittest.TestCase):
    def test_ic_visit_table_dedups_and_orders(self):
        ic = FakeIC()
        # Duplicate a visit across two detectors, as a real collection would.
        ic.data = Table(np.concatenate([ic.data.as_array(), ic.data.as_array()[:1]]))
        v = ic_visit_table(ic)
        self.assertEqual(len(v["visit"]), N_VISITS)
        self.assertTrue(np.all(np.diff(v["mjd_mid"]) > 0))

    def test_sorcha_epoch_prefers_mjd_start(self):
        mid = np.array([60900.5])
        exp = np.array([30.0])
        start = np.array([60900.5 - 15.5 / 86400.0])
        # With mjd_start given, that exact value is returned...
        np.testing.assert_allclose(sorcha_epoch_from_ic(mid, exp, mjd_start=start), start)
        # ...and without it the exposureTime/2 approximation is used instead, which is
        # 0.5 s later because Rubin's mid-exposure uses the visit time, not the
        # exposure time.
        approx = sorcha_epoch_from_ic(mid, exp)
        self.assertAlmostEqual(float((approx - start)[0] * 86400.0), 0.5, places=6)

    def test_sorcha_epoch_falls_back_when_start_is_nan(self):
        out = sorcha_epoch_from_ic(np.array([60900.5]), np.array([30.0]), mjd_start=np.array([np.nan]))
        self.assertTrue(np.isfinite(out).all())


@unittest.skipUnless(HAS_PYARROW, "pyarrow is required for the Sorcha index")
class TestSelectionAndCatalog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.index_path = os.path.join(self.tmp, "index")
        _write_fake_index(self.index_path)
        from kbmod.sorcha_injection import SorchaIndex

        self.index = SorchaIndex(self.index_path)
        self.cfg = SorchaInjectionConfig(index_path=self.index_path, min_obs_per_object=1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_index_roundtrip(self):
        self.assertEqual(self.index.nside, 64)
        self.assertEqual(self.index.populations, ("cc",))
        self.assertEqual(self.index.read().num_rows, 4 * N_VISITS)
        self.assertEqual(self.index.read(visits=VISIT_IDS[:2]).num_rows, 4 * 2)
        self.assertEqual(self.index.read(mag_range=(None, 22.4)).num_rows, N_VISITS)

    def test_catalog_schema_matches_stock_generator(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha
        from kbmod.sorcha_injection.validate import REQUIRED_CATALOG_COLUMNS

        ic = FakeIC(global_wcs=_make_wcs())
        cat = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg)
        for col in REQUIRED_CATALOG_COLUMNS:
            self.assertIn(col, cat.colnames)
        self.assertEqual(len(cat), 4 * N_VISITS)
        self.assertEqual(len(np.unique(cat["injection_id"])), len(cat))

    def test_objids_are_preserved(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        ic = FakeIC(global_wcs=_make_wcs())
        cat = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg)
        self.assertEqual(
            sorted(set(np.asarray(cat["obj_ids"], dtype=str))), [f"TEST|{j:06d}" for j in range(4)]
        )

    def test_obstime_is_bit_identical_to_ic_mjd_mid(self):
        # inject_sources_into_ic selects rows with an exact float equality test,
        # so anything less than bit-identical silently injects nothing.
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        ic = FakeIC(global_wcs=_make_wcs())
        cat = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg)
        ic_times = set(np.asarray(ic.data["mjd_mid"], dtype=float).tolist())
        for t in np.asarray(cat["obstime"], dtype=float):
            self.assertIn(t, ic_times)

    def test_rows_outside_the_detector_are_dropped(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        # A global WCS pointing far away leaves nothing inside the patch.
        far = _make_wcs(crval1=CENTER_RA + 5.0)
        ic = FakeIC(global_wcs=far)
        cat = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg)
        self.assertEqual(len(cat), 0)

    def test_min_obs_filter_drops_short_tracks(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        ic = FakeIC(global_wcs=_make_wcs())
        cfg = SorchaInjectionConfig(index_path=self.index_path, min_obs_per_object=N_VISITS + 1)
        cat = generate_injection_catalog_from_sorcha(ic, self.index, cfg)
        self.assertEqual(len(cat), 0)

    def test_max_objs_caps_whole_tracks(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        ic = FakeIC(global_wcs=_make_wcs())
        cfg = SorchaInjectionConfig(index_path=self.index_path, min_obs_per_object=1, max_objs_per_patch=2)
        cat = generate_injection_catalog_from_sorcha(ic, self.index, cfg)
        ids, counts = np.unique(np.asarray(cat["obj_ids"], dtype=str), return_counts=True)
        self.assertEqual(len(ids), 2)
        # Capping must keep tracks whole, never thin them row-by-row.
        self.assertTrue(np.all(counts == N_VISITS))

    def test_epoch_correction_shifts_positions_slightly(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        ic = FakeIC(global_wcs=_make_wcs())
        on = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg)
        cfg_off = SorchaInjectionConfig(
            index_path=self.index_path, min_obs_per_object=1, correct_epoch_to_mid_exposure=False
        )
        off = generate_injection_catalog_from_sorcha(ic, self.index, cfg_off)
        self.assertEqual(len(on), len(off))
        d = np.hypot(on["ra"] - off["ra"], on["dec"] - off["dec"]) * 3600.0
        # Non-zero (the correction did something) but far below a pixel.
        self.assertGreater(float(np.max(d)), 0.0)
        self.assertLess(float(np.max(d)), 0.2)

    def test_plot_columns_track_the_patch_frame(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha

        gw = _make_wcs()
        ic = FakeIC(global_wcs=gw)
        cat = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg)
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        x, y = gw.world_to_pixel(SkyCoord(np.asarray(cat["ra"]) * u.deg, np.asarray(cat["dec"]) * u.deg))
        np.testing.assert_allclose(x, np.asarray(cat["plot_x"]), atol=1e-6)
        np.testing.assert_allclose(y, np.asarray(cat["plot_y"]), atol=1e-6)

    def test_truth_table_roundtrip(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha, write_injection_truth
        import pandas as pd

        ic = FakeIC(global_wcs=_make_wcs())
        cat, sel = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg, return_selection=True)
        path = os.path.join(self.tmp, "truth.parquet")
        write_injection_truth(path, cat, sel, ic_name="patch_a", patch_id=1)
        write_injection_truth(path, cat, sel, ic_name="patch_b", patch_id=2)
        df = pd.read_parquet(path)
        self.assertEqual(df.ic_name.nunique(), 2)
        self.assertEqual(len(df), 2 * len(cat))
        # The whole point: the same ObjID appears under both collections.
        per = df.groupby("obj_ids").ic_name.nunique()
        self.assertTrue((per == 2).all())

    def test_truth_table_replaces_rather_than_duplicates(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha, write_injection_truth
        import pandas as pd

        ic = FakeIC(global_wcs=_make_wcs())
        cat, sel = generate_injection_catalog_from_sorcha(ic, self.index, self.cfg, return_selection=True)
        path = os.path.join(self.tmp, "truth.parquet")
        write_injection_truth(path, cat, sel, ic_name="patch_a")
        write_injection_truth(path, cat, sel, ic_name="patch_a")
        self.assertEqual(len(pd.read_parquet(path)), len(cat))

    def test_dispatcher_defaults_to_random(self):
        from kbmod.sorcha_injection import generate_injection_catalog_for_ic

        # With injection_config=None the Sorcha path must not be taken; the stock
        # generator is called instead (and here fails on the fake config, proving it).
        ic = FakeIC(global_wcs=_make_wcs())
        with self.assertRaises(Exception):
            generate_injection_catalog_for_ic(ic, {}, _make_wcs(), injection_config=None)


@unittest.skipUnless(HAS_PYARROW, "pyarrow is required for the Sorcha index")
class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.index_path = os.path.join(self.tmp, "index")
        _write_fake_index(self.index_path)
        from kbmod.sorcha_injection import SorchaIndex

        self.index = SorchaIndex(self.index_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_index_checks_pass(self):
        from kbmod.sorcha_injection.validate import check_index

        checks = check_index(self.index, expect_populations=("cc",))
        self.assertTrue(all(c.passed for c in checks), [str(c) for c in checks if not c.passed])

    def test_catalog_checks_pass(self):
        from kbmod.sorcha_injection import generate_injection_catalog_from_sorcha
        from kbmod.sorcha_injection.validate import check_catalog

        ic = FakeIC(global_wcs=_make_wcs())
        cfg = SorchaInjectionConfig(index_path=self.index_path, min_obs_per_object=1)
        cat = generate_injection_catalog_from_sorcha(ic, self.index, cfg)
        checks = check_catalog(cat, ic)
        self.assertTrue(all(c.passed for c in checks), [str(c) for c in checks if not c.passed])

    def test_catalog_check_flags_missing_columns(self):
        from kbmod.sorcha_injection.validate import check_catalog

        ic = FakeIC(global_wcs=_make_wcs())
        bad = Table({"ra": [1.0], "dec": [2.0]})
        checks = check_catalog(bad, ic)
        self.assertFalse(checks[0].passed)


if __name__ == "__main__":
    unittest.main()


class TestTrackGeometry(unittest.TestCase):
    """The vectorised helpers behind the all-sky per-object track product."""

    def test_angsep_matches_astropy(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from kbmod.sorcha_injection.tracks import _angsep_deg

        rng = np.random.default_rng(0)
        ra1, ra2 = rng.uniform(0, 360, 200), rng.uniform(0, 360, 200)
        d1, d2 = rng.uniform(-89, 89, 200), rng.uniform(-89, 89, 200)
        want = SkyCoord(ra1 * u.deg, d1 * u.deg).separation(SkyCoord(ra2 * u.deg, d2 * u.deg)).deg
        np.testing.assert_allclose(_angsep_deg(ra1, d1, ra2, d2), want, atol=1e-9)

    def test_ecliptic_puts_the_ecliptic_pole_at_ninety(self):
        from kbmod.sorcha_injection.tracks import _to_ecliptic

        # The north ecliptic pole sits at RA 18h, Dec +66.56 in equatorial coordinates.
        _, lat = _to_ecliptic(np.array([270.0]), np.array([66.5607]))
        self.assertAlmostEqual(float(lat[0]), 90.0, places=3)
        # A point on the equator at the vernal equinox has zero ecliptic latitude.
        _, lat0 = _to_ecliptic(np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(float(lat0[0]), 0.0, places=6)

    def test_group_median_skips_nan(self):
        from kbmod.sorcha_injection.tracks import _group_median

        vals = np.array([1.0, 3.0, np.nan, 10.0, 20.0, 30.0])
        codes = np.array([0, 0, 0, 1, 1, 1])
        out = _group_median(vals, codes, 2)
        self.assertAlmostEqual(out[0], 2.0)
        self.assertAlmostEqual(out[1], 20.0)
