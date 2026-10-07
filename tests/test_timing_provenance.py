"""Provenance, metadata-only migration, and fail-before-publication regressions."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import astropy.units as u
from astropy.table import Table, vstack
from astropy.time import Time
import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.image_collection import ImageCollection, pack_table, unpack_table
from kbmod.image_collection_timing import (
    ButlerTimingResolver,
    corrected_collection,
    read_collection_table,
    write_upgraded_collection,
)
from kbmod.results import Results
from kbmod.timing import TIMING_FIELDS, butler_timing_metadata, timing_summary
from kbmod.work_unit import WorkUnit
from kbmod_cmdline.kbmod_migrate_imagecollections import main


def sample_collection():
    native = Time(["2025-06-02T11:59:22", "2025-06-03T03:00:00", "2025-06-03T03:10:00"], scale="utc")
    exposures = [30.0, 60.0, 120.0]
    rows = []
    for i, (t, duration) in enumerate(zip(native, exposures)):
        rows.append(
            dict(
                mjd_mid=t.mjd + (duration / 2 + 0.5) / 86400,
                mjd_start=t.mjd,
                obs_day=19990101,
                exposureTime=duration,
                dataId=str(uuid.UUID(int=i + 1)),
                visit=i + 10,
                detector=i,
                std_name="ButlerStandardizer",
                std_idx=i,
                ext_idx=0,
                config='{"psf_std":1.25}',
                location="/original/datastore",
                wcs="{}",
                ra=20.0,
                dec=-5.0,
                obs_lon=-70.0,
                obs_lat=-30.0,
                obs_elev=2000.0,
                custom_label=f"keep-{i}",
            )
        )
    return Table(rows=rows, meta={"n_stds": 3, "campaign": "preserve me"}), native, exposures


class MetadataButler:
    def __init__(self, data, native, exposures):
        self.records = {}
        self.refs_read = self.visits_read = 0
        for row, t, exposure in zip(data, native, exposures):
            identifier = uuid.UUID(row["dataId"])
            self.records[identifier] = (
                dict(visit=row["visit"], detector=row["detector"]),
                t,
                exposure,
            )

    def get_dataset(self, identifier, dimension_records=True):
        self.refs_read += 1
        identity, _, _ = self.records[identifier]
        return SimpleNamespace(
            id=identifier, dataId=identity, makeComponentRef=lambda name: (identifier, name)
        )

    def get(self, component):
        identifier, name = component
        if name != "visitInfo":
            raise AssertionError("Image or unrelated component requested")
        self.visits_read += 1
        _, t, exposure = self.records[identifier]
        return SimpleNamespace(date=SimpleNamespace(toAstropy=lambda: t.tai), exposureTime=exposure)


class TestTimingProvenance(unittest.TestCase):
    def setUp(self):
        self.data, self.native, self.exposures = sample_collection()
        self.butler = MetadataButler(self.data, self.native, self.exposures)
        self.resolver = ButlerTimingResolver(butler=self.butler)

    def corrected(self):
        return corrected_collection(self.data, self.resolver)[0]

    def test_native_upgrade_mixed_exposures_and_idempotence(self):
        before = self.data.copy()
        corrected, report = corrected_collection(self.data, self.resolver)
        self.assertEqual(report["status"], "needs_upgrade")
        np.testing.assert_allclose(corrected["mjd_mid"], self.native.mjd, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            corrected["mjd_start"],
            (self.native - np.asarray(self.exposures) / 2 * u.s).mjd,
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_array_equal(corrected["obs_day"], [20250601, 20250602, 20250602])
        np.testing.assert_array_equal(corrected["timing_previous_mjd_mid"], before["mjd_mid"])
        np.testing.assert_array_equal(self.data["mjd_mid"], before["mjd_mid"])
        for key in ("wcs", "config", "location", "custom_label", "dataId"):
            np.testing.assert_array_equal(corrected[key], before[key])
        repeated, report = corrected_collection(corrected, self.resolver)
        self.assertEqual(report["status"], "verified_current")
        self.assertEqual(len(repeated.meta["timing_migrations"]), 1)
        self.assertEqual(self.butler.visits_read, 3)  # Cached across both files/passes.

    def test_already_correct_unmarked_and_manual_offsets(self):
        for delta in (0.0, -60.25, 97.5):
            with self.subTest(delta=delta):
                data = self.data.copy()
                data["mjd_mid"] = self.native.mjd + delta / 86400
                corrected, report = corrected_collection(data, self.resolver)
                self.assertEqual(report["status"], "needs_upgrade")
                np.testing.assert_allclose(corrected["mjd_mid"], self.native.mjd, rtol=0, atol=1e-10)
                self.assertEqual(timing_summary(corrected)["status"], "current")

    def test_old_current_mixed_and_modified_records(self):
        self.assertEqual(ImageCollection(self.data).timing_provenance["status"], "unknown")
        current = self.corrected()
        for data in (current, pack_table(current.copy())):
            self.assertEqual(timing_summary(data)["status"], "current")
        mixed = vstack([current, self.data], metadata_conflicts="silent")
        self.assertEqual(timing_summary(mixed)["status"], "mixed")
        current["mjd_mid"][0] += 1 / 86400
        self.assertEqual(timing_summary(current)["status"], "inconsistent")
        current = self.corrected()
        current["dataId"][0] = str(uuid.UUID(int=100))
        self.assertEqual(timing_summary(current)["status"], "inconsistent")

    def test_identity_future_convention_unsupported_and_source_failure(self):
        for field, value in (("visit", 100), ("detector", 100), ("std_name", "Other")):
            with self.subTest(field=field):
                data = self.data.copy()
                data[field][0] = value
                with self.assertRaises(ValueError):
                    corrected_collection(data, self.resolver)
        data = self.corrected()
        data["timing_convention"] = ["future.v2"] * len(data)
        with self.assertRaisesRegex(ValueError, "Unsupported recorded"):
            corrected_collection(data, self.resolver)
        self.butler.records.pop(uuid.UUID(self.data["dataId"][-1]))
        with self.assertRaises(KeyError):
            corrected_collection(self.data, ButlerTimingResolver(butler=self.butler))

    def test_reflex_cache_requires_explicit_removal(self):
        data = self.data.copy()
        data["ra_42.0"] = [1.0, 2.0, 3.0]
        data["dec_tl_42.0"] = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "invalidate reflex"):
            corrected_collection(data, self.resolver)
        corrected, report = corrected_collection(data, self.resolver, drop_reflex=True)
        self.assertEqual(report["dropped_columns"], ["ra_42.0", "dec_tl_42.0"])
        self.assertNotIn("ra_42.0", corrected.colnames)
        self.assertIn("ra_42.0", data.colnames)

    def test_ecsv_parquet_roundtrip_no_clobber(self):
        data = self.corrected()
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".ecsv", ".parquet"):
                path = Path(tmp) / ("collection" + suffix)
                write_upgraded_collection(data, path)
                loaded = read_collection_table(path)
                self.assertEqual(timing_summary(loaded)["status"], "current")
                self.assertEqual(loaded.meta["campaign"], "preserve me")
                original_bytes = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    write_upgraded_collection(data, path)
                self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(tmp).glob(".kbmod-timing-*")), [])

    def make_workunit(self, data=None):
        pixels = [np.ones((4, 5), dtype=np.float32) for _ in self.native]
        stack = ImageStackPy(times=self.native.mjd, sci=pixels, var=pixels)
        return WorkUnit(stack, SearchConfiguration(), org_image_meta=data)

    def test_header_audit_skips_row_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix, fmt in ((".ecsv", "ascii.ecsv"), (".parquet", "parquet")):
                pack_table(self.data.copy()).write(Path(tmp) / ("legacy" + suffix), format=fmt)
                Table({"x": [1]}).write(Path(tmp) / ("unrelated" + suffix), format=fmt)
            with (
                redirect_stdout(io.StringIO()),
                patch.object(Table, "read", side_effect=AssertionError("Row data must not be read")),
            ):
                self.assertEqual(main(["--input", tmp]), 2)

    def test_failed_output_validation_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "output.ecsv"
            with patch(
                "kbmod.image_collection_timing.read_collection_table", side_effect=ValueError("broken")
            ):
                with self.assertRaisesRegex(ValueError, "broken"):
                    write_upgraded_collection(self.corrected(), target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_workunit_fits_sharded_lazy_and_changed_epochs(self):
        self.assertEqual(self.make_workunit().timing_provenance["status"], "unknown")
        work = self.make_workunit(self.corrected())
        self.assertEqual(work.timing_provenance["status"], "current")
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "work.fits"
            work.to_fits(file)
            self.assertEqual(WorkUnit.from_fits(file).timing_provenance, work.timing_provenance)
            work.to_sharded_fits("shards.fits", tmp)
            lazy = WorkUnit.from_sharded_fits("shards.fits", tmp, lazy=True)
            self.assertEqual(lazy.timing_provenance, work.timing_provenance)
        work.im_stack.times[0] += 1 / 86400
        self.assertEqual(work.timing_provenance["status"], "inconsistent")

    def test_search_result_provenance_and_merge_unknown(self):
        from kbmod.run_search import SearchRunner
        from kbmod.search import Trajectory

        work = self.make_workunit(self.corrected())
        work.im_stack.sci[1][:] = np.nan  # Verify provenance after image filtering.
        runner = SearchRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.parquet"
            config = SearchConfiguration(
                dict(
                    max_masked_pixels=0.5,
                    stamp_type=None,
                    do_clustering=False,
                    compute_ra_dec=False,
                    save_config=False,
                    result_filename=str(path),
                )
            )
            rows = Results.from_trajectories([Trajectory(x=1, y=1)])
            with patch.object(runner, "do_core_search", return_value=rows):
                result = runner.run_search(
                    config,
                    work.im_stack,
                    trj_generator=[],
                    workunit=work,
                    extra_meta={"timing_provenance": {"status": "fake"}},
                )
            for res in (result, Results.read_table(path), next(Results.read_table_chunks(path))):
                self.assertEqual(res.timing_provenance["status"], "current")
                self.assertEqual(res.timing_provenance["rows"], 2)
                res.set_mjd_utc_mid(res.mjd_mid + 1)
                self.assertEqual(res.timing_provenance["status"], "unknown")
                res.write_table(Path(tmp) / "edited.parquet")
                self.assertEqual(
                    Results.read_table(Path(tmp) / "edited.parquet").table.meta["timing_provenance"][
                        "status"
                    ],
                    "unknown",
                )
            result.extend(Results.from_trajectories([Trajectory(x=2, y=2)]))
            self.assertEqual(result.timing_provenance["status"], "unknown")

    def test_cli_offline_audit_dry_run_upgrade_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            path = inputs / "old.ecsv"
            pack_table(self.data.copy()).write(path, format="ascii.ecsv")
            original_bytes = path.read_bytes()
            Table({"x": [1]}).write(inputs / "unrelated.ecsv", format="ascii.ecsv")

            def run(*args):
                stream = io.StringIO()
                with (
                    redirect_stdout(stream),
                    patch(
                        "kbmod_cmdline.kbmod_migrate_imagecollections.ButlerTimingResolver",
                        return_value=self.resolver,
                    ),
                ):
                    code = main(["--input", str(inputs), *args])
                return code, [json.loads(line) for line in stream.getvalue().splitlines()]

            code, records = run()
            self.assertEqual(code, 2)
            self.assertEqual(records[0]["status"], "unknown")
            self.assertEqual(records[1]["status"], "skipped")
            self.assertEqual(self.butler.visits_read, 0)
            code, records = run("--upgrade", "--dry-run", "--butler", "repo")
            self.assertEqual(code, 2)
            self.assertEqual(records[0]["status"], "needs_upgrade")
            out = root / "corrected"
            code, records = run("--upgrade", "--butler", "repo", "--output-dir", str(out))
            self.assertEqual(code, 0)
            self.assertEqual(records[0]["status"], "upgraded")
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(timing_summary(read_collection_table(out / "old.ecsv"))["status"], "current")
            code, _ = run("--upgrade", "--butler", "repo", "--output-dir", str(out))
            self.assertEqual(code, 1)  # Existing destination is never overwritten.
            broken = self.data.copy()
            broken["dataId"][-1] = str(uuid.UUID(int=9999))
            pack_table(broken).write(inputs / "broken.ecsv", format="ascii.ecsv")
            code, records = run("--verify-source", "--butler", "repo")
            self.assertEqual(code, 1)
            self.assertEqual(records[0]["status"], "error")
            self.assertEqual(records[1]["status"], "needs_upgrade")
            out2 = root / "partial"
            code, records = run("--upgrade", "--butler", "repo", "--output-dir", str(out2))
            self.assertEqual(code, 1)
            self.assertFalse((out2 / "broken.ecsv").exists())
            self.assertTrue((out2 / "old.ecsv").exists())


if __name__ == "__main__":
    unittest.main()
