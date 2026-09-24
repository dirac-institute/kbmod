"""Correctness and failure recovery for compressed result-column writes."""

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astropy.io import fits
import numpy as np

from kbmod.results import Results, write_results_to_files_destructive


def make_results(rows=5, shape=(3, 7, 11)):
    results = Results(
        {key: np.zeros(rows) for key in ("x", "y", "vx", "vy", "flux", "likelihood", "obs_count")}
    )
    results.table["uuid"] = [f"stamp-{i}" for i in range(rows)]
    results.table["stamp"] = np.random.default_rng(1134).normal(size=(rows, *shape)).astype(np.float32)
    return results


class TestParallelColumnWrites(unittest.TestCase):
    def test_start_methods_match_serial(self):
        results = make_results()
        before = results["stamp"].copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = root / "serial.fits"
            results.write_column("stamp", serial, is_image=True)
            for method in mp.get_all_start_methods():
                with self.subTest(method=method), patch("kbmod.results.Pool", mp.get_context(method).Pool):
                    parallel = root / "parallel.fits"
                    results.write_column("stamp", parallel, is_image=True, num_workers=2)
                    with fits.open(serial) as expected, fits.open(parallel) as actual:
                        actual.verify("exception")
                        self.assertEqual(len(actual), len(results) + 1)
                        for key in ("NUMRES", "ISIMG", "COLNAME", "EXTEND"):
                            self.assertEqual(actual[0].header[key], expected[0].header[key])
                        for index, hdu in enumerate(actual[1:]):
                            self.assertEqual(hdu.name, f"IMG_{index}")
                            self.assertEqual(hdu.header["UUID"], results["uuid"][index])
                            np.testing.assert_array_equal(hdu.data, expected[index + 1].data)
                    np.testing.assert_array_equal(results["stamp"], before)
            self.assertEqual({p.name for p in root.iterdir()}, {"serial.fits", "parallel.fits"})

    def test_detached_empty_and_single_row_columns(self):
        for rows in (0, 1):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                results = make_results(rows, (7, 11))
                column = results["stamp"]
                results.remove_column("stamp")
                results.remove_column("uuid")
                path = Path(directory) / "stamp.fits"
                with patch("kbmod.results.Pool") as pool:
                    results.write_column(column, path, is_image=True, num_workers=8)
                    pool.assert_not_called()
                with fits.open(path) as hdus:
                    hdus.verify("exception")
                    self.assertEqual(len(hdus), rows + 1)
                    self.assertEqual(hdus[0].header["NUMRES"], rows)
                    for hdu in hdus[1:]:
                        self.assertNotIn("UUID", hdu.header)
                        np.testing.assert_allclose(hdu.data, column[0], atol=0.00501, rtol=0)

    def test_nonfinite_and_integer_data(self):
        for dtype in (np.float32, np.float64, np.int16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                results = make_results(3, (7, 11))
                data = np.arange(3 * 7 * 11).reshape(3, 7, 11).astype(dtype)
                if np.issubdtype(dtype, np.floating):
                    data[0, 0, :3] = [np.nan, np.inf, -np.inf]
                results.table["stamp"] = data
                serial, parallel = Path(directory) / "serial.fits", Path(directory) / "parallel.fits"
                results.write_column("stamp", serial, is_image=True)
                results.write_column("stamp", parallel, is_image=True, num_workers=np.int64(2))
                with fits.open(serial) as expected, fits.open(parallel) as actual:
                    for index in range(1, 4):
                        np.testing.assert_array_equal(actual[index].data, expected[index].data)

    def test_worker_failure_preserves_destination_and_cleans_scratch(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "stamp.fits"
                results = make_results()
                results.write_column("stamp", path, is_image=True)
                original = path.read_bytes()
                results.table["stamp"] = np.full((5, 7, 11), "not numeric")
                with self.assertRaises((KeyError, TypeError, ValueError)):
                    results.write_column("stamp", path, is_image=True, num_workers=workers)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(root.iterdir()), [path])

    def test_copy_failure_never_publishes_partial_output(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "stamp.fits"
                results = make_results()
                if existing:
                    results.write_column("stamp", path, is_image=True)
                original = path.read_bytes() if existing else None
                with patch("kbmod.results.shutil.copyfileobj", side_effect=OSError("copy failed")):
                    with self.assertRaisesRegex(OSError, "copy failed"):
                        results.write_column("stamp", path, is_image=True, num_workers=2)
                if existing:
                    self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(root.iterdir()), [path] if existing else [])

    def test_pool_startup_failure_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kbmod.results.Pool", side_effect=OSError("pool startup failed")):
                with self.assertRaisesRegex(OSError, "pool startup failed"):
                    make_results().write_column("stamp", root / "stamp.fits", is_image=True, num_workers=2)
            self.assertEqual(list(root.iterdir()), [])

    def test_publish_failure_preserves_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stamp.fits"
            results = make_results()
            results.write_column("stamp", path, is_image=True)
            original = path.read_bytes()
            with patch("kbmod.results.os.replace", side_effect=OSError("publish failed")):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    results.write_column("stamp", path, is_image=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [path])

    def test_no_overwrite_even_if_destination_appears_during_write(self):
        import os

        real_link = os.link
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stamp.fits"

            def competing_writer(source, destination):
                Path(destination).write_bytes(b"another writer's completed output")
                return real_link(source, destination)

            with patch("kbmod.results.os.link", side_effect=competing_writer):
                with self.assertRaises(FileExistsError):
                    make_results().write_column("stamp", path, is_image=True, overwrite=False)
            self.assertEqual(path.read_bytes(), b"another writer's completed output")
            self.assertEqual(list(root.iterdir()), [path])

    def test_no_overwrite_success_and_early_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stamp.fits"
            results = make_results()
            results.write_column("stamp", path, is_image=True, overwrite=False, num_workers=2)
            original = path.read_bytes()
            with patch("kbmod.results.Pool") as pool:
                with self.assertRaises(FileExistsError):
                    results.write_column("stamp", path, is_image=True, overwrite=False, num_workers=2)
                pool.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [path])

    def test_invalid_worker_count_does_not_mutate_results(self):
        for workers in (0, -1, 1.5, None, "2"):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                results = make_results()
                before = results.copy()
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    results.write_column("stamp", root / "stamp.fits", is_image=True, num_workers=workers)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    write_results_to_files_destructive(
                        root / "main.parquet", results, separate_col_files=["stamp"], num_workers=workers
                    )
                self.assertEqual(results.colnames, before.colnames)
                np.testing.assert_array_equal(results["stamp"], before["stamp"])
                self.assertEqual(list(root.iterdir()), [])

    def test_migration_with_spawn_workers(self):
        from kbmod_cmdline.kbmod_migrate_results import process_single_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.search.parquet"
            destination = root / "migrated"
            destination.mkdir()
            legacy = make_results(5, (49,))
            legacy.table["stamp"] = np.arange(5 * 49, dtype=np.float32).reshape(5, 49)
            legacy.write_table(source)
            original = source.read_bytes()
            with patch("kbmod.results.Pool", mp.get_context("spawn").Pool):
                migrated = process_single_file(
                    (source, ["stamp"], 7, str(destination), False, False, True, None, 2)
                )
            self.assertTrue(migrated.success, migrated.error_msg)
            actual = Results.read_table(destination / source.name, load_aux_files=True)
            np.testing.assert_array_equal(actual["stamp"], legacy["stamp"].reshape(5, 7, 7))
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(list(destination.glob(".kbmod-stamps-*")))

    def test_concurrent_calls_do_not_share_stamp_data(self):
        # ThreadPool isolates the parent calls; each call still uses a real process pool.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [make_results(3, (7, 11)) for _ in range(2)]
            inputs[1].table["stamp"] += 100

            def write(index):
                inputs[index].write_column("stamp", root / f"{index}.fits", is_image=True, num_workers=2)

            with patch("kbmod.results.Pool", mp.get_context("spawn").Pool):
                with ThreadPool(2) as threads:
                    threads.map(write, range(2))
            for index in range(2):
                with fits.open(root / f"{index}.fits") as hdus:
                    for row in range(3):
                        np.testing.assert_allclose(
                            hdus[row + 1].data, inputs[index]["stamp"][row], atol=0.00502
                        )
            self.assertEqual(len(list(root.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
