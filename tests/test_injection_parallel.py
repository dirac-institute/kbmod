"""Real process and FITS tests; Rubin rendering is supplied by a small fake."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.image_collection import ImageCollection
from kbmod import injection_parallel as parallel
from kbmod.work_unit import WorkUnit


def make_ic(times=(3.0, 1.0, 2.0, 1.0)):
    n = len(times)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.cdelt = [-0.01, 0.01]
    header = dict(wcs.to_header())
    header.update(NAXIS1=5, NAXIS2=4)
    data = Table(
        {
            "mjd_mid": np.asarray(times) + 59000,
            "dataId": [str(i) for i in range(n)],
            "std_name": ["ButlerStandardizer"] * n,
            "std_idx": np.arange(n),
            "ext_idx": np.zeros(n, dtype=int),
            "config": ["{}"] * n,
            "ra": [10.0] * n,
            "dec": [20.0] * n,
            "wcs": [json.dumps(header)] * n,
            "obs_lon": [-70.0] * n,
            "obs_lat": [-30.0] * n,
            "obs_elev": [2200.0] * n,
        },
        meta={"n_stds": n},
    )
    return ImageCollection(data)


def make_catalog():
    # Deliberately unsorted, with two sources at the same time.
    return Table(
        {
            "obstime": [59003.0, 59001.0, 59001.0, 59002.0],
            "injection_id": [30, 11, 10, 20],
        }
    )


class FakeButler:
    def __init__(self, config=None):
        self.pid = os.getpid()

    def close(self):
        pass


class FakeInjected:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def toWorkUnit(self, search_config, butler):
        sci = [np.arange(20, dtype=np.float32).reshape(4, 5) / 37 + int(i) for i in self.data["dataId"]]
        var = [np.full((4, 5), int(i) + 0.012345, dtype=np.float32) for i in self.data["dataId"]]
        for image in sci:
            image[0, 1] = np.nan
        psfs = [np.eye(3, dtype=np.float32) * (int(i) + 1) for i in self.data["dataId"]]
        stack = ImageStackPy(times=self.data["mjd_mid"], sci=sci, var=var, psfs=psfs)
        metadata = self.data.copy()
        metadata["per_image_wcs"] = [WCS(json.loads(w)) for w in metadata["wcs"]]
        return WorkUnit(stack, search_config, org_image_meta=metadata)


def fake_inject(ic, catalog, butler, **kwargs):
    assert butler.pid == os.getpid(), "A Butler was inherited from the parent"
    assert np.array_equal(ic.data["std_idx"], np.arange(len(ic)))
    assert all(std is None for std in ic._standardizers)
    # Different job durations exercise unordered completion.
    time.sleep(0.03 if int(ic.data["dataId"][0]) == 1 else 0.01)
    catalogs = [catalog[catalog["obstime"] == t] for t in ic.data["mjd_mid"]]
    from astropy.table import vstack

    return FakeInjected(ic.data), vstack(catalogs)


def fake_initializer(config):
    parallel._worker_butler = FakeButler(config)
    parallel.inject_sources_into_ic = fake_inject


def fail_worker(job):
    raise RuntimeError("worker deliberately failed")


class TestInjectionShards(unittest.TestCase):
    def test_partition_stable_and_whole_groups(self):
        shards = parallel.partition_injection_shards([3, 1, 2, 1, 4, 2], 3)
        self.assertEqual([list(s) for s in shards], [[1, 3], [2, 5, 0], [4]])
        self.assertEqual(parallel.partition_injection_shards([], 3), [])

    def test_bad_bounds_and_oversized_groups(self):
        for value in (0, -1, True, 1.5, "2", None):
            with self.assertRaises(ValueError):
                parallel.partition_injection_shards([1], value)
        for times in ([1, np.nan], [np.inf], [[1, 2]]):
            with self.assertRaises(ValueError):
                parallel.partition_injection_shards(times)
        with self.assertRaisesRegex(ValueError, "MJD 1.0 has 3 images"):
            parallel.partition_injection_shards([1, 1, 1], 2)

    def test_metadata_copy_drops_caches_and_unpacks(self):
        ic = make_ic()
        cache = object()
        ic._standardizers[:] = cache
        ic.pack()
        original = ic.data.copy()
        data = parallel._metadata_only(ic)
        self.assertIn("std_name", data.colnames)
        self.assertTrue(ic.is_packed)
        self.assertEqual(ic.data.colnames, original.colnames)
        self.assertTrue(all(std is cache for std in ic._standardizers))
        data["dataId"][0] = "x"
        self.assertEqual(ic.data["dataId"][0], "0")

    def test_preflight_without_butler(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(parallel, "_new_butler") as create:
            for options in (
                {"injection_workers": True},
                {"max_images_per_shard": 1},
                {"variance_scale": 0},
                {"variance_scale": 0.5, "constant_variance": True},
            ):
                with self.assertRaises(ValueError):
                    parallel.inject_sources_to_workunit(
                        make_ic(),
                        make_catalog(),
                        "repo",
                        Path(directory) / "x.fits",
                        **options,
                    )
            create.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_serial_shards_and_spawn_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for workers in (1, 2):
                folder = Path(directory) / str(workers)
                folder.mkdir()
                output = folder / "images.fits"
                with (
                    mock.patch.object(parallel, "_new_butler", FakeButler),
                    mock.patch.object(parallel, "inject_sources_into_ic", fake_inject),
                    mock.patch.object(parallel, "_initialize_worker", fake_initializer),
                ):
                    path, catalog = parallel.inject_sources_to_workunit(
                        make_ic(),
                        make_catalog(),
                        "repo",
                        output,
                        injection_workers=workers,
                        max_images_per_shard=2,
                    )
                self.assertEqual(Path(path), output)
                work = WorkUnit.from_sharded_fits(output.name, str(folder))
                np.testing.assert_array_equal(work.org_img_meta["injection_input_row"], [1, 3, 2, 0])
                np.testing.assert_array_equal(work.org_img_meta["dataId"].astype(str), ["1", "3", "2", "0"])
                self.assertEqual(work._per_image_indices, [[0], [1], [2], [3]])
                np.testing.assert_array_equal(catalog["injection_id"], [11, 10, 11, 10, 20, 30])
                self.assertEqual(len(list(folder.glob("*.fits"))), 5)
                self.assertEqual(len(list(folder.glob(".kbmod-injection-*"))), 0)
                for j, original in enumerate([1, 3, 2, 0]):
                    expected = np.arange(20, dtype=np.float32).reshape(4, 5) / 37 + original
                    expected[0, 1] = np.nan
                    np.testing.assert_array_equal(work.im_stack.sci[j], expected)
                    np.testing.assert_array_equal(work.im_stack.psfs[j], np.eye(3) * (original + 1))
                    self.assertIsNotNone(work.get_wcs(j))
                    with fits.open(folder / f"{j}_images.fits") as hdul:
                        self.assertEqual(hdul[f"SCI_{j}"].header["IND_0"], j)
                        self.assertEqual(hdul[f"MSK_{j}"].data[0, 1], 1)
                outputs.append(work)
            for j in range(4):
                np.testing.assert_array_equal(outputs[0].im_stack.sci[j], outputs[1].im_stack.sci[j])
                np.testing.assert_array_equal(outputs[0].im_stack.var[j], outputs[1].im_stack.var[j])

    def test_failure_cleans_staging(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(parallel, "_initialize_worker", fake_initializer),
            mock.patch.object(parallel, "_inject_shard", fail_worker),
        ):
            with self.assertRaisesRegex(RuntimeError, "deliberately failed"):
                parallel.inject_sources_to_workunit(
                    make_ic(),
                    make_catalog(),
                    "repo",
                    Path(directory) / "x.fits",
                    injection_workers=2,
                    max_images_per_shard=2,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(parallel, "_new_butler") as create:
            output = Path(directory) / "x.fits"
            output.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                parallel.inject_sources_to_workunit(make_ic(), make_catalog(), "repo", output)
            self.assertEqual(output.read_bytes(), b"original")
            create.assert_not_called()

    def test_admission_is_bounded_and_consumer_close_shuts_down(self):
        admitted = []

        def jobs():
            for number in range(20):
                admitted.append(number)
                yield number

        def pool_factory(**kwargs):
            return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

        with mock.patch.object(parallel, "ProcessPoolExecutor", pool_factory):
            stream = parallel._bounded_results(jobs(), 2, None, (), lambda value: value)
            next(stream)
            self.assertEqual(len(admitted), 2)
            stream.close()
            self.assertEqual(len(admitted), 2)

    def test_empty_catalog_and_duplicate_dataset_ids(self):
        ic = make_ic((1.0, 1.0))
        ic.data["dataId"] = ["0", "0"]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(parallel, "_new_butler", FakeButler),
            mock.patch.object(parallel, "inject_sources_into_ic", fake_inject),
        ):
            path, catalog = parallel.inject_sources_to_workunit(
                ic, make_catalog()[:0], "repo", Path(directory) / "x.fits", max_images_per_shard=2
            )
            self.assertEqual(len(catalog), 0)
            work = WorkUnit.from_sharded_fits("x.fits", directory, lazy=True)
            np.testing.assert_array_equal(work.org_img_meta["injection_input_row"], [0, 1])
            self.assertEqual(work.get_num_images(), 2)

    def test_publication_collision_rolls_back_only_our_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "stage"
            staging.mkdir()
            for name in ("0_x.fits", "1_x.fits", "x.fits"):
                (staging / name).write_bytes(b"new")
            (root / "1_x.fits").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                parallel._publish(staging, root / "x.fits", 2)
            self.assertFalse((root / "0_x.fits").exists())
            self.assertFalse((root / "x.fits").exists())
            self.assertEqual((root / "1_x.fits").read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
