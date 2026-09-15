"""USDF smoke test: existing serial injection versus serial/parallel MJD shards.

Run as a script in a Rubin environment. Requires a real IC, the full precomputed
injection catalog, a read-only Butler configuration, and a search YAML file.
"""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import threading
import time

import numpy as np
import psutil
from astropy.io import fits
from astropy.table import Table

from kbmod.configuration import SearchConfiguration
from kbmod.image_collection import ImageCollection
from kbmod.injection import inject_sources_into_ic
from kbmod.injection_parallel import _metadata_only, inject_sources_to_workunit
from kbmod.work_unit import WorkUnit, load_layered_image_from_shard


def measure(fn):
    stop = threading.Event()
    process = psutil.Process()
    peak = [0]

    def sample():
        while not stop.is_set():
            total = 0
            for proc in [process] + process.children(recursive=True):
                try:
                    total += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak[0] = max(peak[0], total)
            stop.wait(0.1)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.monotonic()
    try:
        result = fn()
    finally:
        stop.set()
        monitor.join()
    return result, {
        "seconds": time.monotonic() - start,
        "sampled_peak_sum_rss_gib": peak[0] / 2**30,
    }


def compare(reference, candidate):
    left = WorkUnit.from_sharded_fits(reference.name, str(reference.parent), lazy=True)
    right = WorkUnit.from_sharded_fits(candidate.name, str(candidate.parent), lazy=True)
    np.testing.assert_array_equal(left.get_all_obstimes(), right.get_all_obstimes())
    np.testing.assert_array_equal(left.org_img_meta["dataId"], right.org_img_meta["dataId"])
    assert left._per_image_indices == right._per_image_indices
    assert left.config.to_yaml() == right.config.to_yaml()
    for i, (lpath, rpath) in enumerate(zip(left.file_paths, right.file_paths)):
        l_img = load_layered_image_from_shard(lpath)
        r_img = load_layered_image_from_shard(rpath)
        for label in ("sci", "var", "mask", "psf"):
            np.testing.assert_array_equal(
                getattr(l_img, label),
                getattr(r_img, label),
                err_msg=f"image {i}: {label}",
            )
        l_wcs, r_wcs = left.get_wcs(i), right.get_wcs(i)
        points = [
            [0, 0],
            [l_img.width // 2, l_img.height // 2],
            [l_img.width - 1, l_img.height - 1],
        ]
        np.testing.assert_allclose(
            l_wcs.all_pix2world(points, 0),
            r_wcs.all_pix2world(points, 0),
            rtol=0,
            atol=1e-10,
        )
        del l_img, r_img
    with fits.open(reference) as l_hdul, fits.open(candidate) as r_hdul:
        for key in ("NUMIMG", "NCON", "OBS_LAT", "OBS_LON", "OBS_ELEV"):
            assert l_hdul[0].header[key] == r_hdul[0].header[key], key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic", required=True)
    parser.add_argument("--catalog", required=True, help="Previously saved full injection input catalog")
    parser.add_argument("--butler-config", required=True)
    parser.add_argument("--search-config", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory for the smoke-test outputs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mjd-groups", type=int, default=3)
    parser.add_argument("--images-per-mjd", type=int, default=2)
    parser.add_argument("--zero-background", action="store_true")
    parser.add_argument("--constant-variance", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.mjd_groups, args.images_per_mjd) < 1:
        parser.error("Workers and sample sizes must be positive.")
    output = Path(args.output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    data = _metadata_only(ImageCollection.read(args.ic))
    times = np.unique(data["mjd_mid"])[: args.mjd_groups]
    # Explicitly take a small detector subset for the smoke test, then shuffle
    # it to exercise stable MJD ordering independently of file input order.
    indices = np.concatenate([np.flatnonzero(data["mjd_mid"] == t)[: args.images_per_mjd] for t in times])
    if len(times) < 2:
        raise ValueError("Provide an ImageCollection containing at least two distinct MJDs.")
    data = data[indices[::-1]].copy()
    data["std_idx"] = np.arange(len(data))
    data.meta["n_stds"] = len(data)
    ic = ImageCollection(data)
    ic.write(str(output / "smoke_input.ecsv"), format="ascii.ecsv")
    if args.catalog.endswith(".parquet"):
        import pandas as pd

        catalog = Table.from_pandas(pd.read_parquet(args.catalog))
    else:
        catalog = Table.read(args.catalog)
    catalog = catalog[np.isin(catalog["obstime"], times)]
    if len(catalog) == 0:
        raise ValueError("No catalog sources match the selected smoke-test MJDs.")
    config = SearchConfiguration.from_file(args.search_config)
    options = {
        "zero_background": args.zero_background,
        "constant_variance": args.constant_variance,
    }
    summary = {
        "dataset_ids": list(np.asarray(data["dataId"]).astype(str)),
        "input_mjds": list(np.asarray(data["mjd_mid"], dtype=float)),
        "catalog": str(Path(args.catalog).absolute()),
        "options": options,
        "runs": {},
    }
    for package in (
        "kbmod",
        "numpy",
        "astropy",
        "galsim",
        "lsst-source-injection",
        "lsst-daf-butler",
    ):
        try:
            summary[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            summary[package] = "not registered as a distribution"
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    def baseline():
        from lsst.daf.butler import Butler

        rows = data[np.argsort(data["mjd_mid"], kind="stable")].copy()
        rows["std_idx"] = np.arange(len(rows))
        original = ImageCollection(rows)
        butler = Butler(args.butler_config, writeable=False)
        try:
            injected, rendered = inject_sources_into_ic(original, catalog, butler, **options)
            work = injected.toWorkUnit(search_config=config, butler=butler)
            folder = output / "serial"
            folder.mkdir()
            work.to_sharded_fits(
                "images.fits",
                str(folder),
                compression_type="GZIP_1",
                quantize_level=0.0,
            )
            return str(folder / "images.fits"), rendered
        finally:
            close = getattr(butler, "close", None)
            if close:
                close()

    (reference_path, reference_catalog), stats = measure(baseline)
    summary["runs"]["serial"] = stats
    reference_catalog.write(output / "serial_catalog.ecsv")
    if len(reference_catalog) == 0:
        raise AssertionError("Serial smoke run rendered no sources; choose data/catalog with actual overlap.")
    for workers in sorted(set([1, args.workers])):
        gc.collect()
        path = output / f"workers-{workers}" / "images.fits"
        (result_path, result_catalog), stats = measure(
            lambda: inject_sources_to_workunit(
                ic,
                catalog,
                args.butler_config,
                path,
                injection_workers=workers,
                max_images_per_shard=args.images_per_mjd,
                search_config=config,
                **options,
            )
        )
        summary["runs"][f"workers-{workers}"] = stats
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        result_catalog.write(path.parent / "catalog.ecsv")
        compare(Path(reference_path), Path(result_path))
        assert reference_catalog.colnames == result_catalog.colnames
        for name in reference_catalog.colnames:
            np.testing.assert_array_equal(reference_catalog[name], result_catalog[name], err_msg=name)
        print(
            f"PASS: workers={workers}: pixel/mask/PSF/catalog equality and WCS/index agreement",
            flush=True,
        )
    summary["passed"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
