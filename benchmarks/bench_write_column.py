"""Benchmark FITS column writing and verify every decompressed stamp.

Run each source revision separately with the same seed, shape, workers and repeats.
Timing includes pool startup, compression, temporary files and concatenation; it
excludes fixture generation, verification and deletion. No fsync is requested.
Example::

    python benchmarks/bench_write_column.py --source src/kbmod/results.py \
        --rows 5000 --shape 100 100 --workers 1 2 4 8 --repeats 3 \
        --directory /tmp --output timings.json
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time

import astropy
from astropy.io import fits
import numpy as np
import pyarrow
import kbmod
import kbmod.search


def load_results(source):
    # Keep the canonical module name so Pool workers can import the helper.
    spec = importlib.util.spec_from_file_location("kbmod.results", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(cls, rows, shape, seed):
    data = {key: np.zeros(rows) for key in ("x", "y", "vx", "vy", "flux", "likelihood", "obs_count")}
    data["uuid"] = [f"row-{idx:08d}" for idx in range(rows)]
    results = cls(data)
    stamps = np.random.default_rng(seed).normal(0, 10, (rows, *shape)).astype(np.float32)
    results.table["stamp"] = stamps
    return results


def verify(path, results):
    digest = hashlib.sha256()
    max_error = 0.0
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        assert len(hdus) == len(results) + 1
        assert hdus[0].header["NUMRES"] == len(results)
        assert hdus[0].header["ISIMG"]
        assert hdus[0].header["COLNAME"] == "stamp"
        for idx, hdu in enumerate(hdus[1:]):
            assert hdu.name == f"IMG_{idx}"
            assert hdu.header["UUID"] == results["uuid"][idx]
            pixels = hdu.data
            expected = results["stamp"][idx]
            assert pixels.shape == expected.shape
            assert pixels.dtype == expected.dtype
            error = float(np.max(np.abs(pixels - expected)))
            assert error <= 0.00502, error  # Existing quantize_level=-0.01 policy.
            max_error = max(max_error, error)
            digest.update(pixels.tobytes())
    return digest.hexdigest(), max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--shape", type=int, nargs="+", default=[100, 100])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1134)
    parser.add_argument("--start-method", choices=mp.get_all_start_methods())
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_method:
        mp.set_start_method(args.start_method)
    module = load_results(args.source.resolve())
    supported = "num_workers" in inspect.signature(module.Results.write_column).parameters
    if not supported and args.workers != [1]:
        parser.error("This source supports serial writing only; specify --workers 1")
    if mp.get_start_method() != "fork" and any(n > 1 for n in args.workers):
        parser.error("PR #1134 currently requires fork; test other contexts with the review probes")
    native = Path(kbmod.search.__file__)
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "native_extension": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "astropy": astropy.__version__,
        "pyarrow": pyarrow.__version__,
        "start_method": mp.get_start_method(),
        "rows": args.rows,
        "shape": args.shape,
        "dtype": "float32",
        "seed": args.seed,
        "directory": str(args.directory.resolve()),
        "timer": "perf_counter, no fsync; full write_column call",
        "trials": [],
    }
    results = fixture(module.Results, args.rows, args.shape, args.seed)
    report["input_sha256"] = hashlib.sha256(results["stamp"].data.tobytes()).hexdigest()
    order = list(args.workers) * args.repeats
    random.Random(args.seed).shuffle(order)
    expected_digest = None
    with tempfile.TemporaryDirectory(prefix="kbmod-pr1134-bench-", dir=args.directory) as tmp:
        # Put worker scratch and final output on the same measured filesystem.
        tempfile.tempdir = tmp
        path = Path(tmp) / "stamp.fits"
        for workers in order:
            started = time.perf_counter()
            results.write_column(
                "stamp", path, is_image=True, **({"num_workers": workers} if supported else {})
            )
            seconds = time.perf_counter() - started
            digest, max_error = verify(path, results)
            if expected_digest is None:
                expected_digest = digest
            assert digest == expected_digest, "Worker counts produced different pixel values"
            trial = {
                "workers": workers,
                "seconds": seconds,
                "stamps_per_second": args.rows / seconds,
                "bytes": path.stat().st_size,
                "decoded_sha256": digest,
                "max_abs_quantization_error": max_error,
                "load_average_after": os.getloadavg(),
            }
            report["trials"].append(trial)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(trial), flush=True)
            path.unlink()
    report["medians_seconds"] = {
        str(n): statistics.median(t["seconds"] for t in report["trials"] if t["workers"] == n)
        for n in args.workers
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
