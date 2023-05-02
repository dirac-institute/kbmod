"""Benchmark the barycentric correction

This script benchmarks the barycentric correction function of run_search.
Usage: python -m pytest tests/benchmark.py
       python -m pytest tests/benchmark.py --benchmark-autosave --benchmark-compare=0001
"""
import time

import pytest

import kbmod
from kbmod.configuration import KBMODConfig

im_filepath = "../data/demo"
v_arr = [0, 20, 21]
ang_arr = [0.5, 0.5, 11]

input_parameters = {
    # Required
    "im_filepath": im_filepath,
    "res_filepath": None,
    "time_file": None,
    "output_suffix": "DEMO",
    "v_arr": v_arr,
    "ang_arr": ang_arr,
    "bary_dist": 50.0,
}


@pytest.mark.benchmark(
    min_time=0.8,
    max_time=2.0,
    min_rounds=10,
    calibration_precision=20,
    # timer=time.time,
    # disable_gc=True,
    # warmup=False
    group="barycentric correction",
)
def test_benchmark(benchmark):
    """Benchmark the barycentric correction

    EXAMPLE USAGE:

    python -m pytest tests/benchmark.py
    python -m pytest tests/benchmark.py --benchmark-autosave --benchmark-compare=0001
    # Compare multiple runs and saves a histogram in tmp/hist.svg
    py.test-benchmark compare --histogram=tmp/hist 0001 0002 0003 0004 0005
    """
    # Test the calc_barycentric function of run_search
    run_search = kbmod.run_search.run_search(input_parameters)
    # Load the PSF.
    kb_interface = kbmod.analysis_utils.Interface()
    default_psf = kbmod.search.psf(run_search.config["psf_val"])

    # Load images to search
    _, img_info = kb_interface.load_images(
        run_search.config["im_filepath"],
        run_search.config["time_file"],
        run_search.config["psf_file"],
        run_search.config["mjd_lims"],
        default_psf,
        verbose=run_search.config["debug"],
    )
    baryCoeff = benchmark(run_search._calc_barycentric_corr, img_info, 50.0)
    assert baryCoeff is not None
