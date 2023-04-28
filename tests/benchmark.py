"""Benchmark the barycentric correction

This script benchmarks the barycentric correction function of run_search.
Usage: python -m pytest tests/benchmark.py
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


def test_benchmark(benchmark):
    """Benchmark the barycentric correction"""
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
