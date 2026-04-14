#!/usr/bin/env python
"""Verification script for the KBMOD MPC 80-column export utility.

Run this script to test the MPC export pipeline against real result files.
It can be called repeatedly for development iteration.

Usage:
    python test_mpc_export.py [--input DIR] [--chunk-size N] [--n-uuids N]
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def check_mpc_line_format(filepath):
    """Verify every line in an MPC file is exactly 80 characters."""
    errors = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            stripped = line.rstrip("\n")
            if len(stripped) != 80:
                errors.append(f"  Line {i}: length={len(stripped)}, content={repr(stripped)}")
    return errors


def run_tests(input_dir, chunk_size=1000, n_uuids=10):
    """Run the full verification suite."""
    print("=" * 70)
    print("KBMOD MPC Export - Verification Script")
    print(f"  Input: {input_dir}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Test UUIDs: {n_uuids}")
    print("=" * 70)

    # Import modules
    try:
        from kbmod.mpc_export import (
            format_mpc_line,
            format_result_to_mpc,
            export_results_to_mpc_files,
        )
        from kbmod.results import Results

        print("\n✓ Imports successful")
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False

    # ----- Test 1: format_mpc_line unit test -----
    print("\n--- Test 1: format_mpc_line ---")
    t0 = time.time()
    line = format_mpc_line(
        ra_deg=180.0,
        dec_deg=-45.0,
        mjd=60000.5,
        designation="test01",
        observatory="X05",
    )
    print(f"  Line: '{line}'")
    print(f"  Length: {len(line)}")
    assert len(line) == 80, f"Expected 80 chars, got {len(line)}"
    assert line[77:80] == "X05", f"Expected observatory X05 at cols 78-80, got '{line[77:80]}'"
    print(f"  ✓ format_mpc_line produces valid 80-char line ({time.time()-t0:.3f}s)")

    # Test edge case: Dec = -0 degrees
    line_neg_zero = format_mpc_line(
        ra_deg=10.0,
        dec_deg=-0.5,
        mjd=60000.5,
        designation="test02",
        observatory="X05",
    )
    assert len(line_neg_zero) == 80
    # Check Dec sign is negative for -0.5 deg
    # Dec starts at col 45 (index 44), sign is at that position
    assert line_neg_zero[44] == "-", f"Expected negative Dec sign, got '{line_neg_zero[44]}'"
    print(f"  ✓ Edge case: Dec near zero handled correctly")

    # Test positive dec
    line_pos = format_mpc_line(ra_deg=0.0, dec_deg=+30.5, mjd=60000.5, designation="test03")
    assert len(line_pos) == 80
    assert line_pos[44] == "+", f"Expected positive Dec sign, got '{line_pos[44]}'"
    print(f"  ✓ Edge case: positive Dec handled correctly")

    # ----- Test 2: Get a small set of UUIDs to test with -----
    print(f"\n--- Test 2: Export {n_uuids} UUIDs ---")

    # Read just the first chunk to grab some UUIDs
    parquet_files = sorted(Path(input_dir).rglob("*.search.parquet"))
    if not parquet_files:
        print(f"  ✗ No .search.parquet files found in {input_dir}")
        return False

    print(f"  Found {len(parquet_files)} result file(s)")
    test_file = parquet_files[0]
    print(f"  Using: {test_file.name}")

    # Grab UUIDs from first chunk
    test_uuids = []
    for chunk in Results.read_table_chunks(str(test_file), chunk_size=n_uuids):
        test_uuids = [str(u) for u in chunk.table["uuid"][:n_uuids]]
        break

    print(f"  Selected {len(test_uuids)} UUIDs: {test_uuids[:3]}...")

    tmpdir = tempfile.mkdtemp(prefix="mpc_test_")
    try:
        t0 = time.time()
        manifest = export_results_to_mpc_files(
            directories=[input_dir],
            output_dir=tmpdir,
            uuids=test_uuids,
            observatory="X05",
            chunk_size=chunk_size,
        )
        elapsed = time.time() - t0
        n_exported = len(manifest)
        print(f"\n  Exported {n_exported} MPC files in {elapsed:.2f}s")
        print(f"  Rate: {n_exported/elapsed:.1f} UUIDs/s" if elapsed > 0 else "")
        assert n_exported == len(test_uuids), f"Expected {len(test_uuids)}, got {n_exported}"
        print(f"  Total observations: {sum(manifest['n_obs'])}")

        # Check manifest columns
        for col in ["uuid", "search_file", "mpc_file", "n_obs"]:
            assert col in manifest.colnames, f"Missing column: {col}"
        print("  ✓ Manifest has correct columns")

        # Check manifest file exists
        manifest_path = Path(tmpdir) / "mpc_export_manifest.parquet"
        assert manifest_path.exists(), "Manifest parquet not written"
        print("  ✓ Manifest parquet file written")

        # Verify MPC files exist and have correct format
        txt_files = list(Path(tmpdir).glob("*.txt"))
        assert len(txt_files) == n_exported, f"Expected {n_exported} .txt files, got {len(txt_files)}"

        errors_found = False
        for mpc_file in txt_files:
            errs = check_mpc_line_format(mpc_file)
            if errs:
                print(f"  ✗ {mpc_file.name}: format errors:")
                for e in errs:
                    print(e)
                errors_found = True
        if not errors_found:
            print(f"  ✓ All {len(txt_files)} MPC files have valid 80-column format")

        # Show sample content
        if txt_files:
            sample_file = txt_files[0]
            print(f"\n  Sample MPC file ({sample_file.name}):")
            with open(sample_file) as f:
                for line in list(f)[:3]:
                    print(f"    {line.rstrip()}")
            # Annotate columns
            print(f"    {'|':>5}{'|':>7}{'|':>3}{'|':>17}{'|':>12}{'|':>12}{'|':>21}{'|':>3}")
            print(
                f"    {'1-5':>5}{'6-12':>7}{'13-15':>3}{'16-32':>17}{'33-44':>12}{'45-56':>12}{'57-77':>21}{'78-80':>3}"
            )

        # ----- Test 3: Direct single-file input -----
        print(f"\n--- Test 3: Direct file path input ---")
        tmpdir2 = tempfile.mkdtemp(prefix="mpc_test_direct_")
        try:
            t0 = time.time()
            manifest2 = export_results_to_mpc_files(
                directories=[str(test_file)],
                output_dir=tmpdir2,
                uuids=test_uuids[:3],
                observatory="X05",
                chunk_size=chunk_size,
            )
            elapsed2 = time.time() - t0
            print(f"  Exported {len(manifest2)} MPC files in {elapsed2:.2f}s")
            assert len(manifest2) == 3
            print("  ✓ Direct file path input works")
        finally:
            shutil.rmtree(tmpdir2, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Test the KBMOD MPC export utility.")
    parser.add_argument(
        "--input",
        type=str,
        default="/sdf/scratch/rubin/kbmod/runs/testing",
        help="Directory containing .search.parquet files to test with.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for testing (small for speed).",
    )
    parser.add_argument(
        "--n-uuids",
        type=int,
        default=10,
        help="Number of UUIDs to export in the test.",
    )
    args = parser.parse_args()
    success = run_tests(args.input, args.chunk_size, args.n_uuids)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
