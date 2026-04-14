#!/usr/bin/env python
"""Verification script for the KBMOD FITS/CSV export utility.

Run this script to test the export pipeline against real result files.

Usage:
    python test_fits_export.py [--input DIR] [--chunk-size N] [--n-uuids N]
"""

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path


def run_tests(input_dir, chunk_size=1000, n_uuids=10):
    """Run the full verification suite."""
    print("=" * 70)
    print("KBMOD FITS/CSV Export - Verification Script")
    print(f"  Input: {input_dir}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Test UUIDs: {n_uuids}")
    print("=" * 70)

    # Import modules
    try:
        from kbmod.fits_export import export_results, read_fits_export
        from kbmod.results import Results

        print("\n[OK] Imports successful")
    except ImportError as e:
        print(f"\n[FAIL] Import failed: {e}")
        return False

    # Find parquet files
    parquet_files = sorted(Path(input_dir).rglob("*.search.parquet"))
    if not parquet_files:
        print(f"  [FAIL] No .search.parquet files found in {input_dir}")
        return False

    print(f"\n  Found {len(parquet_files)} result file(s)")
    test_file = parquet_files[0]
    print(f"  Using: {test_file.name}")

    # Grab UUIDs from first chunk
    test_uuids = []
    for chunk in Results.read_table_chunks(str(test_file), chunk_size=n_uuids):
        test_uuids = [str(u) for u in chunk.table["uuid"][:n_uuids]]
        break

    print(f"  Selected {len(test_uuids)} UUIDs: {test_uuids[:3]}...")

    tmpdir = tempfile.mkdtemp(prefix="fits_export_test_")
    try:
        fits_path = Path(tmpdir) / "test_export.fits"
        csv_path = Path(tmpdir) / "test_export.csv"

        # ----- Test 1: Export subset of UUIDs -----
        print(f"\n--- Test 1: Export {n_uuids} UUIDs to FITS + CSV ---")
        t0 = time.time()
        obs_table, traj_table = export_results(
            directories=[input_dir],
            output_path=fits_path,
            output_csv=csv_path,
            uuids=test_uuids,
            chunk_size=chunk_size,
        )
        elapsed = time.time() - t0

        print(f"\n  Exported in {elapsed:.2f}s")
        print(f"  Observations: {len(obs_table)}")
        print(f"  Trajectories: {len(traj_table)}")

        # Verify counts
        assert len(traj_table) == len(
            test_uuids
        ), f"Expected {len(test_uuids)} trajectories, got {len(traj_table)}"
        print("  [OK] Trajectory count matches UUID count")

        # Check columns
        for col in ["uuid", "source_file", "obs_index", "ra_deg", "dec_deg", "mjd"]:
            assert col in obs_table.colnames, f"Missing column: {col}"
        print("  [OK] Observations table has correct columns")

        for col in ["uuid", "source_file", "x", "y", "vx", "vy", "likelihood", "flux", "obs_count"]:
            assert col in traj_table.colnames, f"Missing column: {col}"
        print("  [OK] Trajectories table has correct columns")

        # ----- Test 2: Verify FITS file -----
        print(f"\n--- Test 2: Read back FITS file ---")
        assert fits_path.exists(), "FITS file not created"
        obs_read, traj_read = read_fits_export(fits_path)

        assert len(obs_read) == len(obs_table), "Observations mismatch after read"
        assert len(traj_read) == len(traj_table), "Trajectories mismatch after read"
        print(f"  [OK] FITS file readable, {len(obs_read)} obs, {len(traj_read)} traj")

        # ----- Test 3: Verify CSV file -----
        print(f"\n--- Test 3: Verify CSV file ---")
        assert csv_path.exists(), "CSV file not created"
        with open(csv_path) as f:
            csv_lines = f.readlines()
        # +1 for header
        assert (
            len(csv_lines) == len(obs_table) + 1
        ), f"CSV line count mismatch: {len(csv_lines)} != {len(obs_table) + 1}"
        print(f"  [OK] CSV file has {len(csv_lines) - 1} data rows")

        # ----- Test 4: Sample data inspection -----
        print(f"\n--- Test 4: Sample data ---")
        if len(obs_table) > 0:
            print(f"  First observation:")
            row = obs_table[0]
            print(f"    UUID: {row['uuid']}")
            print(f"    Source: {Path(row['source_file']).name}")
            print(f"    RA/Dec: {row['ra_deg']:.6f}, {row['dec_deg']:.6f}")
            print(f"    MJD: {row['mjd']:.6f}")
            print(f"    Obs index: {row['obs_index']}")

        if len(traj_table) > 0:
            print(f"\n  First trajectory:")
            row = traj_table[0]
            print(f"    UUID: {row['uuid']}")
            print(f"    x,y: {row['x']}, {row['y']}")
            print(f"    vx,vy: {row['vx']:.3f}, {row['vy']:.3f}")
            print(f"    likelihood: {row['likelihood']:.3f}")
            print(f"    flux: {row['flux']:.3f}")
            print(f"    obs_count: {row['obs_count']}")

        # ----- Test 5: Direct file path input -----
        print(f"\n--- Test 5: Direct file path input ---")
        tmpdir2 = tempfile.mkdtemp(prefix="fits_test_direct_")
        try:
            fits_path2 = Path(tmpdir2) / "direct.fits"
            t0 = time.time()
            obs2, traj2 = export_results(
                directories=[str(test_file)],
                output_path=fits_path2,
                uuids=test_uuids[:3],
                chunk_size=chunk_size,
            )
            elapsed2 = time.time() - t0
            print(f"  Exported {len(traj2)} trajectories in {elapsed2:.2f}s")
            assert len(traj2) == 3
            print("  [OK] Direct file path input works")
        finally:
            shutil.rmtree(tmpdir2, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Test the KBMOD FITS/CSV export utility.")
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    success = run_tests(args.input, args.chunk_size, args.n_uuids)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
