# FITS stamp-column benchmark — 2026-09-23, after correctness fixes

Measured writer and benchmark: `d77b8aa6cf7ac708f3884f008c9b590c0b4aa713`.
Baseline: main `04cfc13a5aea04026bb774f754f3b64fd2a71789`, remeasured in this run.

| Machine / source | Workers | Median (s) | Range (s) | Stamps/s | Speedup vs fixed serial |
|---|---:|---:|---:|---:|---:|
| Arnor / main | 1 | 31.66 | 31.64–31.68 | 157.9 | — |
| Arnor / fixed PR | 1 | 31.74 | 31.73–31.75 | 157.5 | 1.00× |
| Arnor / fixed PR | 2 | 16.03 | 15.98–16.07 | 312.0 | 1.98× |
| Arnor / fixed PR | 4 | 8.11 | 8.11–8.13 | 616.5 | 3.91× |
| Arnor / fixed PR | 8 | 4.23 | 4.21–4.24 | 1181.3 | 7.50× |
| Apple M5, default spawn / main | 1 | 19.26 | 19.24–19.27 | 259.6 | — |
| Apple M5, default spawn / fixed PR | 1 | 19.24 | 18.48–20.59 | 259.8 | 1.00× |
| Apple M5, default spawn / fixed PR | 2 | 13.47 | 13.25–13.80 | 371.1 | 1.43× |
| Apple M5, default spawn / fixed PR | 4 | 10.08 | 9.27–10.80 | 495.8 | 1.91× |
| Apple M5, default spawn / fixed PR | 8 | 9.80 | 9.06–10.04 | 510.2 | 1.96× |

Each entry is the median of three complete writes of 5,000 seeded Gaussian float32 stamps, 100 × 100 pixels (seed 1134, standard deviation 10). Worker-count trial order is deterministically shuffled; main is measured separately. Times include pool startup, argument transfer, compression, scratch files, concatenation, and publishing the staged output. They exclude input generation, verification, and deletion. There is no fsync or cache eviction. These are application write timings on local storage; the shared hosts were not reserved.

Arnor: dual AMD EPYC 9555 (128 physical/256 logical CPUs), Linux, Python 3.12.14, NumPy 2.5.3, Astropy 8.0.1, PyArrow 25.0.1, default fork. Output and staging use local `/tmp`. The exact results source is loaded into an existing KBMOD environment; no new CUDA build is involved.

Laptop: Apple M5 (10 logical CPUs), 32 GiB RAM, macOS 26.3.1, Python 3.12.13, NumPy 2.5.2, Astropy 8.0.1, PyArrow 25.0.1, local workspace storage. **These measurements use the default spawn method.** The earlier pre-fix laptop table forced fork because spawn was broken; those numbers are not a comparison under identical process settings.

At eight workers, the corrected writer scales 7.50× on arnor and 1.96× on the laptop. The prior arnor eight-worker median was 4.12 s; the corrected median is 4.23 s on the same workload and environment. Passing chunks explicitly retains the useful Linux speedup, so shared-memory machinery has not been added. Spawn startup and data transfer have visible costs on the laptop. These numbers apply to this workload; small columns and network filesystems can behave differently. The serial default is unchanged.

Every one of the 30 full-file readbacks verifies all decoded pixels, shape, dtype, UUIDs, extension ordering, and primary metadata. Main and every worker count on both hosts produce the same decoded SHA-256, `b5ce66d41a494f2a2196026eb745f3ab4f1865a245153ecd8a0649426cea9845`, also matching the pre-fix runs. All files contain 129,602,880 bytes. Maximum observed quantization error is 0.005001068115234375 under the unchanged `quantize_level=-0.01` policy. Each timed write is followed by an assertion that input pixel bytes are unchanged. Byte-for-byte FITS-file identity is not asserted.

## Correctness and file handling

The focused results suite passes 49 tests, including spawn/fork/forkserver equivalence, concurrent calls, nonfinite/integer data, migration, failure cleanup, output preservation, and a no-overwrite publication race. The writer uses explicit worker inputs and one temporary directory beside the destination. It publishes only the complete staged image-column file. `overwrite=False` publication uses a same-filesystem hard link and requires hard-link support; this is verified on the tested hosts. Per-column publication does not make multi-file Results output transactional, and it does not add an fsync guarantee. Current main's detached-column API and main-table-first ordering are preserved.

## Reproduce

From the checkout with a scientific Python environment that can import KBMOD:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python benchmarks/bench_write_column.py --source src/kbmod/results.py \
  --rows 5000 --shape 100 100 --workers 1 2 4 8 --repeats 3 \
  --directory /tmp --output fixed.json
```

The script honors the platform default; use `--start-method spawn`, `fork`, or `forkserver` to choose another available context. Spawn/forkserver workers load the same pinned source file as the parent. Library callers using spawn need the usual `if __name__ == "__main__"` guard.

Extract the baseline with `git show 04cfc13a5aea04026bb774f754f3b64fd2a71789:src/kbmod/results.py > /tmp/main_results.py`, then repeat with `--source /tmp/main_results.py --workers 1 --output main.json`. JSON outputs record source/native hashes, input/decoded hashes, dependency versions, trial order, times, file sizes, and load averages. Keep them with any future comparison. Run the focused tests with `PYTHONPATH=src python -m unittest discover -s tests -p 'test_results*.py'`.
