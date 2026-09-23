# FITS stamp-column benchmark — 2026-09-23

Measured source: PR #1134 after merging main, commit `627be6198c28e50f6f0ea15215807561a2156944`.
Baseline: main `04cfc13a5aea04026bb774f754f3b64fd2a71789`.

| Machine / source | Workers | Median (s) | Range (s) | Stamps/s | Speedup vs integrated serial |
|---|---:|---:|---:|---:|---:|
| Arnor / main | 1 | 31.96 | 31.91–32.08 | 156.5 | — |
| Arnor / integrated PR | 1 | 31.98 | 31.92–32.05 | 156.4 | 1.00× |
| Arnor / integrated PR | 2 | 16.20 | 16.01–16.24 | 308.6 | 1.97× |
| Arnor / integrated PR | 4 | 8.08 | 8.06–8.09 | 618.7 | 3.96× |
| Arnor / integrated PR | 8 | 4.12 | 4.12–4.13 | 1212.3 | 7.75× |
| Apple M5 (forced fork) / main | 1 | 19.70 | 18.01–19.97 | 253.8 | — |
| Apple M5 (forced fork) / integrated PR | 1 | 18.11 | 17.93–18.29 | 276.0 | 1.00× |
| Apple M5 (forced fork) / integrated PR | 2 | 9.46 | 9.36–9.50 | 528.6 | 1.92× |
| Apple M5 (forced fork) / integrated PR | 4 | 5.05 | 4.94–5.12 | 991.0 | 3.59× |
| Apple M5 (forced fork) / integrated PR | 8 | 3.87 | 3.77–3.99 | 1291.9 | 4.68× |

Each entry is the median of three complete writes of 5,000 seeded Gaussian float32 stamps, 100 × 100 pixels (seed 1134, standard deviation 10). Worker-count trial order was shuffled deterministically; main was measured separately. Times include pool startup, compression, scratch files, and concatenation, and exclude input generation, full readback verification, and deletion. No fsync or cache eviction was used. These are application write timings on local storage, not durable I/O latency or full-search timings. Hosts were shared, not reserved.

Arnor: two AMD EPYC 9555 64-core processors, 256 logical CPUs, Linux, Python 3.12.14, NumPy 2.5.3, Astropy 8.0.1, PyArrow 25.0.1; default fork; worker scratch and destination on local `/tmp`. The exact results source was loaded using an existing KBMOD environment/native extension. This benchmark exercises Python/Astropy I/O, not a rebuilt CUDA environment.

Laptop: Apple M5, 10 logical CPUs, 32 GiB RAM, macOS 26.3.1, Python 3.12.13, NumPy 2.5.2, Astropy 8.0.1, PyArrow 25.0.1; local workspace storage and rebuilt native CPU extension. **Laptop parallel numbers explicitly force fork. The current PR fails with the default spawn method and with forkserver because workers do not inherit its stamp-data globals.**

All 30 full-file readbacks (15 per host) verified every pixel against the input within the existing 0.01 quantization step, plus shape, dtype, UUID, extension ordering, and primary metadata. Main and every worker count produced the same decoded pixel SHA-256, `b5ce66d41a494f2a2196026eb745f3ab4f1865a245153ecd8a0649426cea9845`, and 129,602,880-byte FITS files. Maximum observed quantization error was 0.005001068115234375; this quantization is pre-existing. No new numerical discrepancy was observed. Byte-for-byte file identity was not asserted.

The 8-worker improvement is 7.75× on arnor and 4.68× on the laptop relative to each integrated serial median. Sequential differences from main are small on arnor and variable on the laptop; do not interpret them as a serial optimization. These results apply to this workload and local storage. Small columns, different stamp dimensions, and network filesystems can scale differently. The ordinary `run_search` call remains serial unless a future change exposes the setting there.

Correctness remains open: portable worker data transfer, scratch cleanup on every exception (including allocation), and staging the complete output before replacing the destination. Successful throughput does not resolve these defects.

## Run

Use a scientific Python environment that can import KBMOD. From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python benchmarks/bench_write_column.py --source src/kbmod/results.py \
  --rows 5000 --shape 100 100 --workers 1 2 4 8 --repeats 3 \
  --directory /tmp --output integrated.json
```

For the laptop diagnostic only, add `--start-method fork`. The benchmark deliberately refuses parallel spawn/forkserver runs until the library's portability bug is fixed; it does not select fork silently. The library default remains unchanged.

Extract the baseline with `git show 04cfc13a5aea04026bb774f754f3b64fd2a71789:src/kbmod/results.py > /tmp/main_results.py`, then repeat with `--source /tmp/main_results.py --workers 1 --output main.json`. The script records source and native-extension hashes, input and decoded hashes, dependency versions, filesystem location, trial order, per-trial times and load averages. Keep those JSON files with any future comparison.
