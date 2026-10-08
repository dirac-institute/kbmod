# Parallel injection smoke-test branch

`kbmod.injection_parallel.inject_sources_to_workunit()` partitions a Butler-backed
ImageCollection into complete MJD groups, runs existing serial injection on each
shard, and writes a sharded WorkUnit. Each process owns one read-only Butler.

This is an opt-in CPU path. The existing `inject_sources_into_ic()` API is unchanged.

## API

```python
from kbmod.injection_parallel import inject_sources_to_workunit

output_path, injected_catalog = inject_sources_to_workunit(
    ic, full_catalog, butler_config, "/scratch/run/injected.fits",
    search_config=search_config,
    injection_workers=2,
    max_images_per_shard=8,
    zero_background=False,
    constant_variance=False,
)
```

Generate the catalog once for the full collection. Images and returned catalog
contributions are in **stable MJD order**, including original ordering among images
at equal times. `injection_input_row` in WorkUnit metadata maps back to the input.
Identical timestamps do not collapse distinct detector images.

`injection_workers=1` in this new API runs serial sharding without a process pool.
It provides the comparison between the existing whole-collection call and sharding.
In **kbmod-wf**, `injection_workers=1` retains the original unsharded workflow.

The output primary and numbered image files must not already exist. Workers write
unique global image/constituent indices into temporary sibling files. Publication
uses no-clobber hard links on the same filesystem, with the primary published last.
No pixel files cross process IPC or accumulate in the parent. Float images are
written using lossless GZIP compression with quantization disabled. The persisted
masks and PSF kernels retain the existing WorkUnit conversion semantics.

## Memory and operational limits

- At most one outstanding shard per active worker; each contains at most
  `max_images_per_shard` exposures. A single MJD group exceeding the limit raises
  before loading pixels. Increase it only after checking memory requirements.
- The limit counts images, **not bytes**. A worker can hold the injected shard,
  current input/output copies, injector scratch, and WorkUnit conversion arrays
  simultaneously. Rubin/Butler caches and parent catalogs/metadata add memory.
- This first version does not estimate a total byte budget or auto-tune workers.
  Start at two workers and small shards; measure the whole process tree. The smoke
  script reports summed RSS, which can count shared library pages more than once.
- Set OpenMP/BLAS thread limits before launching. Multiply injection workers by
  concurrent workflow tasks when budgeting CPUs, memory, and Butler connections.
- Use a filesystem supporting same-filesystem hard links. An error drains/shuts
  down the pool before deleting staging. Running work is not instantly cancelled.
- Use the supported `ButlerStandardizer` with one exposure per row. Input metadata
  is copied/unpacked and worker standardizer caches are fresh. Unexpected row loss
  or reordering raises. The existing injector's no-render/RuntimeError policy is
  retained; this branch does not redefine Rubin rendering error classification.
- Catalogs and metadata remain in memory. Downstream loading/search can still
  require all final pixels. This feature only bounds injection-stage pixel residency.

## USDF smoke test

In the usual Rubin/KBMOD environment, check out `feat/parallel-injection` in both
repositories and install them through your normal environment setup. For an
existing editable installation, switching the Python branch is sufficient; no
native/GPU source changes are included here. Confirm imports resolve to these
checkouts:

```bash
python -c 'import kbmod.injection_parallel as p; import kbmod_wf; print(p.__file__); print(kbmod_wf.__file__)'
```

Run from the KBMOD repository. Replace the four input paths with a real existing
IC, its saved **input** injection catalog, Butler configuration, and search YAML.
Use a new output directory each time. The sample is capped at two detector images
from each of three distinct MJDs (at most six images); the script deliberately
reorders those input rows to test sorting.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python benchmarks/smoke_parallel_injection.py \
  --ic /path/to/input.ecsv \
  --catalog /path/to/input.ecsv.injection_input_cat.parquet \
  --butler-config /path/to/butler.yaml \
  --search-config /path/to/search.yaml \
  --output-dir /scratch/parallel-injection-smoke \
  --workers 2 --mjd-groups 3 --images-per-mjd 2 --noise-seed 314159 \
  > /scratch/parallel-injection-smoke.log 2>&1
```

Choose an allocation that fits this sample's full serial baseline as well as the
parallel runs; dimensions and source density determine the actual memory needed.
Use `--images-per-mjd 1 --mjd-groups 2` for a smaller first check.

The script runs existing whole-collection serial injection, serial sharding, then
parallel sharding. It requires actual rendered sources. It compares science,
variance, masks, PSF kernels, output catalogs, dataset ordering, constituent indices,
WCS transforms and primary metadata, and writes `summary.json` with timings and
sampled aggregate peak RSS. It is a correctness smoke test, not a reliable speedup
benchmark: small samples, process startup, and successive cache warming affect time.

The smoke explicitly sets `VisitInjectConfig.noise_seed` to a positive value
(default `--noise-seed 314159`) for all three paths and keeps shot noise enabled.
Rubin restarts that seed for each exposure, incrementing it per source; GalSim
interprets zero as system entropy. The previous smoke used that nondeterministic
zero default, so an exact pixel comparison could fail solely because of noise.
The seed and serialized `inject_config.py` are now recorded alongside the results.
See [Rubin's configuration](https://pipelines.lsst.io/py-api/lsst.source.injection.BaseInjectConfig.html)
and [GalSim's seed semantics](https://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/random.html).

Exact pixel checks are retained for this reproducible fixture. A broad relative
tolerance does not establish equivalence between independent Poisson draws,
especially near zero science values. The smoke deliberately shares a seed across
exposures; it does not introduce a production policy for independent exposure noise
or change the normal injection defaults. Real-stack verification remains necessary.

Repeat with `--zero-background --constant-variance` in a new directory for the
opt-in image modes. Capture any mismatch rather than relaxing tolerances: real
Rubin noise/task state and WCS fallback behavior must agree across shard boundaries.

Please return `summary.json` and the log (including the exception chain on failure),
plus the Rubin stack version and allocated CPUs/RAM. This local development
machine has no Rubin stack or live Butler access; a passing local synthetic process
and FITS test does not establish real Rubin science equivalence.
