"""Bounded, MJD-grouped injection into a disk-backed WorkUnit.

Workers receive metadata and catalogs, never parent Butler/standardizer objects.
Pixel arrays stay in workers; only metadata and catalogs cross back to the parent.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
import logging
import multiprocessing
from multiprocessing.util import Finalize
from numbers import Integral
from pathlib import Path
import os
import tempfile

import numpy as np
from astropy.io import fits
from astropy.table import vstack

from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.image_collection import ImageCollection, unpack_table
from kbmod.injection import inject_sources_into_ic
from kbmod.work_unit import WorkUnit, add_image_data_to_hdul

logger = logging.getLogger(__name__)
_worker_butler = None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def partition_injection_shards(obstimes, max_images_per_shard=8):
    """Return stable MJD-sorted input-row groups, bounded by image count.

    Equal timestamps are never split. An oversized timestamp group is rejected
    before any pixels are loaded. The count bounds image residency, not bytes:
    allow for image dimensions, injector copies, and WorkUnit conversion.
    """
    _positive_integer(max_images_per_shard, "max_images_per_shard")
    if np.any(np.ma.getmaskarray(obstimes)):
        raise ValueError("obstimes must not contain masked values.")
    times = np.asarray(obstimes, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("obstimes must be a finite one-dimensional array.")
    order = np.argsort(times, kind="stable")
    if len(order) == 0:
        return []
    groups = np.split(order, np.flatnonzero(np.diff(times[order]) != 0) + 1)
    shards, current = [], []
    for group in groups:
        if len(group) > max_images_per_shard:
            raise ValueError(
                f"MJD {times[group[0]]} has {len(group)} images, exceeding "
                f"max_images_per_shard={max_images_per_shard}. Increase the limit "
                "only after checking the per-worker memory allocation."
            )
        if len(current) + len(group) > max_images_per_shard:
            shards.append(np.asarray(current, dtype=int))
            current = []
        current.extend(group)
    if current:
        shards.append(np.asarray(current, dtype=int))
    return shards


def _metadata_only(ic):
    """Copy/unpack rows without copying or retaining cached standardizers."""
    if len(ic) == 0:
        raise ValueError("Injection requires a nonempty ImageCollection.")
    data = unpack_table(ic.data.copy(copy_data=True))
    required = {"dataId", "mjd_mid", "std_name", "std_idx", "ext_idx", "config"}
    missing = required.difference(data.colnames)
    if missing:
        raise ValueError(f"Missing Butler injection metadata: {sorted(missing)}")
    if np.any(data["std_name"] != "ButlerStandardizer") or np.any(data["ext_idx"] != 0):
        raise ValueError("Parallel injection requires one Butler exposure per ImageCollection row.")
    return data


def _new_butler(config):
    from lsst.daf.butler import Butler

    return Butler(config, writeable=False)


def _close_butler(butler):
    # Older Rubin releases do not expose close(). Process exit still releases
    # their connection; the parent never passes its live Butler to a child.
    close = getattr(butler, "close", None)
    if close is not None:
        close()


def _initialize_worker(butler_config):
    global _worker_butler
    _worker_butler = _new_butler(butler_config)
    Finalize(None, _close_butler, args=(_worker_butler,), exitpriority=10)


@dataclass
class _ShardJob:
    number: int
    rows: object
    catalog: object
    input_indices: object
    offset: int
    directory: str
    filename: str
    search_config_yaml: str
    inject_config_text: object
    injection_options: dict


def _write_shard(work, job):
    """Write directly with global image/constituent indices; return no pixels."""
    count = len(job.rows)
    if work.get_num_images() != count or work.n_constituents != count:
        raise RuntimeError("Injection shard lost images during WorkUnit conversion.")
    if work._per_image_indices != [[i] for i in range(count)]:
        raise RuntimeError("Injection shards must have one constituent per image.")
    for local_index in range(count):
        global_index = job.offset + local_index
        with fits.HDUList() as hdul:
            add_image_data_to_hdul(
                hdul,
                global_index,
                work.im_stack.sci[local_index],
                work.im_stack.var[local_index],
                work.im_stack.get_mask(local_index),
                work.im_stack.times[local_index],
                psf_kernel=work.im_stack.psfs[local_index],
                wcs=work.get_wcs(local_index),
                compression_type="GZIP_1",
                quantize_level=0.0,
            )
            hdul[f"SCI_{global_index}"].header["NIND"] = 1
            hdul[f"SCI_{global_index}"].header["IND_0"] = global_index
            hdul.writeto(Path(job.directory) / f"{global_index}_{job.filename}")
    metadata = work.org_img_meta.copy(copy_data=True)
    metadata["injection_input_row"] = np.asarray(job.input_indices, dtype=int)
    # Keep provenance standardizer references globally consistent, too.
    metadata["std_idx"] = np.arange(job.offset, job.offset + count)
    return metadata, np.asarray(work.im_stack.times)


def _inject_shard(job, butler=None):
    butler = _worker_butler if butler is None else butler
    rows = job.rows
    rows["std_idx"] = np.arange(len(rows))
    rows.meta["n_stds"] = len(rows)
    ic = ImageCollection(rows)
    inject_config = None
    if job.inject_config_text is not None:
        from lsst.source.injection import VisitInjectConfig

        inject_config = VisitInjectConfig()
        inject_config.loadFromString(job.inject_config_text)
    try:
        injected, catalog = inject_sources_into_ic(
            ic,
            job.catalog,
            butler,
            inject_config=inject_config,
            **job.injection_options,
        )
        if len(injected) != len(rows) or not np.array_equal(
            np.asarray(injected.data["dataId"]).astype(str),
            np.asarray(rows["dataId"]).astype(str),
        ):
            raise RuntimeError("Injection reconstruction changed the image count or dataset order.")
        work = injected.toWorkUnit(
            search_config=SearchConfiguration.from_yaml(job.search_config_yaml),
            butler=butler,
        )
        metadata, times = _write_shard(work, job)
        return job.number, metadata, times, catalog, work.observatory
    except Exception as exc:
        raise RuntimeError(
            f"Injection shard {job.number} failed for input rows {list(job.input_indices)}."
        ) from exc


def _bounded_results(jobs, workers, initializer, initargs, worker):
    """Admit at most one job per worker; discard each completed future promptly."""
    jobs = iter(jobs)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        pending = set()
        try:
            for _ in range(workers):
                job = next(jobs, None)
                if job is not None:
                    pending.add(pool.submit(worker, job))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                while done:
                    future = done.pop()
                    result = future.result()
                    del future
                    yield result
                    del result
                    job = next(jobs, None)
                    if job is not None:
                        pending.add(pool.submit(worker, job))
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def _publish(staging, output_path, count):
    """Publish new image files, then the primary; never overwrite old results."""
    created = []
    try:
        for name in [f"{i}_{output_path.name}" for i in range(count)] + [output_path.name]:
            target = output_path.parent / name
            # Staging is on the same filesystem. Hard links provide no-clobber
            # publication and avoid an additional copy of all the pixel files.
            os.link(Path(staging) / name, target)
            created.append(target)
    except BaseException:
        for target in reversed(created):
            target.unlink()
        raise


def inject_sources_to_workunit(
    ic,
    catalog,
    butler_config,
    output_path,
    *,
    injection_workers=1,
    max_images_per_shard=8,
    search_config=None,
    inject_config=None,
    variance_scale=1.0,
    zero_background=False,
    constant_variance=False,
):
    """Inject MJD-grouped shards and publish a lossless sharded WorkUnit.

    Unlike ``inject_sources_into_ic``, this API returns ``(output_path, catalog)``
    and retains no final pixel collection in the parent. Images and catalog
    contributions are in stable MJD order; ``injection_input_row`` in the WorkUnit
    metadata maps back to the input. Input metadata and cached exposures are not
    mutated. Generate the full source catalog before calling this function.

    ``butler_config`` is a repository path/URI or serializable Butler configuration,
    including any collection/default settings needed for dataset resolution.
    Each process creates its own read-only Butler. ``injection_workers=1`` runs
    the same sharding path directly, useful for serial-versus-sharded validation.
    ``max_images_per_shard`` is a strict image-count cap, not a byte limit. Equal
    MJD groups larger than this cap are rejected. Allow for runtime, image copies,
    and WorkUnit conversion when selecting both limits. Catalogs/metadata remain
    in memory, and worker/runtime caches are not bounded by this count.

    Output must not exist. Pixel files are written in a temporary sibling directory
    and published before the primary file. A failure cleans the temporary output;
    no partial WorkUnit is reported as successful. Native thread limits and total
    concurrency across workflow tasks must be set by the caller before spawning.
    """
    _positive_integer(injection_workers, "injection_workers")
    _positive_integer(max_images_per_shard, "max_images_per_shard")
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError("variance_scale must be positive and finite.")
    if constant_variance and variance_scale != 1.0:
        raise ValueError("constant_variance cannot be combined with variance_scale != 1.0.")
    data = _metadata_only(ic)
    shards = partition_injection_shards(data["mjd_mid"], max_images_per_shard)
    if "obstime" not in catalog.colnames:
        raise ValueError("Injection catalog requires an obstime column.")
    output_path = Path(output_path).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for name in [output_path.name] + [f"{i}_{output_path.name}" for i in range(len(data))]:
        if (output_path.parent / name).exists():
            raise FileExistsError(output_path.parent / name)
    config = SearchConfiguration() if search_config is None else search_config
    config_yaml = config.to_yaml()
    config_text = None if inject_config is None else inject_config.saveToString()
    options = dict(
        variance_scale=variance_scale,
        zero_background=zero_background,
        constant_variance=constant_variance,
    )
    workers = min(int(injection_workers), len(shards))
    logger.info(
        "Injecting %d images in %d MJD shards with %d workers; at most %d images per shard.",
        len(data),
        len(shards),
        workers,
        max_images_per_shard,
    )
    with tempfile.TemporaryDirectory(prefix=".kbmod-injection-", dir=output_path.parent) as staging:

        def jobs():
            offset = 0
            for number, indices in enumerate(shards):
                rows = data[indices].copy(copy_data=True)
                sources = catalog[np.isin(catalog["obstime"], rows["mjd_mid"])].copy(copy_data=True)
                yield _ShardJob(
                    number,
                    rows,
                    sources,
                    indices,
                    offset,
                    staging,
                    output_path.name,
                    config_yaml,
                    config_text,
                    options,
                )
                offset += len(indices)

        results = [None] * len(shards)
        if workers == 1:
            butler = _new_butler(butler_config)
            try:
                for job in jobs():
                    result = _inject_shard(job, butler=butler)
                    results[result[0]] = result
            finally:
                _close_butler(butler)
        else:
            # Close the generator (and wait for running workers) before removing
            # staging even if result validation raises in the consumer.
            with closing(
                _bounded_results(jobs(), workers, _initialize_worker, (butler_config,), _inject_shard)
            ) as stream:
                for result in stream:
                    number = result[0]
                    if not 0 <= number < len(results) or results[number] is not None:
                        raise RuntimeError("Duplicate or invalid injection shard result.")
                    results[number] = result
        if any(result is None for result in results):
            raise RuntimeError("Missing injection shard result.")
        metadata = vstack([result[1] for result in results], metadata_conflicts="silent")
        expected = np.concatenate(shards)
        if not np.array_equal(metadata["injection_input_row"], expected):
            raise RuntimeError("Injection shard row mapping is incomplete or out of order.")
        times = np.concatenate([result[2] for result in results])
        catalogs = vstack([result[3] for result in results], metadata_conflicts="silent")
        combined = WorkUnit(
            ImageStackPy(),
            config,
            lazy=True,
            org_image_meta=metadata,
            per_image_indices=[[i] for i in range(len(data))],
            obstimes=times,
            file_paths=[str(output_path.parent / f"{i}_{output_path.name}") for i in range(len(data))],
            observatory=results[0][4],
        )
        with combined.metadata_to_hdul() as hdul:
            hdul.writeto(Path(staging) / output_path.name)
        _publish(staging, output_path, len(data))
    return str(output_path), catalogs
