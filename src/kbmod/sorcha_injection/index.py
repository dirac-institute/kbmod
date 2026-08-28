"""Build and query the spatial/temporal index over the Sorcha ephemeris.

The raw Sorcha DP2 outputs are ~3 TB across ~10^9 rows, one row per
(object, visit) detection. Scanning that for every patch would be hopeless, so the
scan is done **once** into a compact partitioned parquet dataset that per-patch
queries can hit cheaply.

Three filters do nearly all the work of shrinking it:

* **Visit whitelist.** Sorcha simulated all 80,859 visits in the pointing database,
  but the DP2 ImageCollections only cover 5,068 of them (52 nights) -- about 6%.
* **Magnitude ceiling.** Sorcha's ``fiveSigmaDepth_mag`` was deliberately faked to
  34.5 for every visit so that no detection filtering would occur (see
  :func:`build_sorcha_index` notes), which means the outputs run down to mag ~32.
  Anything fainter than roughly 27 is not plausibly recoverable from 30 s LSSTCam
  difference images even when stacked, and only inflates the index.
* **Column projection.** Only 6 of the 16 Sorcha columns are needed.

The result is partitioned by ``population`` and ``night`` (integer floor of
``fieldMJD_TAI``). A patch query touches only the nights its visits fall on, and each
such partition is small enough to read whole and filter in memory. A ``healpix``
column (NEST ordering) is stored for the cheap spatial pre-filter.

Importantly the builder **resolves the visit id at build time** by snapping each
row's ``fieldMJD_TAI`` onto the pointing database, so per-patch selection is a pure
exact integer join and never has to reason about TAI/UTC again.
"""

import glob
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import ALL_POPULATIONS, DEFAULT_POPULATIONS, DEFAULT_SORCHA_ROOT
from .visits import PointingDB

logger = logging.getLogger(__name__)

# Columns read out of the raw Sorcha parquets. Projection alone is a >2x I/O saving.
_SOURCE_COLUMNS = [
    "ObjID",
    "fieldMJD_TAI",
    "RA_deg",
    "Dec_deg",
    "optFilter",
    "trailedSourceMag",
]

# Leading underscore so pyarrow's dataset discovery ignores it (it skips `_`/`.` prefixes
# by default, the same convention as `_metadata`/`_common_metadata`).
_META_FILENAME = "_index_meta.json"

# Sorcha emits a handful of anomalous rows with absurd magnitudes (>40, seen in `cen`
# and `de`). A finite-and-sane guard drops them regardless of the configured ceiling.
_MAG_SANITY_MIN = 10.0
_MAG_SANITY_MAX = 40.0


def _shard_paths(sorcha_root, population):
    """Sorted parquet shard paths for one population."""
    return sorted(glob.glob(os.path.join(sorcha_root, population, "*.parquet")))


def _process_shard(args):
    """Read one Sorcha shard and reduce it to indexable rows.

    Runs in a worker process, so all arguments must be picklable and the return value
    is kept small (the filters below typically retain well under 1% of input rows).
    """
    path, population, mag_max, nside, db_path, whitelist = args

    try:
        table = pq.read_table(path, columns=_SOURCE_COLUMNS)
    except Exception as exc:  # truncated footer on a shard still being written
        logger.warning("Skipping unreadable shard %s: %s: %s", path, type(exc).__name__, exc)
        return None, dict(path=path, n_in=0, n_out=0, n_obj_seen=0, skipped=True)

    n_in = table.num_rows
    if n_in == 0:
        return None, dict(path=path, n_in=0, n_out=0, n_obj_seen=0, skipped=False)

    # --- resolve visit ids once, on the unfiltered table ---
    pdb = _get_pointing_db(db_path)
    field_mjd = table["fieldMJD_TAI"].to_numpy(zero_copy_only=False)
    visit, visit_time = pdb.visits_for_field_times(field_mjd)
    in_visits = visit >= 0
    if whitelist is not None:
        in_visits &= np.isin(visit, whitelist)

    # Objects the whitelisted visits saw at all, before any magnitude ceiling. Reported
    # separately because "observed at least once" is a geometric question and should not
    # silently inherit whatever mag_max the index was built with. Objects never span
    # shards (Sorcha writes each to exactly one), so these per-shard counts simply add.
    n_obj_seen = (
        int(pc.count_distinct(table["ObjID"].filter(pa.array(in_visits))).as_py()) if in_visits.any() else 0
    )

    # --- magnitude cut: the most selective filter, applied on top of the visit cut ---
    mag = table["trailedSourceMag"]
    keep = pc.and_(pc.greater_equal(mag, _MAG_SANITY_MIN), pc.less(mag, _MAG_SANITY_MAX))
    if mag_max is not None:
        keep = pc.and_(keep, pc.less(mag, mag_max))
    keep = pc.and_(keep, pc.is_valid(mag))
    good = np.asarray(keep) & in_visits
    if not good.any():
        return None, dict(path=path, n_in=n_in, n_out=0, n_obj_seen=n_obj_seen, skipped=False)

    table = table.filter(pa.array(good))
    visit = visit[good]
    visit_time = visit_time[good]
    field_mjd = field_mjd[good]

    ra = table["RA_deg"].to_numpy(zero_copy_only=False)
    dec = table["Dec_deg"].to_numpy(zero_copy_only=False)

    import hpgeom

    healpix = hpgeom.angle_to_pixel(nside, ra, dec, nest=True, lonlat=True, degrees=True)

    out = pa.table(
        {
            "ObjID": table["ObjID"].cast(pa.string()),
            "population": pa.array(np.full(table.num_rows, population), type=pa.string()),
            "visit": pa.array(visit, type=pa.int64()),
            "fieldMJD_TAI": pa.array(field_mjd, type=pa.float64()),
            "visit_time": pa.array(visit_time, type=pa.float64()),
            "RA_deg": pa.array(ra, type=pa.float64()),
            "Dec_deg": pa.array(dec, type=pa.float64()),
            "optFilter": table["optFilter"].cast(pa.string()),
            "trailedSourceMag": table["trailedSourceMag"].cast(pa.float64()),
            "healpix": pa.array(healpix.astype(np.int64), type=pa.int64()),
            "night": pa.array(np.floor(field_mjd).astype(np.int32), type=pa.int32()),
        }
    )
    return out, dict(path=path, n_in=n_in, n_out=out.num_rows, n_obj_seen=n_obj_seen, skipped=False)


# One PointingDB per worker process, loaded lazily and cached.
_PDB_CACHE = {}


def _get_pointing_db(db_path):
    if db_path not in _PDB_CACHE:
        _PDB_CACHE[db_path] = PointingDB(db_path)
    return _PDB_CACHE[db_path]


def build_sorcha_index(
    out_path,
    sorcha_root=DEFAULT_SORCHA_ROOT,
    populations=DEFAULT_POPULATIONS,
    pointing_db=None,
    visit_whitelist=None,
    mag_max=27.0,
    nside=64,
    n_workers=16,
    overwrite=False,
):
    """Scan the Sorcha outputs once and write a partitioned index.

    Parameters
    ----------
    out_path : `str`
        Directory to write the parquet dataset into.
    sorcha_root : `str`, optional
        Directory holding one subdirectory per population.
    populations : sequence of `str`, optional
        Populations to include. ``re_52`` is excluded by default because it was still
        being written; shards that fail to parse are skipped with a warning either way.
    pointing_db : `str`, optional
        Path to the pointing database Sorcha was run against. Defaults to
        ``DEFAULT_POINTING_DB``.
    visit_whitelist : array-like of `int`, or `None`, optional
        Keep only rows whose resolved visit id is in this set. Pass the union of the
        ``visit`` columns of the ImageCollections you intend to inject into -- for the
        DP2 collections this is ~6% of the simulated visits and is the single biggest
        size reduction available. `None` keeps every visit.
    mag_max : `float` or `None`, optional
        Drop rows with ``trailedSourceMag`` at or above this. Rows outside
        ``[10, 40]`` are always dropped as anomalous.
    nside : `int`, optional
        NSIDE for the stored NEST ``healpix`` column.
    n_workers : `int`, optional
        Shard-level process parallelism.
    overwrite : `bool`, optional
        Replace an existing index at ``out_path``.

    Returns
    -------
    meta : `dict`
        The metadata written alongside the dataset (row counts, filters, provenance).

    Notes
    -----
    A word on why no detection-completeness correction is needed. Sorcha's outputs are
    often a *detection* table rather than a full ephemeris, which would leave gaps in
    faint objects' tracks. That is not the case here: the run had ``default_snr_cut =
    False``, both ``[FADINGFUNCTION]`` and ``[LINKINGFILTER]`` commented out, and the
    pointing database's ``fiveSigmaDepth`` hardcoded to 34.5 for all 80,859 visits, so
    every source sits ~7 mag "above depth" and nothing is photometrically filtered.
    The only losses are geometric -- ``camera_model = footprint`` with a 2" edge
    threshold, which correctly removes objects falling in chip gaps -- and the
    ``bright_limit = 16.0`` saturation cut, which is irrelevant at these magnitudes.
    Tracks are therefore complete, which is what makes whole-track injection valid.
    """
    from .config import DEFAULT_POINTING_DB

    pointing_db = pointing_db or DEFAULT_POINTING_DB
    populations = tuple(populations)
    bad = set(populations) - set(ALL_POPULATIONS)
    if bad:
        raise ValueError(f"Unknown population(s): {sorted(bad)}")

    if os.path.exists(out_path):
        if not overwrite:
            raise FileExistsError(f"{out_path} exists; pass overwrite=True to replace it")
        import shutil

        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    whitelist = None
    if visit_whitelist is not None:
        whitelist = np.unique(np.asarray(visit_whitelist, dtype=np.int64))
        logger.info("Restricting index to %d whitelisted visits", len(whitelist))

    # Validate the pointing DB up front rather than in every worker.
    pdb = PointingDB(pointing_db)

    tasks = []
    for pop in populations:
        shards = _shard_paths(sorcha_root, pop)
        if not shards:
            logger.warning("No parquet shards found for population %r under %s", pop, sorcha_root)
        for path in shards:
            tasks.append((path, pop, mag_max, nside, pointing_db, whitelist))
    if not tasks:
        raise ValueError(f"No Sorcha shards found under {sorcha_root} for {populations}")

    logger.info(
        "Scanning %d shards across %d populations with %d workers", len(tasks), len(populations), n_workers
    )

    t0 = time.time()
    tables, n_in_total, n_skipped, per_pop = [], 0, 0, {p: 0 for p in populations}
    seen_total, seen_pop = 0, {p: 0 for p in populations}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_shard, t): t for t in tasks}
        for fut in as_completed(futures):
            table, stat = fut.result()
            done += 1
            n_in_total += stat["n_in"]
            n_skipped += int(stat["skipped"])
            seen_total += stat.get("n_obj_seen", 0)
            seen_pop[futures[fut][1]] += stat.get("n_obj_seen", 0)
            if table is not None and table.num_rows:
                tables.append(table)
                per_pop[futures[fut][1]] += table.num_rows
            if done % 50 == 0 or done == len(tasks):
                logger.info(
                    "  %d/%d shards, %.1f Mrow read, %.0f krow kept, %.0f s",
                    done,
                    len(tasks),
                    n_in_total / 1e6,
                    sum(t.num_rows for t in tables) / 1e3,
                    time.time() - t0,
                )

    if not tables:
        raise ValueError("Every shard filtered to zero rows -- check mag_max and visit_whitelist")

    combined = pa.concat_tables(tables)
    # With all ten populations the concatenated string columns (e.g. ObjID) can exceed
    # the 2 GB span a 32-bit ``string`` offset can address, which overflows inside the
    # sort's take(). Promote string columns to ``large_string`` (64-bit offsets) first;
    # parquet stores both identically, so readers are unaffected.
    schema = combined.schema
    promote = [i for i, f in enumerate(schema) if pa.types.is_string(f.type)]
    if promote:
        for i in promote:
            schema = schema.set(i, schema.field(i).with_type(pa.large_string()))
        combined = combined.cast(schema)
    # Sorting by (night, healpix) makes row-group statistics useful for the spatial
    # pre-filter, on top of the directory-level night partitioning.
    combined = combined.sort_by([("night", "ascending"), ("healpix", "ascending")])

    pq.write_to_dataset(
        combined,
        root_path=out_path,
        partition_cols=["population", "night"],
        existing_data_behavior="overwrite_or_ignore",
    )

    obj_ids = combined["ObjID"].to_pandas()
    meta = {
        "sorcha_root": sorcha_root,
        "pointing_db": pointing_db,
        "populations": list(populations),
        "mag_max": mag_max,
        "mag_sanity_range": [_MAG_SANITY_MIN, _MAG_SANITY_MAX],
        "nside": int(nside),
        "nest": True,
        "n_shards": len(tasks),
        "n_shards_skipped": n_skipped,
        "n_rows_scanned": int(n_in_total),
        "n_rows_kept": int(combined.num_rows),
        "n_objects": int(obj_ids.nunique()),
        "n_objects_seen_any_mag": int(seen_total),
        "objects_seen_per_population": {k: int(v) for k, v in seen_pop.items()},
        "rows_per_population": {k: int(v) for k, v in per_pop.items()},
        "n_visits": int(pc.count_distinct(combined["visit"]).as_py()),
        "n_nights": int(pc.count_distinct(combined["night"]).as_py()),
        "visit_whitelist_size": None if whitelist is None else int(len(whitelist)),
        "pointing_db_visits": len(pdb),
        "build_seconds": round(time.time() - t0, 1),
        "epoch_note": (
            "fieldMJD_TAI is the exposure START in TAI; the KBMOD mid-exposure epoch is "
            "fieldMJD_TAI - 37s + exposureTime/2 in UTC. See kbmod.sorcha_injection.visits."
        ),
    }
    with open(os.path.join(out_path, _META_FILENAME), "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "Index written to %s: %d rows, %d objects, %d visits, %d nights in %.0f s",
        out_path,
        meta["n_rows_kept"],
        meta["n_objects"],
        meta["n_visits"],
        meta["n_nights"],
        meta["build_seconds"],
    )
    return meta


class SorchaIndex:
    """Reader for the partitioned Sorcha ephemeris index.

    Parameters
    ----------
    path : `str`
        Directory written by :func:`build_sorcha_index`.
    """

    def __init__(self, path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Sorcha index not found: {path}")
        self.path = path
        meta_path = os.path.join(path, _META_FILENAME)
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                self.meta = json.load(fh)
        else:
            self.meta = {}
            logger.warning("No %s in %s; nside/mag provenance unknown", _META_FILENAME, path)
        self.nside = int(self.meta.get("nside", 64))
        self._dataset = ds.dataset(path, format="parquet", partitioning="hive")

    def __repr__(self):
        return (
            f"<SorchaIndex {self.path}: {self.meta.get('n_rows_kept', '?')} rows, "
            f"{self.meta.get('n_objects', '?')} objects, nside={self.nside}>"
        )

    @property
    def populations(self):
        return tuple(self.meta.get("populations", ()))

    def read(
        self,
        visits=None,
        nights=None,
        populations=None,
        healpix=None,
        mag_range=None,
        bands=None,
        columns=None,
    ):
        """Pull a filtered slice of the index.

        Every argument is optional; each one that is supplied narrows the result.
        Partition-level filters (``nights``, ``populations``) prune whole directories,
        so supplying them is much cheaper than filtering afterwards.

        Parameters
        ----------
        visits : array-like of `int`, optional
            Keep only these visit ids.
        nights : array-like of `int`, optional
            Keep only these nights (integer floor of ``fieldMJD_TAI``). Partition key.
        populations : sequence of `str`, optional
            Keep only these populations. Partition key.
        healpix : array-like of `int`, optional
            Keep only these NEST healpix cells at ``self.nside``.
        mag_range : `tuple` of (`float` or `None`, `float` or `None`), optional
            Bounds on ``trailedSourceMag``.
        bands : sequence of `str`, optional
            Keep only these ``optFilter`` values.
        columns : sequence of `str`, optional
            Columns to return. Defaults to all.

        Returns
        -------
        table : `pyarrow.Table`
        """
        expr = None

        def _and(e, new):
            return new if e is None else e & new

        if nights is not None:
            nights = np.unique(np.asarray(nights, dtype=np.int32))
            expr = _and(expr, ds.field("night").isin(nights))
        if populations is not None:
            expr = _and(expr, ds.field("population").isin(list(populations)))
        if visits is not None:
            visits = np.unique(np.asarray(visits, dtype=np.int64))
            expr = _and(expr, ds.field("visit").isin(visits))
        if healpix is not None:
            healpix = np.unique(np.asarray(healpix, dtype=np.int64))
            expr = _and(expr, ds.field("healpix").isin(healpix))
        if bands is not None:
            expr = _and(expr, ds.field("optFilter").isin(list(bands)))
        if mag_range is not None:
            lo, hi = mag_range
            if lo is not None:
                expr = _and(expr, ds.field("trailedSourceMag") >= lo)
            if hi is not None:
                expr = _and(expr, ds.field("trailedSourceMag") < hi)

        return self._dataset.to_table(filter=expr, columns=columns)
