"""Per-object sky tracks across the LSST-observed sky.

The injection index (:mod:`kbmod.sorcha_injection.index`) is deliberately narrow: it keeps
only the visits belonging to the ImageCollections being injected into, because that is what
makes per-patch queries cheap. This module builds the complementary wide product -- every
synthetic object's **track across the whole observed survey** -- which answers a different
set of questions:

* Which objects sweep across many patches, and are therefore the useful targets for
  cross-collection linking tests?
* Where on the sky, and over what arc, is a given ``ObjID`` injectable at all?
* What does the synthetic population look like as a function of sky position, rate and
  magnitude, restricted to sky Rubin actually observed?

"LSST-observed sky" here means every visit in the pointing database Sorcha was run against
-- 80,859 real Rubin exposures spanning 2025-04-13 to 2026-03-20 -- not the ~6% of them that
the DP2 ImageCollections happen to cover.

Two outputs are written:

``object_tracks.parquet``
    One row per object: arc, endpoints, sky bounding box, rates, magnitudes, ecliptic
    position, and how many healpix cells the track touches. Small enough to load whole
    and sort/filter interactively.

``object_positions/``
    The tracks themselves, thinned to one position per object per night, partitioned by
    population. This is what you plot, and what you query to ask "where was this object
    on night N".

The build is shard-parallel with no merge step, because Sorcha writes each object to
exactly one shard -- verified: object sets from different shards of the same population
are disjoint. So a worker can compute complete per-object statistics from its shard alone.
"""

import glob
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ALL_POPULATIONS, DEFAULT_SORCHA_ROOT
from .index import _MAG_SANITY_MAX, _MAG_SANITY_MIN, _shard_paths

logger = logging.getLogger(__name__)

_TRACK_COLUMNS = ["ObjID", "fieldMJD_TAI", "RA_deg", "Dec_deg", "trailedSourceMag", "Obj_Sun_LTC_km"]

# Sun-object distance is reported in km; KBMOD reasons in AU (``helio_guess_dist``).
_KM_PER_AU = 1.495978707e8

_META_FILENAME = "_tracks_meta.json"

# Visit tiers. Sorcha was driven by the *raw* Rubin exposure log, which includes
# acquisition, in-focus alignment and AOS-stability exposures. Those never become
# difference images, so an object "seen" in one is not an observation anybody could
# ever recover, and counting them inflates every per-object detection count.
TIER_NOT_PROCESSED = 0  # observation_type != 'science' -- acquisition/alignment
TIER_SCIENCE = 1  # science exposure, outside the DP2 processing run
TIER_DP2 = 2  # visit actually processed into DP2 difference images

# One visit table per worker process, loaded lazily and cached.
_VISIT_CACHE = {}


def _get_pointing_db_for_tracks(_unused=None):
    """The pointing database, cached per worker (shared with the index builder)."""
    from .config import DEFAULT_POINTING_DB
    from .index import _get_pointing_db

    return _get_pointing_db(DEFAULT_POINTING_DB)


def _get_visit_table(path):
    """Sorted (visit id, tier, 5-sigma depth) arrays for vectorised lookup."""
    if path not in _VISIT_CACHE:
        import pandas as pd

        df = pd.read_parquet(path).sort_values("observationId")
        _VISIT_CACHE[path] = (
            df.observationId.to_numpy(dtype=np.int64),
            df.tier.to_numpy(dtype=np.uint8),
            df.m5.to_numpy(dtype=np.float64),
        )
    return _VISIT_CACHE[path]


# Obliquity of the ecliptic at J2000, for the equatorial -> ecliptic rotation. Doing this
# by hand rather than via astropy keeps it vectorised and cheap inside the workers; for a
# sky map the sub-arcsecond difference from a rigorous frame transform is irrelevant.
_OBLIQUITY_DEG = 23.439281


def _to_ecliptic(ra_deg, dec_deg):
    """Equatorial (ICRS) to ecliptic longitude/latitude, in degrees."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    eps = np.radians(_OBLIQUITY_DEG)
    sin_b = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(eps) * np.sin(ra)
    sin_b = np.clip(sin_b, -1.0, 1.0)
    lat = np.arcsin(sin_b)
    y = np.sin(ra) * np.cos(eps) + np.tan(dec) * np.sin(eps)
    lon = np.arctan2(y, np.cos(ra))
    return np.degrees(lon) % 360.0, np.degrees(lat)


def _angsep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees, vectorised, via the haversine form."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dra, ddec = ra2 - ra1, dec2 - dec1
    h = np.sin(ddec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def _group_median(values, codes, n_groups):
    """Median of ``values`` within each integer group, skipping NaN.

    Uses a pandas groupby because it is the one reduction here that cannot be expressed
    as a ufunc ``reduceat``; everything else stays in numpy.
    """
    import pandas as pd

    med = pd.Series(values).groupby(codes).median()
    out = np.full(n_groups, np.nan)
    out[med.index.to_numpy()] = med.to_numpy()
    return out


def _process_shard(args):
    """Reduce one Sorcha shard to per-object track summaries and nightly positions."""
    path, population, mag_max, nside, visit_table_path = args

    try:
        table = pq.read_table(path, columns=_TRACK_COLUMNS)
    except Exception as exc:
        logger.warning("Skipping unreadable shard %s: %s: %s", path, type(exc).__name__, exc)
        return None, None, dict(n_in=0, skipped=True)

    n_in = table.num_rows
    if n_in == 0:
        return None, None, dict(n_in=0, skipped=False)

    obj = np.asarray(table["ObjID"].to_pandas(), dtype=object).astype(str)
    mjd = table["fieldMJD_TAI"].to_numpy(zero_copy_only=False)
    ra = table["RA_deg"].to_numpy(zero_copy_only=False)
    dec = table["Dec_deg"].to_numpy(zero_copy_only=False)
    mag = table["trailedSourceMag"].to_numpy(zero_copy_only=False)
    dist = table["Obj_Sun_LTC_km"].to_numpy(zero_copy_only=False) / _KM_PER_AU

    keep = np.isfinite(mag) & (mag >= _MAG_SANITY_MIN) & (mag < _MAG_SANITY_MAX)
    if mag_max is not None:
        keep &= mag < mag_max
    keep &= np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mjd)
    if not keep.any():
        return None, None, dict(n_in=n_in, skipped=False)
    obj, mjd, ra, dec, mag, dist = obj[keep], mjd[keep], ra[keep], dec[keep], mag[keep], dist[keep]

    # --- resolve every detection to its visit, and that visit's tier and real depth ---
    from .visits import PointingDB

    pdb = _get_pointing_db_for_tracks(visit_table_path)
    v_ids, v_tier, v_m5 = _get_visit_table(visit_table_path)
    visit, _ = pdb.visits_for_field_times(mjd)
    pos = np.searchsorted(v_ids, visit)
    pos = np.clip(pos, 0, len(v_ids) - 1)
    hit = v_ids[pos] == visit
    tier = np.where(hit, v_tier[pos], TIER_NOT_PROCESSED).astype(np.uint8)
    m5 = np.where(hit, v_m5[pos], np.nan)
    # "Bright" means brighter than that visit's measured 5-sigma point-source limit --
    # single-visit detectability, before any shift-and-stack gain.
    bright = (tier == TIER_DP2) & np.isfinite(m5) & (mag < m5)

    # Sort by (object, time) so each object's track is one contiguous, time-ordered run.
    order = np.lexsort((mjd, obj))
    obj, mjd, ra, dec, mag, dist = obj[order], mjd[order], ra[order], dec[order], mag[order], dist[order]
    tier, bright = tier[order], bright[order]

    boundaries = np.flatnonzero(np.r_[True, obj[1:] != obj[:-1]])
    counts = np.diff(np.r_[boundaries, len(obj)])
    starts = boundaries
    n_obj = len(starts)
    obj_ids = obj[starts]

    # --- per-step motion, only within an object (mask out the cross-object steps) ---
    step_sep = np.zeros(len(obj))
    step_dt = np.zeros(len(obj))
    if len(obj) > 1:
        same = obj[1:] == obj[:-1]
        sep = _angsep_deg(ra[:-1], dec[:-1], ra[1:], dec[1:])
        dt = mjd[1:] - mjd[:-1]
        step_sep[1:] = np.where(same, sep, 0.0)
        step_dt[1:] = np.where(same, dt, 0.0)

    # Polyline arc length per object.
    cum = np.r_[0.0, np.cumsum(step_sep)]
    arc_deg = cum[starts + counts] - cum[starts]

    # Rates in arcsec/hr, using only steps short enough to be a real sample of the motion
    # (a gap of weeks between detections says nothing about instantaneous rate).
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(step_dt > 1e-6, step_sep * 3600.0 / (step_dt * 24.0), np.nan)
    rate[step_dt > 1.0] = np.nan  # ignore steps spanning more than a day

    first_idx, last_idx = starts, starts + counts - 1
    mjd_first, mjd_last = mjd[first_idx], mjd[last_idx]
    ra_first, dec_first = ra[first_idx], dec[first_idx]
    ra_last, dec_last = ra[last_idx], dec[last_idx]

    # Integer group code per row, so reductions become ufunc.reduceat / bincount calls.
    codes = np.repeat(np.arange(n_obj), counts)

    # fmin/fmax rather than minimum/maximum so NaN rates do not poison a whole object.
    rate_max = np.fmax.reduceat(rate, starts)
    ra_min, ra_max = np.minimum.reduceat(ra, starts), np.maximum.reduceat(ra, starts)
    dec_min, dec_max = np.minimum.reduceat(dec, starts), np.maximum.reduceat(dec, starts)
    mag_min = np.minimum.reduceat(mag, starts)
    rate_med = _group_median(rate, codes, n_obj)
    mag_med = _group_median(mag, codes, n_obj)
    # Sun-object distance: the quantity KBMOD's reflex correction is parameterised by.
    dist_med = _group_median(dist, codes, n_obj)

    n_sci = np.bincount(codes, weights=(tier >= TIER_SCIENCE), minlength=n_obj).astype(np.int32)
    n_dp2 = np.bincount(codes, weights=(tier == TIER_DP2), minlength=n_obj).astype(np.int32)
    n_dp2_bright = np.bincount(codes, weights=bright, minlength=n_obj).astype(np.int32)

    nights = np.floor(mjd).astype(np.int32)
    # Rows are sorted by (object, time), so nights are non-decreasing within an object and
    # a distinct night is simply a change point -- no per-group unique() needed.
    night_key_new = np.r_[True, (obj[1:] != obj[:-1]) | (nights[1:] != nights[:-1])]
    n_nights = np.bincount(codes[night_key_new], minlength=n_obj).astype(np.int32)

    import hpgeom

    healpix = hpgeom.angle_to_pixel(nside, ra, dec, nest=True, lonlat=True, degrees=True)

    # Ecliptic position at the track midpoint -- a compact way to say where in the solar
    # system's plane the object sits, which is the natural axis for a TNO map.
    mid = starts + counts // 2
    elon, elat = _to_ecliptic(ra[mid], dec[mid])

    span_deg = _angsep_deg(ra_first, dec_first, ra_last, dec_last)

    summary = pa.table(
        {
            "ObjID": pa.array(obj_ids, type=pa.string()),
            "population": pa.array(np.full(n_obj, population), type=pa.string()),
            "n_det": pa.array(counts.astype(np.int32), type=pa.int32()),
            "n_det_science": pa.array(n_sci, type=pa.int32()),
            "n_det_dp2": pa.array(n_dp2, type=pa.int32()),
            "n_det_dp2_bright": pa.array(n_dp2_bright, type=pa.int32()),
            "n_nights": pa.array(n_nights, type=pa.int32()),
            "mjd_first": pa.array(mjd_first, type=pa.float64()),
            "mjd_last": pa.array(mjd_last, type=pa.float64()),
            "arc_days": pa.array(mjd_last - mjd_first, type=pa.float64()),
            "ra_first": pa.array(ra_first, type=pa.float64()),
            "dec_first": pa.array(dec_first, type=pa.float64()),
            "ra_last": pa.array(ra_last, type=pa.float64()),
            "dec_last": pa.array(dec_last, type=pa.float64()),
            "ra_min": pa.array(ra_min, type=pa.float64()),
            "ra_max": pa.array(ra_max, type=pa.float64()),
            "dec_min": pa.array(dec_min, type=pa.float64()),
            "dec_max": pa.array(dec_max, type=pa.float64()),
            "path_deg": pa.array(arc_deg, type=pa.float64()),
            "span_deg": pa.array(span_deg, type=pa.float64()),
            "rate_med_arcsec_hr": pa.array(rate_med, type=pa.float64()),
            "rate_max_arcsec_hr": pa.array(rate_max, type=pa.float64()),
            "mag_min": pa.array(mag_min, type=pa.float64()),
            "mag_med": pa.array(mag_med, type=pa.float64()),
            "helio_dist_au": pa.array(dist_med, type=pa.float64()),
            "ecl_lon_deg": pa.array(elon, type=pa.float64()),
            "ecl_lat_deg": pa.array(elat, type=pa.float64()),
        }
    )

    # --- nightly-thinned positions: the track itself ---
    # Keep the first detection of each (object, night). Objects are usually visited a
    # couple of times a night, and a second point hours later adds nothing to a sky track.
    sel = np.flatnonzero(night_key_new)
    # A night can contain visits of several tiers; carry the best one, so that filtering
    # the map to "DP2 only" keeps the nights on which the object really was in DP2.
    night_tier = np.maximum.reduceat(tier, sel)
    night_bright = np.maximum.reduceat(bright.astype(np.uint8), sel)
    positions = pa.table(
        {
            "ObjID": pa.array(obj[sel], type=pa.string()),
            "population": pa.array(np.full(len(sel), population), type=pa.string()),
            "night": pa.array(nights[sel], type=pa.int32()),
            "mjd": pa.array(mjd[sel], type=pa.float64()),
            "ra": pa.array(ra[sel], type=pa.float64()),
            "dec": pa.array(dec[sel], type=pa.float64()),
            "mag": pa.array(mag[sel], type=pa.float64()),
            "healpix": pa.array(healpix[sel].astype(np.int64), type=pa.int64()),
            "tier": pa.array(night_tier, type=pa.uint8()),
            "bright": pa.array(night_bright, type=pa.uint8()),
        }
    )

    return summary, positions, dict(n_in=n_in, n_obj=n_obj, n_pos=len(sel), skipped=False)


def build_object_tracks(
    out_path,
    sorcha_root=DEFAULT_SORCHA_ROOT,
    populations=("cc", "hc", "cen", "de", "re_21", "re_32"),
    mag_max=27.0,
    nside=64,
    n_workers=32,
    overwrite=False,
    write_positions=True,
    visit_table_path=None,
):
    """Build per-object sky tracks over every visit Sorcha was run against.

    Parameters
    ----------
    out_path : `str`
        Directory to write ``object_tracks.parquet`` and ``object_positions/`` into.
    sorcha_root : `str`, optional
        Directory holding one subdirectory per population.
    populations : sequence of `str`, optional
        Populations to include. Defaults to every population with complete Sorcha output;
        ``re_52`` is excluded because it was still being written.
    mag_max : `float` or `None`, optional
        Keep only detections brighter than this.
    nside : `int`, optional
        NSIDE used for the ``n_healpix`` sky-coverage measure and the ``healpix`` column.
    n_workers : `int`, optional
        Shard-level process parallelism.
    overwrite : `bool`, optional
        Replace an existing product at ``out_path``.
    write_positions : `bool`, optional
        Also write the nightly-thinned position table. Set False for just the summary.
    visit_table_path : `str`
        Parquet file with columns ``observationId``, ``tier``, ``m5``. Required: it is
        what separates real science visits from acquisition exposures, and what supplies
        the measured 5-sigma depth used for the single-visit detectability flag.

    Returns
    -------
    meta : `dict`
        Provenance and row counts, also written to ``_tracks_meta.json``.
    """
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
    pos_dir = os.path.join(out_path, "object_positions")

    tasks = []
    for pop in populations:
        shards = _shard_paths(sorcha_root, pop)
        if not shards:
            logger.warning("No shards found for population %r", pop)
        for path in shards:
            tasks.append((path, pop, mag_max, nside, visit_table_path))
    if not tasks:
        raise ValueError(f"No Sorcha shards found under {sorcha_root} for {populations}")
    if visit_table_path is None or not os.path.exists(str(visit_table_path)):
        raise ValueError(
            "visit_table_path is required -- without it every acquisition exposure would "
            "be counted as an observation. Build it from the pointing database plus the "
            "DP2 visit list."
        )

    logger.info("Building tracks from %d shards across %d populations", len(tasks), len(populations))

    t0 = time.time()
    summaries, n_in_total, n_skipped, n_pos_total, done = [], 0, 0, 0, 0
    pos_writers = {}

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_process_shard, t) for t in tasks]
            for fut in as_completed(futures):
                summary, positions, stat = fut.result()
                done += 1
                n_in_total += stat["n_in"]
                n_skipped += int(stat["skipped"])
                if summary is not None:
                    summaries.append(summary)
                if write_positions and positions is not None and positions.num_rows:
                    # Stream positions straight to disk, one file per population; the
                    # resonant populations alone are far too large to hold in memory.
                    pop = positions["population"][0].as_py()
                    if pop not in pos_writers:
                        os.makedirs(pos_dir, exist_ok=True)
                        pos_writers[pop] = pq.ParquetWriter(
                            os.path.join(pos_dir, f"{pop}.parquet"),
                            positions.schema,
                            compression="zstd",
                        )
                    pos_writers[pop].write_table(positions)
                    n_pos_total += positions.num_rows
                if done % 100 == 0 or done == len(tasks):
                    logger.info(
                        "  %d/%d shards | %.0f Mrow read | %.1f M objects | %.1f M positions | %.0f s",
                        done,
                        len(tasks),
                        n_in_total / 1e6,
                        sum(s.num_rows for s in summaries) / 1e6,
                        n_pos_total / 1e6,
                        time.time() - t0,
                    )
    finally:
        for w in pos_writers.values():
            w.close()

    if not summaries:
        raise ValueError("Every shard filtered to zero objects -- check mag_max")

    combined = pa.concat_tables(summaries)
    combined = combined.sort_by([("population", "ascending"), ("path_deg", "descending")])
    pq.write_table(combined, os.path.join(out_path, "object_tracks.parquet"), compression="zstd")

    meta = {
        "sorcha_root": sorcha_root,
        "populations": list(populations),
        "mag_max": mag_max,
        "nside": int(nside),
        "sky_scope": "all visits in the Sorcha pointing database (LSST-observed sky)",
        "n_shards": len(tasks),
        "n_shards_skipped": n_skipped,
        "n_rows_scanned": int(n_in_total),
        "n_objects": int(combined.num_rows),
        "n_positions_nightly": int(n_pos_total),
        "visit_table": visit_table_path,
        "tiers": {
            "0": "not processed (acquisition/alignment)",
            "1": "science exposure outside DP2",
            "2": "DP2-processed visit",
        },
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_path, _META_FILENAME), "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "Tracks written to %s: %d objects, %d nightly positions, %d rows scanned in %.0f s",
        out_path,
        meta["n_objects"],
        meta["n_positions_nightly"],
        meta["n_rows_scanned"],
        meta["build_seconds"],
    )
    return meta
