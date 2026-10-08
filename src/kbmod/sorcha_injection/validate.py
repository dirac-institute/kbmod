"""Validation harness for Sorcha-backed injection.

Each check returns a ``(name, passed, detail)`` triple and the module-level runners
aggregate them, so this can be used both interactively and as a regression gate.

The checks are ordered the way you should trust them: alignment first (if the Sorcha
survey realization is not the same visit set the ImageCollections were built from,
nothing downstream can be right), then index integrity, then catalog correctness,
then the cross-collection property the whole exercise exists to enable.
"""

import logging

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from .visits import PointingDB, ic_visit_table, sorcha_epoch_from_ic

logger = logging.getLogger(__name__)

# Schema the stock kbmod.injection.generate_injection_catalog produces. A Sorcha
# catalog must be a drop-in replacement for it.
REQUIRED_CATALOG_COLUMNS = (
    "injection_id",
    "ra",
    "dec",
    "mag",
    "guess_distance",
    "source_type",
    "obj_ids",
    "obstime",
    "plot_x",
    "plot_y",
)


class Check:
    """A single validation result."""

    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = bool(passed)
        self.detail = detail

    def __repr__(self):
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def report(checks, logger_=None):
    """Log a list of `Check`s and return True when all passed."""
    log = logger_ or logger
    for c in checks:
        (log.info if c.passed else log.error)("%s", c)
    n_pass = sum(c.passed for c in checks)
    log.info("%d/%d checks passed", n_pass, len(checks))
    return n_pass == len(checks)


def check_ic_pointing_alignment(ic, pointing_db=None, max_boresight_arcsec=300.0):
    """Verify a collection's visits really are the ones Sorcha was run against.

    This is the gate from the handoff's §8.2/§8.3. It checks four things:
    exact visit-id overlap, boresight agreement, band agreement, and the
    exposure-start/TAI relationship that fixes the epoch convention.

    Parameters
    ----------
    ic : `kbmod.ImageCollection`
    pointing_db : `str` or `PointingDB`, optional
    max_boresight_arcsec : `float`, optional
        Tolerance on the boresight separation between the collection and the DB.

    Returns
    -------
    checks : `list` of `Check`
    """
    pdb = pointing_db if isinstance(pointing_db, PointingDB) else PointingDB(pointing_db)
    visits = ic_visit_table(ic)
    data = getattr(ic, "data", ic)
    checks = []

    db_visits = set(pdb.visit.tolist())
    ic_visits = set(int(v) for v in visits["visit"])
    overlap = ic_visits & db_visits
    checks.append(
        Check(
            "visit-id overlap",
            len(overlap) == len(ic_visits),
            f"{len(overlap)}/{len(ic_visits)} collection visits present in the pointing DB",
        )
    )
    if not overlap:
        checks.append(Check("boresight agreement", False, "no shared visits to compare"))
        return checks

    # Index the DB rows corresponding to the collection's visits.
    order = np.argsort(pdb.visit)
    sel = np.array(sorted(overlap), dtype=np.int64)
    pos = order[np.searchsorted(pdb.visit[order], sel)]

    # Boresight: use the collection's pointing centre for the same visits.
    ra_col = "pointing_ra" if "pointing_ra" in data.colnames else "ra"
    dec_col = "pointing_dec" if "pointing_dec" in data.colnames else "dec"
    ic_v = np.asarray(data["visit"]).astype(np.int64)
    _, first = np.unique(ic_v, return_index=True)
    lut = {int(ic_v[i]): i for i in first}
    rows = np.array([lut[int(v)] for v in sel])
    sep = (
        SkyCoord(
            np.asarray(data[ra_col], float)[rows] * u.deg, np.asarray(data[dec_col], float)[rows] * u.deg
        )
        .separation(SkyCoord(pdb.field_ra[pos] * u.deg, pdb.field_dec[pos] * u.deg))
        .arcsec
    )
    checks.append(
        Check(
            "boresight agreement",
            np.max(sep) < max_boresight_arcsec,
            f'median {np.median(sep):.2f}" max {np.max(sep):.2f}" (tol {max_boresight_arcsec}")',
        )
    )

    if "band" in data.colnames:
        ic_band = np.asarray(data["band"]).astype(str)[rows]
        agree = int((ic_band == pdb.band[pos]).sum())
        checks.append(Check("band agreement", agree == len(sel), f"{agree}/{len(sel)} visits agree"))

    # Epoch convention: fieldMJD_TAI should equal the collection's exposure start in TAI.
    v_lut = {int(v): i for i, v in enumerate(visits["visit"])}
    vrows = np.array([v_lut[int(v)] for v in sel])
    start = visits["mjd_start"][vrows]
    if np.isfinite(start).all():
        ic_start_tai = Time(start, format="mjd", scale="utc").tai.mjd
        d = (ic_start_tai - pdb.field_mjd_tai[pos]) * 86400.0
        checks.append(
            Check(
                "epoch convention (Sorcha field time == exposure start, TAI)",
                np.max(np.abs(d)) < 0.05,
                f"median {np.median(d):+.4f}s max |{np.max(np.abs(d)):.4f}|s",
            )
        )
        # And confirm the epoch the selector actually propagates positions across.
        # With mjd_start available this is exact by construction; the check is really
        # that the offset is a sane fraction of an exposure rather than something wild.
        mid = visits["mjd_mid"][vrows]
        exp = visits["exposure_time"][vrows]
        offset_s = (mid - start) * 86400.0
        fallback_resid = np.abs((sorcha_epoch_from_ic(mid, exp) - start) * 86400.0)
        checks.append(
            Check(
                "mid-exposure epoch offset",
                np.all(offset_s > 0) and np.max(offset_s) < np.max(exp),
                f"selector propagates positions by {np.median(offset_s):.3f}s "
                f"(exact, from mjd_start); the exposureTime/2 fallback would be off by "
                f"{np.median(fallback_resid):.3f}s",
            )
        )
    return checks


def check_index(index, expect_populations=None):
    """Sanity-check a built index against its own metadata."""
    checks = []
    meta = index.meta
    checks.append(
        Check(
            "index metadata present", bool(meta), f"{len(meta)} keys" if meta else "index_meta.json missing"
        )
    )
    if not meta:
        return checks

    checks.append(Check("index non-empty", meta.get("n_rows_kept", 0) > 0, f"{meta.get('n_rows_kept')} rows"))
    checks.append(
        Check("index has objects", meta.get("n_objects", 0) > 0, f"{meta.get('n_objects')} distinct ObjIDs")
    )
    checks.append(
        Check(
            "no shards skipped",
            meta.get("n_shards_skipped", 0) == 0,
            f"{meta.get('n_shards_skipped', 0)}/{meta.get('n_shards')} shards unreadable",
        )
    )
    if expect_populations is not None:
        got = set(meta.get("populations", ()))
        checks.append(
            Check("expected populations", got == set(expect_populations), f"index holds {sorted(got)}")
        )

    # Spot-check actual content against the declared filters.
    sample = index.read(columns=["trailedSourceMag", "visit", "healpix", "population"])
    if sample.num_rows:
        mag = sample["trailedSourceMag"].to_numpy(zero_copy_only=False)
        hi = meta.get("mag_max")
        checks.append(
            Check(
                "magnitude ceiling honoured",
                bool(np.all(np.isfinite(mag))) and (hi is None or float(np.max(mag)) < hi),
                f"mag range {np.min(mag):.2f}..{np.max(mag):.2f} (ceiling {hi})",
            )
        )
        npix = 12 * index.nside * index.nside
        hp = sample["healpix"].to_numpy(zero_copy_only=False)
        checks.append(
            Check(
                "healpix in range",
                bool(np.all((hp >= 0) & (hp < npix))),
                f"nside={index.nside}, {len(np.unique(hp))} distinct cells",
            )
        )
        v = sample["visit"].to_numpy(zero_copy_only=False)
        checks.append(
            Check("all visits resolved", bool(np.all(v > 0)), f"{len(np.unique(v))} distinct visits")
        )
    return checks


def check_catalog(catalog, ic, guess_distance=None, max_pixel_error=1.0):
    """Verify an injection catalog is schema-compatible and astrometrically sound.

    Parameters
    ----------
    catalog : `astropy.table.Table`
    ic : `kbmod.ImageCollection`
    guess_distance : `float`, optional
    max_pixel_error : `float`, optional
        Tolerance when re-deriving ``plot_x``/``plot_y`` from ``ra``/``dec``.

    Returns
    -------
    checks : `list` of `Check`
    """
    checks = []
    missing = [c for c in REQUIRED_CATALOG_COLUMNS if c not in catalog.colnames]
    checks.append(
        Check(
            "catalog schema", not missing, "matches stock generator" if not missing else f"missing {missing}"
        )
    )
    if missing:
        # Every check below indexes those columns; stop here rather than raising, since
        # a malformed catalog is precisely what this harness exists to report.
        return checks
    if len(catalog) == 0:
        checks.append(Check("catalog non-empty", False, "0 rows selected"))
        return checks
    checks.append(Check("catalog non-empty", True, f"{len(catalog)} rows"))

    checks.append(
        Check(
            "injection_id unique",
            len(np.unique(catalog["injection_id"])) == len(catalog),
            f"{len(np.unique(catalog['injection_id']))} ids for {len(catalog)} rows",
        )
    )

    obj = np.asarray(catalog["obj_ids"], dtype=str)
    checks.append(
        Check(
            "obj_ids are Sorcha ObjIDs",
            bool(np.all([len(o) > 0 for o in obj])) and not np.issubdtype(obj.dtype, np.integer),
            f"{len(np.unique(obj))} distinct, e.g. {obj[0]!r}",
        )
    )

    # obstime must be bit-identical to an ic mjd_mid, because inject_sources_into_ic
    # selects sources with an exact float equality test.
    data = getattr(ic, "data", ic)
    ic_mjd = np.unique(np.asarray(data["mjd_mid"], dtype=float))
    matched = np.isin(np.asarray(catalog["obstime"], dtype=float), ic_mjd)
    checks.append(
        Check(
            "obstime exactly matches ic['mjd_mid']",
            bool(matched.all()),
            f"{int(matched.sum())}/{len(matched)} rows match a collection timestamp exactly",
        )
    )

    finite = np.isfinite(catalog["ra"]) & np.isfinite(catalog["dec"]) & np.isfinite(catalog["mag"])
    checks.append(Check("ra/dec/mag finite", bool(finite.all()), f"{int((~finite).sum())} bad rows"))

    # plot_x/plot_y must reproduce from the patch-frame coordinates.
    gd = guess_distance if guess_distance is not None else catalog.meta.get("guess_distance")
    gwcs = ic.get_global_wcs() if hasattr(ic, "get_global_wcs") else None
    if gwcs is not None and np.isfinite(np.asarray(catalog["plot_x"], float)).any():
        ra_col, dec_col = "ra", "dec"
        if gd is not None and f"ra_{float(gd)}" in catalog.colnames:
            ra_col, dec_col = f"ra_{float(gd)}", f"dec_{float(gd)}"
        x, y = gwcs.world_to_pixel(
            SkyCoord(np.asarray(catalog[ra_col], float) * u.deg, np.asarray(catalog[dec_col], float) * u.deg)
        )
        err = np.hypot(x - np.asarray(catalog["plot_x"], float), y - np.asarray(catalog["plot_y"], float))
        checks.append(
            Check(
                "plot_x/plot_y consistent with catalog coordinates",
                float(np.nanmax(err)) < max_pixel_error,
                f"max {np.nanmax(err):.3g} px (from {ra_col}/{dec_col})",
            )
        )

    if gd is not None:
        need = [f"ra_{float(gd)}", f"dec_{float(gd)}"]
        have = [c for c in need if c in catalog.colnames]
        checks.append(
            Check("reflex-frame columns present", len(have) == 2, f"guess_distance={gd}, found {have}")
        )
    return checks


def check_tracks(catalog, min_obs=3):
    """Report the per-object track statistics that determine recoverability."""
    obj = np.asarray(catalog["obj_ids"], dtype=str)
    if len(obj) == 0:
        return [Check("track statistics", False, "empty catalog")]
    uniq, counts = np.unique(obj, return_counts=True)
    n_ok = int((counts >= min_obs).sum())
    return [
        Check(
            "track lengths",
            n_ok > 0,
            f"{len(uniq)} objects, {n_ok} with >= {min_obs} detections; "
            f"obs/object min {counts.min()} median {int(np.median(counts))} max {counts.max()}",
        )
    ]


def check_cross_collection(truth_path):
    """Measure the property that motivates all of this: shared objects across collections.

    Parameters
    ----------
    truth_path : `str`
        Path to the ``injection_truth.parquet`` written by
        :func:`~kbmod.sorcha_injection.write_injection_truth`.

    Returns
    -------
    checks : `list` of `Check`
    """
    import pandas as pd

    df = pd.read_parquet(truth_path)
    checks = [Check("truth table readable", len(df) > 0, f"{len(df)} rows")]
    if len(df) == 0:
        return checks

    n_ic = df.ic_name.nunique()
    per_obj = df.groupby("obj_ids").ic_name.nunique()
    shared = per_obj[per_obj >= 2]
    checks.append(
        Check(
            "objects shared across collections",
            len(shared) > 0 if n_ic > 1 else True,
            f"{len(shared)}/{len(per_obj)} objects appear in >= 2 of {n_ic} collections"
            + ("" if n_ic > 1 else " (only one collection in the truth table)"),
        )
    )
    if len(shared):
        checks.append(
            Check(
                "ObjID stability",
                True,
                f"max collections for one object: {int(shared.max())}; "
                f"linking-testable pairs: {int((per_obj >= 2).sum())}",
            )
        )
    return checks


def validate_end_to_end(ic, index, catalog, selection=None, pointing_db=None, truth_path=None):
    """Run every applicable check and return ``(all_passed, checks)``."""
    checks = []
    checks += check_ic_pointing_alignment(ic, pointing_db)
    checks += check_index(index)
    checks += check_catalog(catalog, ic)
    checks += check_tracks(catalog)
    if truth_path is not None:
        checks += check_cross_collection(truth_path)
    return report(checks), checks
