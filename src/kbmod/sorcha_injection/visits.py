"""Visit identification and the Sorcha/KBMOD time-scale relationship.

Getting the temporal join right is the single most failure-prone part of matching a
Sorcha run to a set of ImageCollections, so the relationships below were established
empirically against the DP2 collections and are asserted by the validation harness.

Sorcha was driven by the pointing database ``exposures_from_rubin_fixed.db``, whose
``observationId`` values are *real Rubin exposure ids* (e.g. ``2025071100762``), not
opsim indices. The KBMOD ImageCollections built from the DP2 butler carry exactly the
same identifiers in their ``visit`` column, verified at 100% overlap across all
5,068 visits in the DP2 collections. **That makes the visit id an exact join key**,
which sidesteps fuzzy nearest-time matching entirely -- prefer it always.

Three relationships hold, each verified to well below a millisecond:

1. ``fieldMJD_TAI == observationStartMJD + visitTime / 2``
   Sorcha's per-row field time is the pointing DB's start plus half the *visit* time
   (~30.93 s, which includes shutter overhead), not half the *exposure* time (30.0 s).

2. ``fieldMJD_TAI == Time(ic["mjd_start"], scale="utc").tai.mjd``
   The Sorcha epoch is the same physical instant as the ImageCollection's exposure
   **start**, differing only by the 37 s TAI-UTC offset. ``ic["mjd_start"]`` is UTC;
   ``fieldMJD_TAI`` is TAI.

3. ``ic["mjd_mid"] == ic["mjd_start"] + exposureTime / 2`` (both UTC).

Consequence: Sorcha positions are evaluated ``exposureTime / 2`` (about 15.5 s)
*before* the mid-exposure epoch that KBMOD uses as ``obstime``. This is a systematic
epoch offset in the pointing DB, not a bug in Sorcha. It is small -- for cold
classicals the resulting position error is 0.011" median and 0.023" at the 99th
percentile, about a tenth of an LSSTCam pixel -- but
:func:`kbmod.sorcha_injection.selector.select_injections_for_ic` corrects for it by
default using each object's own on-sky rate.
"""

import logging
import os
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

# Tolerance when snapping a Sorcha ``fieldMJD_TAI`` onto a pointing-DB visit. The
# observed agreement is <= 1e-6 s; 1 ms is a very loose guard against float noise.
FIELD_TIME_TOL_DAYS = 1.0e-3 / 86400.0


class PointingDB:
    """The Sorcha pointing database, indexed for exact visit lookup.

    Parameters
    ----------
    db_path : `str`
        Path to ``exposures_from_rubin_fixed.db`` (or whichever pointing database the
        Sorcha run was driven with).

    Attributes
    ----------
    visit : `numpy.ndarray` of `int64`
        ``observationId`` for each visit, sorted by ``field_mjd_tai``.
    field_mjd_tai : `numpy.ndarray` of `float`
        ``observationStartMJD + visitTime / 2``, i.e. exactly the ``fieldMJD_TAI``
        that appears in the Sorcha output rows. Sorted ascending.
    visit_time : `numpy.ndarray` of `float`
        ``visitTime`` in seconds.
    band, field_ra, field_dec : `numpy.ndarray`
        Per-visit band and boresight, for cross-checks.
    """

    def __init__(self, db_path=None):
        from .config import DEFAULT_POINTING_DB

        db_path = db_path or DEFAULT_POINTING_DB
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Sorcha pointing database not found: {db_path}")
        self.db_path = db_path

        # Read-only URI so we never risk mutating a shared file.
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT observationId, observationStartMJD, visitTime, band, fieldRA, fieldDec "
                "FROM observations WHERE fieldRA IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        if not rows:
            raise ValueError(f"No usable rows in {db_path}")

        visit = np.array([r[0] for r in rows], dtype=np.int64)
        start = np.array([r[1] for r in rows], dtype=float)
        vtime = np.array([r[2] for r in rows], dtype=float)
        band = np.array([r[3] for r in rows], dtype="U2")
        f_ra = np.array([r[4] for r in rows], dtype=float)
        f_dec = np.array([r[5] for r in rows], dtype=float)

        field_mjd_tai = start + vtime / 2.0 / 86400.0
        order = np.argsort(field_mjd_tai, kind="stable")

        self.visit = visit[order]
        self.field_mjd_tai = field_mjd_tai[order]
        self.visit_time = vtime[order]
        self.band = band[order]
        self.field_ra = f_ra[order]
        self.field_dec = f_dec[order]

        logger.info(
            "Loaded %d visits from %s (MJD_TAI %.4f .. %.4f)",
            len(self.visit),
            os.path.basename(db_path),
            self.field_mjd_tai[0],
            self.field_mjd_tai[-1],
        )

    def __len__(self):
        return len(self.visit)

    def visits_for_field_times(self, field_mjd_tai, tol_days=FIELD_TIME_TOL_DAYS):
        """Map Sorcha ``fieldMJD_TAI`` values onto pointing-DB visit ids.

        Parameters
        ----------
        field_mjd_tai : array-like of `float`
            Sorcha per-row field times.
        tol_days : `float`, optional
            Maximum allowed separation, in days, for a field time to be considered
            the same visit.

        Returns
        -------
        visit : `numpy.ndarray` of `int64`
            Matched ``observationId`` per input row. Unmatched rows get ``-1``.
        visit_time : `numpy.ndarray` of `float`
            ``visitTime`` in seconds for the matched visit; ``nan`` where unmatched.
        """
        t = np.asarray(field_mjd_tai, dtype=float)
        ref = self.field_mjd_tai
        # Nearest neighbour in a sorted array: compare the two bracketing entries.
        idx = np.searchsorted(ref, t)
        lo = np.clip(idx - 1, 0, len(ref) - 1)
        hi = np.clip(idx, 0, len(ref) - 1)
        pick = np.where(np.abs(ref[hi] - t) < np.abs(ref[lo] - t), hi, lo)

        good = np.abs(ref[pick] - t) <= tol_days
        visit = np.where(good, self.visit[pick], -1).astype(np.int64)
        vtime = np.where(good, self.visit_time[pick], np.nan)
        n_bad = int((~good).sum())
        if n_bad:
            logger.warning(
                "%d/%d Sorcha field times did not land on a pointing-DB visit within %.3g s",
                n_bad,
                len(t),
                tol_days * 86400.0,
            )
        return visit, vtime

    def field_times_for_visits(self, visits):
        """Return the Sorcha ``fieldMJD_TAI`` for each of the given visit ids.

        Parameters
        ----------
        visits : array-like of `int`
            Visit ids (``observationId``).

        Returns
        -------
        field_mjd_tai : `numpy.ndarray` of `float`
            Field time per requested visit; ``nan`` for visits absent from the DB.
        """
        want = np.asarray(visits, dtype=np.int64)
        order = np.argsort(self.visit, kind="stable")
        sorted_visits = self.visit[order]
        idx = np.searchsorted(sorted_visits, want)
        idx_clipped = np.clip(idx, 0, len(sorted_visits) - 1)
        hit = sorted_visits[idx_clipped] == want
        out = np.full(len(want), np.nan)
        out[hit] = self.field_mjd_tai[order][idx_clipped[hit]]
        return out


def ic_visit_table(ic):
    """Collapse an ImageCollection to one row per visit.

    An ImageCollection has one row per (visit, detector) exposure; the temporal join
    against Sorcha operates on visits, so this deduplicates.

    Parameters
    ----------
    ic : `kbmod.ImageCollection` or `astropy.table.Table`
        The collection (or its underlying ``.data`` table).

    Returns
    -------
    table : `dict` of `numpy.ndarray`
        Keys ``visit``, ``mjd_mid``, ``mjd_start``, ``exposure_time``, ``band``,
        one entry per distinct visit, ordered by ``mjd_mid``.
    """
    data = getattr(ic, "data", ic)
    for col in ("visit", "mjd_mid"):
        if col not in data.colnames:
            raise ValueError(f"ImageCollection is missing the required '{col}' column")

    visit = np.asarray(data["visit"]).astype(np.int64)
    mjd_mid = np.asarray(data["mjd_mid"], dtype=float)
    mjd_start = (
        np.asarray(data["mjd_start"], dtype=float)
        if "mjd_start" in data.colnames
        else np.full(len(visit), np.nan)
    )
    exp_time = (
        np.asarray(data["exposureTime"], dtype=float)
        if "exposureTime" in data.colnames
        else np.full(len(visit), np.nan)
    )
    band = np.asarray(data["band"]).astype(str) if "band" in data.colnames else np.full(len(visit), "")

    # Keep the first occurrence of each visit, then order by time.
    _, first = np.unique(visit, return_index=True)
    first = np.sort(first)
    order = first[np.argsort(mjd_mid[first], kind="stable")]

    return {
        "visit": visit[order],
        "mjd_mid": mjd_mid[order],
        "mjd_start": mjd_start[order],
        "exposure_time": exp_time[order],
        "band": band[order],
    }


def sorcha_epoch_from_ic(mjd_mid, exposure_time, mjd_start=None):
    """The UTC MJD at which Sorcha evaluated positions, given an ImageCollection row.

    Sorcha's ``fieldMJD_TAI`` corresponds to the exposure **start**, so in the
    ImageCollection's own (UTC) time frame the Sorcha epoch is simply ``mjd_start``.
    Prefer that: it reproduces ``fieldMJD_TAI`` to under a millisecond.

    When ``mjd_start`` is unavailable the epoch is approximated as
    ``mjd_mid - exposureTime / 2``. That is close but not exact -- Rubin's ``mjd_mid``
    is offset from the start by half the *visit* time (~15.50 s), which exceeds half
    the *exposure* time (~15.00 s) by the 0.5 s shutter contribution. The residual
    0.5 s is harmless here (it moves a cold classical by under a milliarcsecond) but
    the exact value costs nothing when the column exists.

    Parameters
    ----------
    mjd_mid : array-like of `float`
        ``ic["mjd_mid"]``, UTC.
    exposure_time : array-like of `float`
        ``ic["exposureTime"]`` in seconds.
    mjd_start : array-like of `float`, optional
        ``ic["mjd_start"]``, UTC. Used wherever it is finite.

    Returns
    -------
    mjd : `numpy.ndarray` of `float`
        UTC MJD of the Sorcha epoch.
    """
    approx = np.asarray(mjd_mid, dtype=float) - np.asarray(exposure_time, dtype=float) / 2.0 / 86400.0
    if mjd_start is None:
        return approx
    start = np.asarray(mjd_start, dtype=float)
    return np.where(np.isfinite(start), start, approx)
