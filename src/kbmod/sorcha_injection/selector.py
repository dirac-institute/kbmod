"""Select the Sorcha rows that belong in one ImageCollection.

The join is naturally per-visit: a Sorcha row already *is* "object X seen in pointing P
at time T", so matching it to a collection means

1. **Temporal** -- the row's visit id is one of the collection's visits. This is an
   exact integer join (see :mod:`kbmod.sorcha_injection.visits`); no time tolerance is
   involved.
2. **Spatial** -- the row's own per-visit ``(RA, Dec)`` falls on one of the detectors
   the collection holds for that visit, and (optionally) inside the patch footprint.

Using each row's own per-visit position rather than a single mean position is what
keeps fast movers landing in the right place.

Density control samples **whole tracks** by ``ObjID``. Thinning row-by-row would
punch holes in tracks and destroy exactly the property that makes this useful for
linking tests.
"""

import json
import logging
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u

from .visits import ic_visit_table, sorcha_epoch_from_ic

logger = logging.getLogger(__name__)


def _patch_healpix_cells(global_wcs, nside, pad_factor=1.5):
    """NEST healpix cells plausibly overlapping a global WCS footprint.

    Uses the circumscribing circle of the footprint, inflated a little, and asks for
    inclusive coverage. This is only a pre-filter -- exact containment is tested later
    -- so erring generous is correct.
    """
    import hpgeom

    try:
        corners = global_wcs.calc_footprint()
    except Exception:
        return None
    if corners is None or len(corners) == 0:
        return None

    center = SkyCoord(corners[:, 0] * u.deg, corners[:, 1] * u.deg).cartesian.mean()
    center = SkyCoord(center, representation_type="cartesian").represent_as("unitspherical")
    c_ra, c_dec = center.lon.deg, center.lat.deg
    radius = (
        SkyCoord(c_ra * u.deg, c_dec * u.deg)
        .separation(SkyCoord(corners[:, 0] * u.deg, corners[:, 1] * u.deg))
        .deg.max()
    )
    return hpgeom.query_circle(
        nside, c_ra, c_dec, radius * pad_factor, inclusive=True, nest=True, lonlat=True, degrees=True
    )


def _ic_observed_healpix_cells(ic, nside, pad_factor=1.5, margin_deg=0.2):
    """NEST healpix cells covering an ImageCollection's images in the OBSERVED frame.

    The Sorcha index is keyed on each detection's *raw on-sky* position, so the
    healpix pre-filter must be built in the observed frame. Building it from a
    reflex-corrected patch WCS (as :func:`_patch_healpix_cells` does) is wrong:
    that footprint is shifted from the observed sky by the parallax the reflex
    correction removes -- up to a degree or more -- so an object with a large
    reflex shift lands outside the cells and is silently dropped, even though its
    raw detections fall squarely on the patch's images. This uses the collection's
    own observed image centres (the ``ra``/``dec`` columns), padded generously; the
    exact in-patch containment is still tested later against ``global_wcs``.

    Returns ``None`` (no pre-filter) when the collection carries no observed
    positions, which is the correct generous fallback.
    """
    import hpgeom

    data = getattr(ic, "data", ic)
    colnames = getattr(data, "colnames", None) or []
    if "ra" not in colnames or "dec" not in colnames:
        return None
    ra = np.asarray(data["ra"], dtype=float)
    dec = np.asarray(data["dec"], dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    if not good.any():
        return None
    ra, dec = ra[good], dec[good]

    center = SkyCoord(ra * u.deg, dec * u.deg).cartesian.mean()
    center = SkyCoord(center, representation_type="cartesian").represent_as("unitspherical")
    c_ra, c_dec = center.lon.deg, center.lat.deg
    radius = (
        SkyCoord(c_ra * u.deg, c_dec * u.deg)
        .separation(SkyCoord(ra * u.deg, dec * u.deg))
        .deg.max()
    )
    # ``ra``/``dec`` are image centres; pad by roughly a detector half-size so the
    # corners of edge exposures are covered before the generous pad_factor.
    radius = (radius + margin_deg) * pad_factor
    return hpgeom.query_circle(
        nside, c_ra, c_dec, radius, inclusive=True, nest=True, lonlat=True, degrees=True
    )


def _row_wcs(ic_data, i):
    """Per-exposure `astropy.wcs.WCS` for row ``i`` of an ImageCollection table."""
    raw = ic_data["wcs"][i]
    try:
        raw = json.loads(raw)
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(raw, relax=True)


def _wcs_pixel_shape(wcs, ic_data, i):
    """Pixel dimensions ``(nx, ny)`` for an exposure's WCS."""
    if getattr(wcs, "pixel_shape", None) is not None:
        return wcs.pixel_shape
    if getattr(wcs, "array_shape", None) is not None:
        return wcs.array_shape[1], wcs.array_shape[0]
    raise ValueError(f"Cannot determine pixel dimensions for ImageCollection row {i}")


def _rates_from_track(mjd, ra, dec):
    """Per-row on-sky rate (deg/day in RA*cos(dec) and Dec) from a single object's track.

    Uses centred differences where possible, one-sided at the ends. Returns zeros for
    single-detection tracks, which is the right no-op behaviour.
    """
    n = len(mjd)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    order = np.argsort(mjd)
    t, a, d = mjd[order], ra[order], dec[order]
    cosd = np.cos(np.radians(d))
    # Unwrap RA so a 0/360 crossing does not produce an absurd rate.
    da = np.diff(np.unwrap(np.radians(a))) * np.degrees(1.0) * cosd[:-1]
    dd = np.diff(d)
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    v_a_seg, v_d_seg = da / dt, dd / dt

    v_a, v_d = np.empty(n), np.empty(n)
    v_a[0], v_d[0] = v_a_seg[0], v_d_seg[0]
    v_a[-1], v_d[-1] = v_a_seg[-1], v_d_seg[-1]
    if n > 2:
        v_a[1:-1] = 0.5 * (v_a_seg[:-1] + v_a_seg[1:])
        v_d[1:-1] = 0.5 * (v_d_seg[:-1] + v_d_seg[1:])

    out_a, out_d = np.empty(n), np.empty(n)
    out_a[order], out_d[order] = v_a, v_d
    return np.nan_to_num(out_a), np.nan_to_num(out_d)


def _apply_epoch_correction(obj_ids, mjd_sorcha, ra, dec, dt_days):
    """Propagate positions forward by ``dt_days`` using each object's own rate.

    Sorcha evaluated positions at the exposure start; KBMOD's ``obstime`` is
    mid-exposure. The gap is half an exposure (~15.5 s), over which motion is linear
    to far better than the resulting sub-milliarcsecond residual.
    """
    ra_out, dec_out = ra.copy(), dec.copy()
    order = np.argsort(obj_ids, kind="stable")
    sorted_ids = obj_ids[order]
    bounds = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi]
        v_a, v_d = _rates_from_track(mjd_sorcha[idx], ra[idx], dec[idx])
        cosd = np.cos(np.radians(dec[idx]))
        cosd[cosd == 0] = 1.0
        ra_out[idx] = ra[idx] + v_a * dt_days[idx] / cosd
        dec_out[idx] = dec[idx] + v_d * dt_days[idx]
    return ra_out, dec_out


def select_injections_for_ic(ic, index, config, global_wcs=None, guess_distance=None, earth_loc=None):
    """Choose the Sorcha detections to inject into one ImageCollection.

    Parameters
    ----------
    ic : `kbmod.ImageCollection`
        The (patch) collection to inject into.
    index : `kbmod.sorcha_injection.SorchaIndex`
        Prebuilt ephemeris index.
    config : `kbmod.sorcha_injection.SorchaInjectionConfig`
        Selection configuration.
    global_wcs : `astropy.wcs.WCS`, optional
        Patch WCS. Defaults to ``ic.get_global_wcs()``. Required when
        ``config.require_in_patch`` is set.
    guess_distance : `float`, optional
        Heliocentric guess distance (AU) the patch is reflex-corrected at. Defaults to
        the collection's ``helio_guess_dist`` column when present. ``None`` or ``0.0``
        means the patch is in the observed frame.
    earth_loc : `astropy.coordinates.EarthLocation`, optional
        Observatory location, needed only when reflex-correcting. Defaults to
        ``ic.get_observatory()``.

    Returns
    -------
    sel : `dict` of `numpy.ndarray`
        One entry per (object, exposure) pair to inject:

        ``ObjID``, ``population``, ``visit``, ``detector``, ``ic_row``,
        ``obstime`` (the collection's own ``mjd_mid``, bit-identical so that
        ``inject_sources_into_ic``'s equality test matches), ``ra``/``dec``
        (observed frame, epoch-corrected to mid-exposure), ``ra_reflex``/``dec_reflex``
        (patch frame, present only when reflex-corrected), ``mag``, ``band``,
        ``det_x``/``det_y`` (pixel position on its own detector), and
        ``patch_x``/``patch_y`` (pixel position in the patch frame).
    """
    data = getattr(ic, "data", ic)
    if global_wcs is None and hasattr(ic, "get_global_wcs"):
        global_wcs = ic.get_global_wcs()
    if guess_distance is None and "helio_guess_dist" in data.colnames:
        guess_distance = float(data["helio_guess_dist"][0])
    if guess_distance is not None and guess_distance == 0.0:
        guess_distance = None

    visits = ic_visit_table(ic)
    if len(visits["visit"]) == 0:
        raise ValueError("ImageCollection has no visits")

    # ---------- 1. temporal: exact visit-id join, restricted to the right nights ----------
    field_times = _field_times_for_ic(visits)
    nights = np.unique(np.floor(field_times[np.isfinite(field_times)])).astype(np.int32)

    # Sky pre-filter for the index pull. The index is keyed on each detection's raw
    # observed position, so the healpix cells must come from the collection's observed
    # image footprints -- NOT the reflex-corrected patch WCS, whose footprint is shifted
    # by the parallax and would wrongly exclude large-reflex-shift objects. Exact in-patch
    # containment is still enforced later via ``global_wcs``.
    healpix = _ic_observed_healpix_cells(ic, index.nside)

    table = index.read(
        visits=visits["visit"],
        nights=nights if len(nights) else None,
        populations=config.populations,
        healpix=healpix,
        mag_range=config.mag_range,
        bands=config.bands,
    )
    logger.info(
        "Index pull for %d visits over %d nights: %d rows", len(visits["visit"]), len(nights), table.num_rows
    )
    if table.num_rows == 0:
        return _empty_selection(guess_distance)

    obj_id = np.asarray(table["ObjID"].to_pandas(), dtype=object).astype(str)
    population = np.asarray(table["population"].to_pandas(), dtype=object).astype(str)
    s_visit = table["visit"].to_numpy(zero_copy_only=False)
    s_field_mjd = table["fieldMJD_TAI"].to_numpy(zero_copy_only=False)
    s_ra = table["RA_deg"].to_numpy(zero_copy_only=False)
    s_dec = table["Dec_deg"].to_numpy(zero_copy_only=False)
    s_mag = table["trailedSourceMag"].to_numpy(zero_copy_only=False)
    s_band = np.asarray(table["optFilter"].to_pandas(), dtype=object).astype(str)

    # ---------- 2. epoch: shift start-of-exposure positions to mid-exposure ----------
    v_order = np.argsort(visits["visit"], kind="stable")
    v_sorted = visits["visit"][v_order]
    pos = np.searchsorted(v_sorted, s_visit)
    pos = np.clip(pos, 0, len(v_sorted) - 1)
    row_of_visit = v_order[pos]
    obstime = visits["mjd_mid"][row_of_visit]
    exp_time = visits["exposure_time"][row_of_visit]
    start_time = visits["mjd_start"][row_of_visit]

    if config.correct_epoch_to_mid_exposure:
        sorcha_epoch_utc = sorcha_epoch_from_ic(obstime, exp_time, mjd_start=start_time)
        dt_days = np.nan_to_num(obstime - sorcha_epoch_utc)
        s_ra, s_dec = _apply_epoch_correction(obj_id, s_field_mjd, s_ra, s_dec, dt_days)

    # ---------- 3. spatial: land each row on a real detector, and in the patch ----------
    keep_rows, ic_rows, det_x, det_y = _match_to_detectors(
        data, s_visit, s_ra, s_dec, require=config.require_on_detector
    )
    if len(keep_rows) == 0:
        logger.info("No Sorcha rows landed on a detector in this collection")
        return _empty_selection(guess_distance)

    idx = keep_rows
    obj_id, population = obj_id[idx], population[idx]
    s_visit, s_ra, s_dec, s_mag, s_band = s_visit[idx], s_ra[idx], s_dec[idx], s_mag[idx], s_band[idx]
    obstime = obstime[idx]

    ra_reflex = dec_reflex = None
    if guess_distance is not None:
        ra_reflex, dec_reflex = _reflex_correct(ic, s_ra, s_dec, obstime, guess_distance, earth_loc)

    patch_x = patch_y = None
    if global_wcs is not None:
        frame_ra = s_ra if ra_reflex is None else ra_reflex
        frame_dec = s_dec if dec_reflex is None else dec_reflex
        patch_x, patch_y = global_wcs.world_to_pixel(SkyCoord(frame_ra * u.deg, frame_dec * u.deg))

        if config.require_in_patch:
            nx, ny = _global_pixel_shape(global_wcs, data)
            inside = (patch_x >= 0) & (patch_x < nx) & (patch_y >= 0) & (patch_y < ny)
            n_drop = int((~inside).sum())
            if n_drop:
                logger.info("Dropping %d/%d rows outside the patch footprint", n_drop, len(inside))
            (
                obj_id,
                population,
                s_visit,
                s_ra,
                s_dec,
                s_mag,
                s_band,
                obstime,
                ic_rows,
                det_x,
                det_y,
                patch_x,
                patch_y,
            ) = (
                arr[inside]
                for arr in (
                    obj_id,
                    population,
                    s_visit,
                    s_ra,
                    s_dec,
                    s_mag,
                    s_band,
                    obstime,
                    ic_rows,
                    det_x,
                    det_y,
                    patch_x,
                    patch_y,
                )
            )
            if ra_reflex is not None:
                ra_reflex, dec_reflex = ra_reflex[inside], dec_reflex[inside]

    sel = {
        "ObjID": obj_id,
        "population": population,
        "visit": s_visit,
        "detector": (
            np.asarray(data["detector"])[ic_rows]
            if "detector" in data.colnames
            else np.full(len(ic_rows), -1)
        ),
        "ic_row": ic_rows,
        "obstime": obstime,
        "ra": s_ra,
        "dec": s_dec,
        "mag": s_mag,
        "band": s_band,
        "det_x": det_x,
        "det_y": det_y,
    }
    if ra_reflex is not None:
        sel["ra_reflex"], sel["dec_reflex"] = ra_reflex, dec_reflex
    if patch_x is not None:
        sel["patch_x"], sel["patch_y"] = patch_x, patch_y

    sel = _apply_track_filters(sel, config)
    n_obj = len(np.unique(sel["ObjID"])) if len(sel["ObjID"]) else 0
    logger.info("Selected %d detections of %d Sorcha objects for injection", len(sel["ObjID"]), n_obj)
    return sel


def _field_times_for_ic(visits):
    """Sorcha field times (TAI) implied by a collection's visits, for night pruning."""
    from astropy.time import Time

    start = visits["mjd_start"]
    bad = ~np.isfinite(start)
    if bad.any():
        # Fall back to mid-exposure minus half an exposure when mjd_start is absent.
        start = np.where(bad, sorcha_epoch_from_ic(visits["mjd_mid"], visits["exposure_time"]), start)
    finite = np.isfinite(start)
    out = np.full(len(start), np.nan)
    if finite.any():
        out[finite] = Time(start[finite], format="mjd", scale="utc").tai.mjd
    return out


def _match_to_detectors(data, s_visit, s_ra, s_dec, require=True):
    """Assign each Sorcha row to the exposure whose pixel grid contains it.

    Returns
    -------
    keep : `numpy.ndarray` of `int`
        Indices into the Sorcha arrays that landed on some exposure.
    ic_rows : `numpy.ndarray` of `int`
        Matching ImageCollection row index for each kept Sorcha row.
    x, y : `numpy.ndarray` of `float`
        Pixel coordinates on that exposure.
    """
    ic_visit = np.asarray(data["visit"]).astype(np.int64)

    if not require:
        # Attach each row to the first IC row sharing its visit, without a bounds test.
        order = np.argsort(ic_visit, kind="stable")
        pos = np.clip(np.searchsorted(ic_visit[order], s_visit), 0, len(order) - 1)
        rows = order[pos]
        hit = ic_visit[rows] == s_visit
        keep = np.flatnonzero(hit)
        return keep, rows[keep], np.full(len(keep), np.nan), np.full(len(keep), np.nan)

    # Group Sorcha rows by visit once, then test only the exposures of that visit.
    keep, ic_rows, xs, ys = [], [], [], []
    by_visit = {}
    for i, v in enumerate(ic_visit):
        by_visit.setdefault(int(v), []).append(i)

    s_order = np.argsort(s_visit, kind="stable")
    s_sorted = s_visit[s_order]
    bounds = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        v = int(s_sorted[lo])
        rows = by_visit.get(v)
        if not rows:
            continue
        sidx = s_order[lo:hi]
        coords = SkyCoord(s_ra[sidx] * u.deg, s_dec[sidx] * u.deg)
        unassigned = np.ones(len(sidx), dtype=bool)
        for r in rows:
            if not unassigned.any():
                break
            wcs = _row_wcs(data, r)
            nx, ny = _wcs_pixel_shape(wcs, data, r)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y = wcs.world_to_pixel(coords)
            good = unassigned & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
            if not good.any():
                continue
            g = np.flatnonzero(good)
            keep.extend(sidx[g])
            ic_rows.extend([r] * len(g))
            xs.extend(x[g])
            ys.extend(y[g])
            unassigned[g] = False

    if not keep:
        return np.array([], int), np.array([], int), np.array([]), np.array([])
    keep = np.asarray(keep, int)
    order = np.argsort(keep, kind="stable")
    return keep[order], np.asarray(ic_rows, int)[order], np.asarray(xs)[order], np.asarray(ys)[order]


def _reflex_correct(ic, ra, dec, obstime, guess_distance, earth_loc=None):
    """Forward reflex (parallax) correction of observed coordinates into the patch frame."""
    from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized

    if earth_loc is None:
        earth_loc = ic.get_observatory() if hasattr(ic, "get_observatory") else None
    if earth_loc is None:
        raise ValueError("An EarthLocation is required to reflex-correct; none found on the ImageCollection")
    corrected, _ = correct_parallax_geometrically_vectorized(ra, dec, obstime, guess_distance, earth_loc)
    return corrected.ra.deg, corrected.dec.deg


def _global_pixel_shape(global_wcs, data):
    if getattr(global_wcs, "pixel_shape", None) is not None:
        return global_wcs.pixel_shape
    if "global_wcs_pixel_shape_0" in getattr(data, "colnames", []):
        return int(data["global_wcs_pixel_shape_0"][0]), int(data["global_wcs_pixel_shape_1"][0])
    if getattr(global_wcs, "array_shape", None) is not None:
        return global_wcs.array_shape[1], global_wcs.array_shape[0]
    raise ValueError("Cannot determine the patch pixel dimensions; set global_wcs.pixel_shape")


def _apply_track_filters(sel, config):
    """Drop short tracks, then optionally subsample whole objects."""
    obj = sel["ObjID"]
    if len(obj) == 0:
        return sel

    if config.min_obs_per_object > 1:
        uniq, counts = np.unique(obj, return_counts=True)
        keep_ids = uniq[counts >= config.min_obs_per_object]
        mask = np.isin(obj, keep_ids)
        n_drop = len(uniq) - len(keep_ids)
        if n_drop:
            logger.info("Dropping %d objects with < %d detections", n_drop, config.min_obs_per_object)
        sel = {k: v[mask] for k, v in sel.items()}
        obj = sel["ObjID"]

    if config.max_objs_per_patch is not None and len(obj):
        uniq = np.unique(obj)
        if len(uniq) > config.max_objs_per_patch:
            rng = np.random.default_rng(config.seed)
            keep_ids = rng.choice(uniq, size=config.max_objs_per_patch, replace=False)
            mask = np.isin(obj, keep_ids)
            logger.info("Capping %d objects down to %d", len(uniq), config.max_objs_per_patch)
            sel = {k: v[mask] for k, v in sel.items()}
    return sel


def _empty_selection(guess_distance):
    empty_f = np.array([], dtype=float)
    empty_i = np.array([], dtype=np.int64)
    empty_s = np.array([], dtype=str)
    sel = {
        "ObjID": empty_s,
        "population": empty_s,
        "visit": empty_i,
        "detector": empty_i,
        "ic_row": empty_i,
        "obstime": empty_f,
        "ra": empty_f,
        "dec": empty_f,
        "mag": empty_f,
        "band": empty_s,
        "det_x": empty_f,
        "det_y": empty_f,
        "patch_x": empty_f,
        "patch_y": empty_f,
    }
    if guess_distance is not None:
        sel["ra_reflex"], sel["dec_reflex"] = empty_f, empty_f
    return sel
