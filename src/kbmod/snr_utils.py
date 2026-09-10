"""Predicted signal-to-noise for a point source in an `ImageCollection`.

This is a first-principles estimate: given a source's magnitude and the
per-visit depth already recorded in an `ImageCollection`, it predicts the
stacked SNR a shift-and-stack search would accumulate. No detection, search or
image data is required.

Do not confuse it with the two *measured* quantities KBMOD reports:

* the search likelihood, ``sum(psi) / sqrt(sum(phi))`` over a trajectory's
  valid epochs, and
* the on-object SNR, the same restricted to epochs matched to a known object.

Those describe what a search found. This module describes what is there to be
found, and the two agreeing is a meaningful check rather than a tautology.

The model is sky-limited PSF photometry, per visit::

    flux_i  = 10 ** (-0.4 * (mag_i - zeroPoint_i))
    noise_i = skyNoise_i * sqrt(psfArea_i)
    snr_i   = flux_i / noise_i
    stacked = sqrt(sum(snr_i ** 2))

``zeroPoint``, ``skyNoise`` and ``psfArea`` are per-visit and band-specific -
sky noise varies by roughly 2x between g/r/i/z - so the magnitude must be the
source's magnitude *in that visit's band*. Collapsing a source's colours to a
single mean magnitude biases the prediction; see `predicted_snr`.
"""

import logging

import numpy as np
from astropy.table import Table

__all__ = [
    "epoch_snr",
    "stack_snr",
    "predicted_snr",
    "SNR_COLUMNS",
]

logger = logging.getLogger(__name__)

SNR_COLUMNS = ("zeroPoint", "skyNoise", "psfArea")
"""ImageCollection columns the prediction is computed from."""


def epoch_snr(mag, zero_point, sky_noise, psf_area):
    """Signal-to-noise of a point source in a single exposure.

    Sky-limited PSF photometry: the source's flux in the exposure's own
    photometric system, divided by the sky noise integrated over the effective
    PSF footprint.

    Parameters
    ----------
    mag : `float` or `np.ndarray`
        Source magnitude, in the same band as ``zero_point``.
    zero_point : `float` or `np.ndarray`
        The exposure's photometric zero point.
    sky_noise : `float` or `np.ndarray`
        Standard deviation of the sky background, in the exposure's units.
    psf_area : `float` or `np.ndarray`
        Effective PSF area in pixels.

    Returns
    -------
    snr : `np.ndarray`
        Per-exposure signal-to-noise.
    """
    mag = np.asarray(mag, dtype=float)
    zero_point = np.asarray(zero_point, dtype=float)
    sky_noise = np.asarray(sky_noise, dtype=float)
    psf_area = np.asarray(psf_area, dtype=float)

    flux = 10 ** (-0.4 * (mag - zero_point))
    noise = sky_noise * np.sqrt(psf_area)
    return flux / noise


def stack_snr(snr):
    """Combine per-epoch SNR into the stacked SNR.

    Epochs add in quadrature, which is what makes an N-epoch stack grow as
    ``sqrt(N)`` for a steady source. It is also why a single epoch's
    "contribution" is its *squared* SNR, not its SNR.

    Parameters
    ----------
    snr : array-like
        Per-epoch signal-to-noise values.

    Returns
    -------
    stacked : `float`
        The stacked signal-to-noise.
    """
    snr = np.asarray(snr, dtype=float)
    if snr.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(np.square(snr))))


def _resolve_table(ic):
    """Return the metadata table of ``ic``, which may be an ImageCollection."""
    data = getattr(ic, "data", ic)
    if not isinstance(data, Table):
        raise TypeError(f"Expected an ImageCollection or astropy Table, got {type(ic)}.")

    missing = [c for c in SNR_COLUMNS if c not in data.colnames]
    if missing:
        raise ValueError(
            f"predicted_snr requires the columns {list(SNR_COLUMNS)}. Missing: {missing}. "
            "These come from the exposure summary statistics, so the collection must be "
            "Butler-backed."
        )
    return data


def _resolve_mag(mag, data):
    """Broadcast ``mag`` to one magnitude per row of ``data``.

    A single magnitude is only unambiguous when every row is the same band. A
    source has a different magnitude in each band, and per-visit depth is
    band-specific, so silently reusing one magnitude across bands would bias
    the prediction.
    """
    n_rows = len(data)

    if isinstance(mag, dict):
        if "band" not in data.colnames:
            raise ValueError("A per-band magnitude mapping needs a 'band' column in the collection.")
        bands = np.asarray(data["band"])
        unknown = sorted(set(bands.tolist()) - set(mag))
        if unknown:
            raise ValueError(f"No magnitude given for band(s) {unknown}. Known: {sorted(mag)}.")
        return np.array([float(mag[b]) for b in bands])

    mag = np.asarray(mag, dtype=float)
    if mag.ndim == 0:
        if "band" in data.colnames:
            bands = set(np.asarray(data["band"]).tolist())
            if len(bands) > 1:
                raise ValueError(
                    f"The collection spans bands {sorted(bands)} but a single magnitude was given. "
                    "Depth is band-specific, so pass a {band: mag} mapping or a per-row array. "
                    "Averaging a source's magnitudes across bands overstates its SNR."
                )
        return np.full(n_rows, float(mag))

    if mag.shape != (n_rows,):
        raise ValueError(f"Magnitude array must have shape ({n_rows},), got {mag.shape}.")
    return mag


def _select_rows(data, times, visits, time_threshold):
    """Boolean mask of the rows named by ``times`` and/or ``visits``."""
    mask = np.ones(len(data), dtype=bool)

    if visits is not None:
        if "visit" not in data.colnames:
            raise ValueError("Cannot select by visit without a 'visit' column.")
        mask &= np.isin(np.asarray(data["visit"]), np.atleast_1d(visits))

    if times is not None:
        if "mjd_mid" not in data.colnames:
            raise ValueError("Cannot select by time without a 'mjd_mid' column.")
        obs = np.asarray(data["mjd_mid"], dtype=float)
        # Match every row within the threshold, not just the nearest one: a
        # visit can appear as several rows and they all share an obstime.
        queries = np.atleast_1d(np.asarray(times, dtype=float))
        time_mask = np.any(np.abs(obs[:, None] - queries[None, :]) <= time_threshold, axis=1)
        mask &= time_mask

    return mask


def predicted_snr(
    ic,
    mag,
    times=None,
    visits=None,
    cumulative=False,
    one_row_per_visit=True,
    time_threshold=0.0007,
):
    """Predict the stacked SNR of a point source across an `ImageCollection`.

    Parameters
    ----------
    ic : `ImageCollection` or `astropy.table.Table`
        Butler-backed collection carrying `SNR_COLUMNS`.
    mag : `float`, `dict`, or array-like
        The source's magnitude. A ``{band: mag}`` mapping is the safe form for
        a multi-band collection; a per-row array is also accepted. A single
        float is rejected when the collection spans more than one band.
    times : array-like, optional
        Restrict to rows whose ``mjd_mid`` is within ``time_threshold`` of one
        of these.
    visits : array-like, optional
        Restrict to these visit ids.
    cumulative : `bool`
        Return only the stacked SNR as a float. The per-epoch table is
        returned otherwise, with the same value in ``meta['cumulative_snr']``.
    one_row_per_visit : `bool`
        Keep the first row of each visit. A visit covers many detectors but a
        source falls on one of them, so counting every row would multiply-count
        the epoch.
    time_threshold : `float`
        Match tolerance for ``times``, in days. Default 1 minute.

    Returns
    -------
    result : `astropy.table.Table` or `float`
        Per-epoch table with an ``snr`` column and a running ``cumulative_snr``
        in time order, or the stacked SNR when ``cumulative`` is set.
    """
    data = _resolve_table(ic)
    mags = _resolve_mag(mag, data)

    mask = _select_rows(data, times, visits, time_threshold)
    rows, mags = data[mask], mags[mask]

    if len(rows) == 0:
        logger.warning("No rows matched the requested times/visits; predicted SNR is 0.")
        return 0.0 if cumulative else Table(names=("snr",), dtype=(float,))

    order = np.argsort(np.asarray(rows["mjd_mid"], dtype=float)) if "mjd_mid" in rows.colnames else None
    if order is not None:
        rows, mags = rows[order], mags[order]

    if one_row_per_visit and "visit" in rows.colnames:
        _, first = np.unique(np.asarray(rows["visit"]), return_index=True)
        keep = np.sort(first)
        rows, mags = rows[keep], mags[keep]

    snr = epoch_snr(mags, rows["zeroPoint"], rows["skyNoise"], rows["psfArea"])
    stacked = stack_snr(snr)

    if cumulative:
        return stacked

    out = Table()
    for col in ("mjd_mid", "visit", "detector", "band", "filter"):
        if col in rows.colnames:
            out[col] = rows[col]
    out["mag"] = mags
    for col in SNR_COLUMNS:
        out[col] = rows[col]
    out["snr"] = snr
    # Running stack in time order - the build-up a search accumulates epoch by
    # epoch, and the quantity to plot against a measured likelihood curve.
    out["cumulative_snr"] = np.sqrt(np.cumsum(np.square(snr)))
    out.meta["cumulative_snr"] = stacked
    out.meta["n_epochs"] = len(out)
    return out
