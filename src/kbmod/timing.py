"""Versioned timing provenance. Missing records are unknown, never inferred.

Per-image columns survive ImageCollection packing, slicing, stacking, and
WorkUnit FITS serialization. The recorded epoch and dataset ID bind the claim
to the values that were actually extracted, rather than to a file's age.
"""

import hashlib

import numpy as np

BUTLER_MIDPOINT_CONVENTION = "butler.visitinfo.utc_midpoint.v1"
MIDPOINT_TOLERANCE_SECONDS = 0.001
TIMING_FIELDS = (
    "timing_convention",
    "timing_scale",
    "timing_format",
    "timing_standardizer_version",
    "timing_kbmod_version",
    "timing_mjd_mid",
    "timing_data_id",
)


def butler_timing_metadata(mjd_mid, data_id):
    """Record provenance only when extracting a native Butler midpoint."""
    try:
        from kbmod._version import version
    except ImportError:
        version = "unknown"
    return dict(
        timing_convention=BUTLER_MIDPOINT_CONVENTION,
        timing_scale="utc",
        timing_format="mjd",
        timing_standardizer_version="ButlerStandardizer/1",
        timing_kbmod_version=version,
        timing_mjd_mid=float(mjd_mid),
        timing_data_id=str(data_id),
    )


def epoch_digest(times):
    """Hash ordered float64 epochs, including duplicates, independent of host byte order."""
    return hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest()


def timing_summary(data):
    """Summarize recorded claims without fetching sources or upgrading data.

    ``current`` means the recognized record agrees with its stored epoch and
    dataset ID, not independent source verification. Supports packed columns
    and masked rows from stacking old and new collections.
    """
    unknown = {"schema_version": 1, "status": "unknown"}
    if data is None or len(data) == 0:
        return unknown

    def values(key):
        if key in data.colnames:
            return np.ma.asarray(data[key])
        if key in data.meta.get("shared_cols", []):
            return np.ma.asarray([data.meta[key]] * len(data))
        raise KeyError(key)

    try:
        fields = {key: values(key) for key in TIMING_FIELDS}
        times = values("mjd_mid").astype(float).filled(np.nan)
        recorded = fields["timing_mjd_mid"].astype(float).filled(np.nan)
        recognized = np.ones(len(data), dtype=bool)
        for key, expected in (
            ("timing_convention", BUTLER_MIDPOINT_CONVENTION),
            ("timing_scale", "utc"),
            ("timing_format", "mjd"),
            ("timing_standardizer_version", "ButlerStandardizer/1"),
        ):
            recognized &= (fields[key] == expected).filled(False)
        recognized &= ~np.ma.getmaskarray(fields["timing_kbmod_version"])
        identity = (fields["timing_data_id"] == values("dataId")).filled(False)
        valid = (
            recognized
            & identity
            & np.isfinite(times)
            & np.isfinite(recorded)
            & np.isclose(times, recorded, rtol=0, atol=MIDPOINT_TOLERANCE_SECONDS / 86400)
        )
    except (KeyError, TypeError, ValueError):
        return unknown
    status = "current" if np.all(valid) else "mixed" if np.any(valid) else "unknown"
    if np.any(recognized & ~valid):
        status = "inconsistent"
    return {
        "schema_version": 1,
        "status": status,
        "rows": len(data),
        "current_rows": int(np.count_nonzero(valid)),
        "scale": "utc" if np.all(valid) else "unknown",
        "format": "mjd" if np.all(valid) else "unknown",
        "reference": "midpoint" if np.all(valid) else "unknown",
        "conventions": sorted(set(str(v) for v in fields["timing_convention"].compressed())),
        "standardizer_versions": sorted(
            set(str(v) for v in fields["timing_standardizer_version"].compressed())
        ),
        "kbmod_versions": sorted(set(str(v) for v in fields["timing_kbmod_version"].compressed())),
    }
