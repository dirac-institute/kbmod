"""Named mask policies for Butler ingestion and legacy FITS formats."""

from enum import Enum


class MaskProfile(str, Enum):
    """Supported Butler mask profiles, also accepted as lowercase strings."""

    LSSTCAM = "lsstcam"
    COMCAM = "comcam"
    DEEP = "deep"


DEEP_MASK_FLAGS = (
    "BAD",
    "CLIPPED",
    "CR",
    "CROSSTALK",
    "EDGE",
    "NO_DATA",
    "SAT",
    "SENSOR_EDGE",
    "SUSPECT",
)

# Mask negative detections and spikes while retaining injected sources.
RUBIN_MASK_FLAGS = (
    "BAD",
    "CLIPPED",
    "CR",
    "CROSSTALK",
    "DETECTED_NEGATIVE",
    "EDGE",
    "INEXACT_PSF",
    "INTRP",
    "ITL_DIP",
    "NOT_DEBLENDED",
    "NO_DATA",
    "REJECTED",
    "SAT",
    "SAT_TEMPLATE",
    "SENSOR_EDGE",
    "SPIKE",
    "STREAK",
    "SUSPECT",
    "UNMASKEDNAN",
    "VIGNETTED",
)


def resolve_mask_flags(profile, mask_flags=None):
    """Return a fresh flag list; explicit flags (including []) override the profile."""
    profile = MaskProfile(profile)
    if mask_flags is not None:
        return list(mask_flags)
    # LSSTCam and ComCam currently share the Rubin policy.
    return list(DEEP_MASK_FLAGS if profile == MaskProfile.DEEP else RUBIN_MASK_FLAGS)


LEGACY_FITS_MASK_FLAGS = ("BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN")

# Fixed bits describe the legacy FITS format. Butler reads each exposure's map.
LEGACY_FITS_MASK_BITS = {
    "BAD": 2**0,
    "CLIPPED": 2**9,
    "CR": 2**3,
    "CROSSTALK": 2**10,
    "DETECTED": 2**5,
    "DETECTED_NEGATIVE": 2**6,
    "EDGE": 2**4,
    "INEXACT_PSF": 2**11,
    "INTRP": 2**2,
    "NOT_DEBLENDED": 2**12,
    "NO_DATA": 2**8,
    "REJECTED": 2**13,
    "SAT": 2**1,
    "SENSOR_EDGE": 2**14,
    "SUSPECT": 2**7,
    "UNMASKEDNAN": 2**15,
}
