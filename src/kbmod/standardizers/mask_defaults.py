"""Mask defaults for each ingestion path; these policies intentionally differ."""

BUTLER_MASK_FLAGS = (
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

# The collection builder applies a broader mask than direct Butler ingestion.
# Mask negative detections and spikes while retaining injected sources.
IC_BUILD_MASK_FLAGS = (
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
