"""Class for standardizing FITS files produced by the Vera C. Rubin
Science Pipelines as they were specified during the KBMOD V1.0 development.

There are no guarantees that current Rubin Science Pipeline Data Products can
still be standardized with this class. To ensure correct behavior for any
version of the Rubin Stack, use the `ButlerStandardizer`.
"""

from .configurable_standardizer import ConfigurableStandardizer


class KBMODV1(ConfigurableStandardizer):

    name = "KBMODV1"

    priority = 2

    bit_flag_map = {
        "BAD": 2**0,
        "SAT": 2**1,
        "INTRP": 2**2,
        "CR": 2**3,
        "EDGE": 2**4,
        "DETECTED": 2**5,
        "DETECTED_NEGATIVE": 2**6,
        "SUSPECT": 2**7,
        "NO_DATA": 2**8,
        "CLIPPED": 2**9,
        "CROSSTALK": 2**10,
        "INEXACT_PSF": 2**11,
        "NOT_DEBLENDED": 2**12,
        "REJECTED": 2**13,
        "SENSOR_EDGE": 2**14,
        "UNMASKEDNAN": 2**15,
    }
    """Flag to bit value map."""

    mask_flags = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
    """List of flags that will be masked."""

    primary_keys = ["FILTER", "IDNUM", "OBSERVAT", "OBS-LAT",
                    "OBS-LONG", "OBS-ELEV"]
    """List of keys to extract from the primary header."""

    default_config = {
        "metadata": {
            "timestamp": "DATE-AVG",
            "time_fmt": "isot",
            "primary_keys": primary_keys,
            "additional_keys": []
        },
        "science": {
            "ext": 1  # probably also named SCIENCE or SCI, not sure though
        },
        "variance": {
            "has_variance": True,
            "ext": "VARIANCE"
        },
        "mask": {
            "do_mask": True,
            "has_mask": True,
            "ext": "MASK",
            "grow_mask": True,
            "grow_kernel": (10, 10),
            "brightness_threshold": None,
            "bit_flag_map": bit_flag_map,
            "mask_flags": mask_flags,
        }
    }
    """The default metadata, mask, variance, PSF, and science image processing
    configuration."""

    # We can standardize only Vera C. Rubin Science Pipeline Data Products,
    # i.e. it's got to be a Rubin-like FITS file. FYI, parent canStd always
    # returns False so we skip that check. This is a rough best guess that
    # we are dealing with Rubin data product, might need strengthening in the
    # future.
    @classmethod
    def canStandardize(cls, location, **kwargs):
        _, hdulist = super().canStandardize(location)
        primary = hdulist["PRIMARY"].header
        canStandardize = all(("ZTENSION" in primary,
                              "ZPCOUNT" in primary,
                              "ZGCOUNT" in primary,
                              "CCDNUM" in primary))

        return canStandardize, hdulist

    def __init__(self, location, conf=None, **kwargs):
        if conf is None:
            conf = self.default_config
        super().__init__(location, conf)
