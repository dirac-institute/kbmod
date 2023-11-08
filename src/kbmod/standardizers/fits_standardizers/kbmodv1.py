"""Class for standardizing FITS files produced by the Vera C. Rubin
Science Pipelines as they were specified during the KBMOD V1.0 development.

There are no guarantees that current Rubin Science Pipeline Data Products can
still be standardized with this class. To ensure correct behavior for any
version of the Rubin Stack, use the `ButlerStandardizer`.
"""

from astropy.time import Time
from astropy.nddata import bitmask

import numpy as np
from scipy.signal import convolve2d

from .multi_extension_fits import MultiExtensionFits
from ..standardizer import StandardizerConfig


__all__ = ["KBMODV1", "KBMODV1Config", ]


class KBMODV1Config(StandardizerConfig):
    do_mask = True
    do_bitmask = True
    do_threshold = False

    brightness_treshold = 10
    grow_mask = True
    grow_kernel = (10, 10)

    bit_flag_map = {
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
    """Mapping between the flag meaning to its value."""

    mask_flags = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
    """List of flags that will be used when masking."""


class KBMODV1(MultiExtensionFits):
    name = "KBMODV1"
    priority = 2
    configClass = KBMODV1Config

    @classmethod
    def canStandardize(cls, location):
        parentCanStandardize, hdulist = super().canStandardize(location)

        if not parentCanStandardize:
            return False, []

        # A rough best guess here, I'm certain we can find a Rubin
        # signature somewhere in the header that's a clearer signal
        # that this is a Rubin Sci Pipe product
        primary = hdulist["PRIMARY"].header
        isRubin = all(("ZTENSION" in primary,
                       "ZPCOUNT" in primary,
                       "ZGCOUNT" in primary,
                       "CCDNUM" in primary))

        canStandardize = parentCanStandardize and isRubin
        return canStandardize, hdulist

    def __init__(self, location, config=None, **kwargs):
        super().__init__(location, config=config, **kwargs)
        # this is the only science-image header for Rubin
        self.processable = [self.hdulist[1], ]

    def translateHeader(self, *args, **kwargs):
        # this is the 1 mandatory piece of metadata we need to extract
        standardizedHeader = {}
        obs_datetime = Time(self.primary["DATE-AVG"], format="isot")
        standardizedHeader["mjd"] = obs_datetime.mjd

        # these are all optional things
        standardizedHeader["filter"] = self.primary["FILTER"]
        standardizedHeader["visit_id"] = self.primary["IDNUM"]
        standardizedHeader["observat"] = self.primary["OBSERVAT"]
        standardizedHeader["obs_lat"] = self.primary["OBS-LAT"]
        standardizedHeader["obs_lon"] = self.primary["OBS-LONG"]
        standardizedHeader["obs_elev"] = self.primary["OBS-ELEV"]

        return standardizedHeader

    def standardizeMaskImage(self):
        # Return empty masks if no masking is done
        if not self.config["do_mask"]:
            sizes = self._bestGuessImageDimensions()
            return (np.zeros(size) for size in sizes)

        # Otherwise load the mask extension and process it
        mask = self.hdulist["MASK"]

        if self.config["do_bitmask"]:
            mask = bitmask.bitfield_to_boolean_mask(
                bitfield=mask,
                ignore_flags=self.config["mask_flags"],
                flag_name_map=self.config["bit_flag_map"]
            )

        if self.config["do_threshold"]:
            bmask = self.processable[0] > self.config["brightness_threshold"]
            mask = mask & bmask

        if self.config["grow_mask"]:
            grow_kernel = np.ones(self.config["grow_kernel"])
            mask = convolve2d(mask, grow_kernel, mode="same")

        return [mask, ]

    def standardizeVarianceImage(self):
        return [self.hdulist["VARIANCE"].data, ]
