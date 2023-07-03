"""
Class for standardizing FITS files produced by the Vera C. Rubin
Science Pipelines.
"""
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.nddata import bitmask

from scipy.signal import convolve2d

from kbmod.standardizers import MultiExtensionFits


__all__ = ["RubinSciPipeFits", ]


class RubinSciPipeFits(MultiExtensionFits):

    name = "RubinSciencePipelineFits"

    priority = 2

    # it must be the power right?
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

    def __init__(self, location):
        super().__init__(location)
        # this is the only science-image header for Rubin
        self.exts = [self.hdulist[1], ]

    def translateHeader(self):
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

        idx = self.hdulist.index_of("MASK")
        bitfield = self.hdulist[idx].data
        bit_mask = bitmask.bitfield_to_boolean_mask(
            bitfield=bitfield,
            ignore_flags=self.mask_flags,
            flag_name_map=self.bit_flag_map
        )

        idx = self.hdulist.index_of("IMAGE")
        image = self.hdulist[idx].data
        brigthness_threshold = image.mean() - image.std()
        threshold_mask = image > brightness_threshold

        net_mask = threshold_mask & bit_mask

        # this should grow the mask for 5 pixels each side
        grow_kernel = np.ones((11, 11))
        grown_mask = convolve2d(net_mask, grow_kernel, mode="same")

        return grown_mask

    def standardizeVarianceImage(self):
        idx = self.hdulist.index_of("VARIANCE")
        return self.hdulist[idx].data
