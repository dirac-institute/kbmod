"""
Class for standardizing FITS files produced by the Vera C. Rubin
Science Pipelines.
"""
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from scipy.signal import convolve2d

from kbmod.standardizers import MultiExtensionFits


__all__ = ["RubinSciPipeFits", ]


class RubinSciPipeFits(MultiExtensionFits):

    name = "RubinSciencePipelineFits"

    priority = 2

    bit_flag_map = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }

    mask_flags = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]

    @classmethod
    def canStandardize(cls, location):

        # A rough best guess here, I'm certain we can find a Rubin
        # signature somewhere in the header that's a clearer signal
        # that this is a Rubin Sci Pipe product
        #hdulist = fits.open(location)
        #primary = hdulist["PRIMARY"].header
        #isRubin = all(("ZTENSION" in primary,
        #               "ZPCOUNT" in primary,
        #               "ZGCOUNT" in primary,
        #               "CCDNUM" in primary))
        #return isRubin, hdulist
        parentCanStandardize, hdulist = super().canStandardize(location)

        if not parentCanStandardize:
            return False, []

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
        idx = self.hdulist.index_of("IMAGE")
        threshold_mask = self.hdulist[idx].data > 100

        net_flag = sum([2**self.bit_flag_map[f] for f in self.mask_flags])
        idx = self.hdulist.index_of("MASK")
        mask_data = self.hdulist[idx].data
        flag_mask = net_flag & mask_data

        net_mask = threshold_mask & flag_mask

        # this should grow the mask for 5 pixels each side
        grow_kernel = np.ones((11, 11))
        grown_mask = convolve2d(net_mask, grow_kernel, mode="same")

        return grown_mask

    def standardizeVarianceImage(self):
        idx = self.hdulist.index_of("VARIANCE")
        return self.hdulist[idx].data
