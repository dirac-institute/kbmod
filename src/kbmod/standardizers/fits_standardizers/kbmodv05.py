"""Class for standardizing FITS files produced by deccam as they were
specified during the KBMOD V0.5 development.
"""

from astropy.time import Time
from astropy.nddata import bitmask

import logging
import numpy as np

from .multi_extension_fits import MultiExtensionFits, FitsStandardizerConfig

__all__ = [
    "KBMODV0_5",
    "KBMODV0_5Config",
]
logger = logging.getLogger(__name__)


class KBMODV0_5Config(FitsStandardizerConfig):
    do_mask = True
    """Perform masking if ``True``, otherwise return an empty mask."""

    do_bitmask = True
    """Mask ``mask_flags`` from the mask plane in the FITS file."""

    do_threshold = False
    """Mask all pixels above the given count threshold."""

    grow_mask = True
    """Grow mask footprint by ``grow_kernel_shape``"""

    brightness_treshold = 10
    """Pixels with value greater than this threshold will be masked."""

    grow_kernel_shape = (10, 10)
    """Size of the symmetric square kernel by which mask footprints will be
    increased by."""

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


class KBMODV0_5(MultiExtensionFits):
    """Standardizer for the legacy FITS files used in the v0.5 runs.

    This standardizer exists for backward compatibility purposes. Its use is
    not recommended. Use `ButlerStandardizer` instead.

    Parameters
    ----------
    location : `str` or `None`, optional
        Location of the file (if any) that is standardized. Required if
        ``hdulist`` is not provided.
    hdulist : `astropy.io.fits.HDUList` or `None`, optional
        HDUList to standardize. Required if ``location`` is not provided.
        If provided, and ``location`` is not given, defaults to ``:memory:``.
    config : `StandardizerConfig`, `dict` or `None`, optional
        Configuration key-values used when standardizing the file.

    Attributes
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        All HDUs found in the FITS file
    primary : `~astropy.io.fits.PrimaryHDU`
        The primary HDU.
    processable : `list`
        Any additional extensions marked by the standardizer for further
        processing. Does not include the  primary HDU if it doesn't contain any
        image data. Contains at least 1 entry.
    wcs : `list`
        WCSs associated with the processable image data. Will contain
        at least 1 WCS.
    bbox : `list`
        Bounding boxes associated with each WCS.
    """

    name = "KBMODV0_5"
    priority = -1
    configClass = KBMODV0_5Config
    can_volunteer = False

    @classmethod
    def resolveTarget(cls, tgt):
        parentCanStandardize, resources = super().resolveTarget(tgt)

        if not parentCanStandardize:
            return False

        # Check that we have at least 4 extensions (header, science, mask, variance)
        # and that the three data extensions have the same dimensions.
        hdulist = resources["hdulist"]
        if len(hdulist) < 4:
            return False
        if (hdulist[1].data is None) or (hdulist[2].data is None) or (hdulist[3].data is None):
            return False

        shape1 = hdulist[1].data.shape
        shape2 = hdulist[2].data.shape
        shape3 = hdulist[3].data.shape
        if len(shape1) != 2:
            return False
        if (len(shape2) != 2) or (shape1[0] != shape2[0]) or (shape1[1] != shape2[1]):
            return False
        if (len(shape3) != 2) or (shape1[0] != shape3[0]) or (shape1[1] != shape3[1]):
            return False

        # Check that we have all the fields we will need (just the date).
        primary = hdulist["PRIMARY"].header
        canStandardize = "DATE-AVG" in primary

        return canStandardize, resources

    def __init__(self, location=None, hdulist=None, config=None, **kwargs):
        super().__init__(location=location, hdulist=hdulist, config=config, **kwargs)

        # this is the only science-image header for the legacy deccam
        self.processable = [
            self.hdulist[1],
        ]

    def translateHeader(self):
        """Returns at least the following metadata, read from the primary header,
         as a dictionary:

        ======== ========== =========================================================
        Key      Header Key Description
        ======== ========== =========================================================
        mjd      DATE-AVG   Decimal MJD timestamp of the middle of the exposure (UTC)
        filter   FILTER     Filter band
        visit    EXPID      Exposure ID
        ======== ========== =========================================================
        """
        # this is the 1 mandatory piece of metadata we need to extract
        standardizedHeader = {}
        obs_datetime = Time(self.primary["DATE-AVG"], format="isot")
        standardizedHeader["mjd"] = obs_datetime.mjd
        standardizedHeader["mjd_mid"] = obs_datetime.mjd

        # these are all optional things
        standardizedHeader["filter"] = self.primary["FILTER"]
        standardizedHeader["visit"] = self.primary["EXPID"]

        # If no observatory information is given, default to the Deccam data
        # (Cerro Tololo Inter-American Observatory).
        standardizedHeader["obs_lon"] = self.primary.get("OBS-LONG", 70.81489)
        standardizedHeader["obs_lat"] = self.primary.get("OBS-LAT", -30.16606)
        standardizedHeader["obs_elev"] = self.primary.get("OBS-ELEV", 2215.0)

        return standardizedHeader

    def _standardizeMask(self):
        # Return empty masks if no masking is done
        if not self.config["do_mask"]:
            sizes = self._bestGuessImageDimensions()
            return (np.zeros(size) for size in sizes)

        # Otherwise load the mask extension and process it
        mask = self.hdulist[2].data

        if self.config["do_bitmask"]:
            # flip_bits makes ignore_flags into mask_these_flags
            mask = bitmask.bitfield_to_boolean_mask(
                bitfield=mask,
                ignore_flags=self.config["mask_flags"],
                flag_name_map=self.config["bit_flag_map"],
                flip_bits=True,
            )

        if self.config["do_threshold"]:
            bmask = self.processable[0].data > self.config["brightness_threshold"]
            mask = mask | bmask

        if self.config["grow_mask"]:
            # Only import the scipy module if we actually need it.
            from scipy.signal import convolve2d

            grow_kernel = np.ones(self.config["grow_kernel_shape"])
            mask = convolve2d(mask, grow_kernel, mode="same").astype(bool)

        return mask

    def standardizeMaskImage(self):
        return (self._standardizeMask() for i in self.processable)

    def standardizeVarianceImage(self):
        return (self.hdulist[3].data for i in self.processable)
