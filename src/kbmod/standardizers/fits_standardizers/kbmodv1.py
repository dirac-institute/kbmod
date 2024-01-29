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

from .multi_extension_fits import MultiExtensionFits, FitsStandardizerConfig


__all__ = [
    "KBMODV1",
    "KBMODV1Config",
]


class KBMODV1Config(FitsStandardizerConfig):
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


class KBMODV1(MultiExtensionFits):
    """Standardizer for Vera C. Rubin Science Pipelines Imdiff Data Products,
    as they were produced by the Science Pipelines and procedure described in
    arXiv:2109.03296 and arXiv:2310.03678.

    This standardizer exists for backward compatibility purposes. Its use is
    not recommended. Use `ButlerStandardizer` instead.

    This standardizer will volunteer to process FITS whose primary header
    contains ``"ZTENSION"``, ``"ZPCOUNT"``, ``"ZGCOUNT"`` and ``"CCDNUM"``.

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

    name = "KBMODV1"
    priority = 2
    configClass = KBMODV1Config

    @classmethod
    def resolveTarget(cls, tgt):
        parentCanStandardize, resources = super().resolveTarget(tgt)

        if not parentCanStandardize:
            return False

        # A rough best guess here, I'm certain we can find a Rubin
        # signature somewhere in the header that's a clearer signal
        # that this is a Rubin Sci Pipe product
        hdulist = resources["hdulist"]
        primary = hdulist["PRIMARY"].header
        isRubin = all(
            ("ZTENSION" in primary, "ZPCOUNT" in primary, "ZGCOUNT" in primary, "CCDNUM" in primary)
        )

        canStandardize = parentCanStandardize and isRubin
        return canStandardize, resources

    def __init__(self, location=None, hdulist=None, config=None, **kwargs):
        super().__init__(location=location, hdulist=hdulist, config=config, **kwargs)

        # this is the only science-image header for Rubin
        self.processable = [
            self.hdulist["IMAGE"],
        ]

    def translateHeader(self):
        """Returns the following metadata, read from the primary header, as a
        dictionary:

        ======== ========== ===================================================
        Key      Header Key Description
        ======== ========== ===================================================
        mjd      DATE-AVG   Decimal MJD timestamp of the middle of the exposure
        filter   FILTER     Filter band
        visit_id IDNUM      Visit ID
        observat OBSERVAT   Observatory name
        obs_lat  OBS-LAT    Observatory Latitude
        obs_lon  OBS-LONG   Observatory Longitude
        obs_elev OBS-ELEV   Observatory elevation.
        ======== ========== ===================================================
        """
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

    def _standardizeMask(self):
        # Return empty masks if no masking is done
        if not self.config["do_mask"]:
            sizes = self._bestGuessImageDimensions()
            return (np.zeros(size) for size in sizes)

        # Otherwise load the mask extension and process it
        mask = self.hdulist["MASK"].data

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
            grow_kernel = np.ones(self.config["grow_kernel_shape"])
            mask = convolve2d(mask, grow_kernel, mode="same").astype(bool)

        return mask

    # hmm, making these generators made sense when thinking about
    # ImageCollection, makes it kinda awkward now, we could yield from
    # _stdMask but then we need to raise StopIteration
    def standardizeMaskImage(self):
        return (self._standardizeMask() for i in self.processable)

    def standardizeVarianceImage(self):
        return (self.hdulist["VARIANCE"].data for i in self.processable)
