"""Class for standardizing FITS files produced by the mocking module."""

from astropy.time import Time
from .multi_extension_fits import MultiExtensionFits, FitsStandardizerConfig


__all__ = [
    "TestDataStdConfig",
    "TestDataStd",
]


class TestDataStdConfig(FitsStandardizerConfig):
    pass


class TestDataStd(MultiExtensionFits):
    """Standardizer for test-data produced by the mocking module

    The standardizer will never volunteer to process any data, it must be
    explicitly forced.

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

    name = "TestDataStd"
    priority = 0
    configClass = TestDataStdConfig

    @classmethod
    def resolveTarget(cls, tgt):
        return False

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
        observat OBSERVAT   Observatory name
        obs_lat  OBS-LAT    Observatory Latitude
        obs_lon  OBS-LONG   Observatory Longitude
        obs_elev OBS-ELEV   Observatory elevation.
        ======== ========== ===================================================
        """
        # required
        standardizedHeader = {}
        obs_datetime = Time(self.primary["OBS-MJD"], format="mjd")
        standardizedHeader["mjd_mid"] = obs_datetime.mjd
        # optional
        standardizedHeader["observat"] = self.primary["OBSERVAT"]
        standardizedHeader["obs_lat"] = self.primary["OBS-LAT"]
        standardizedHeader["obs_lon"] = self.primary["OBS-LONG"]
        standardizedHeader["obs_elev"] = self.primary["OBS-ELEV"]
        return standardizedHeader

    def standardizeMaskImage(self):
        return (self.hdulist["MASK"].data for i in self.processable)

    def standardizeVarianceImage(self):
        return (self.hdulist["VARIANCE"].data for i in self.processable)
