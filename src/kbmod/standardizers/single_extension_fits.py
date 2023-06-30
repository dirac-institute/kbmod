from .fits_standardizer import FitsStandardizer
from astropy.wcs import WCS


class SingleExtensionFits(FitsStandardizer):
    """Suppports processing of a single, single-extension, FITS file.

    The PRIMARY HDU contains the data and is the sole element of the ``exts``.
    The WCS is extracted from the PRIMARY HDU as well.

    Parameters
    ----------
    location : `str`
        Location of the FITS file, can be an URI or local filesystem path.

    Attributes
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        All HDUs found in the FITS file
    primary : `~astropy.io.fits.PrimaryHDU`
        The primary HDU.
    exts : `list`
        Any additional extensions marked by the standardizer for further
        processing, `~astropy.io.fits.CompImageHDU` or
        `~astropy.io.fits.ImageHDU` expected. Does not include the  primary HDU
        if it doesn't contain any image data. Contains at least 1 entry.
    wcs : `list`
        WCSs associated with the processable image data. Will contain
        at least 1 WCS.
    bbox : `list`
        Bounding boxes associated
    """
    name = "SingleExtensionFits"
    priority = 1

    def __init__(self, location):
        super().__init__(location)
        self.exts = [self.primary, ]

    @classmethod
    def canStandardize(cls, uploadedFile):
        parentCanStandardize, hdulist = super().canStandardize(uploadedFile)
        if not parentCanStandardize:
            return False, []

        canStandardize = parentCanStandardize and not cls._isMultiExtFits(hdulist)
        return canStandardize, hdulist
