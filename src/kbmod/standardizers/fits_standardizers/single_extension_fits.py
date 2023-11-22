from .fits_standardizer import FitsStandardizer, FitsStandardizerConfig


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
    # Standardizers we don't want to register themselves we leave nameless
    # Since FitsStd isn't usable by itself - we do not register it.
    # name = "SingleExtensionFits"
    # priority = 1
    configClass = FitsStandardizerConfig

    def __init__(self, location=None, hdulist=None, config=None, **kwargs):
        super().__init__(location=location, hdulist=hdulist, config=config, **kwargs)
        self.processable = [self.primary, ]

    @classmethod
    def resolveTarget(cls, tgt):
        parentCanStandardize, res = super().resolveTarget(tgt)
        if not parentCanStandardize:
            return False, {}

        canStandardize = parentCanStandardize and len(res["hdulist"]) == 1
        return canStandardize, res
