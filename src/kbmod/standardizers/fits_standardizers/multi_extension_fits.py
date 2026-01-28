"""Standardizer for FITS files containing multiple extensions."""

from astropy.io.fits import CompImageHDU, PrimaryHDU, ImageHDU

from .fits_standardizer import FitsStandardizer, FitsStandardizerConfig

__all__ = [
    "MultiExtensionFits",
]


class MultiExtensionFits(FitsStandardizer):
    """Suppports processing of a single, multi-extension FITS file.

    Extensions for which it's possible to extract WCS, bounding boxes and masks
    are required to be places in the ``exts`` attribute. For single extension
    FITS files this is the primary header, as it contains the image data.

    Parameters
    ----------
    location : `str`
        Location of the FITS file, can be an URI or local filesystem path.
    set_exts : `bool`
        When `True`, finds all HDUs that are image-like and sets them as
        elements of `exts` list. Note that using the default `_isImageLike`
        implementation is rather costly as it loads the whole data into memory.

    Attributes
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        All HDUs found in the FITS file
    primary : `~astropy.io.fits.PrimaryHDU`
        The primary HDU.
    exts : `list`
        All HDUs from `hdulist` marked as "image-like" for further processing
        with KBMOD. Does not include the  primary HDU, when it doesn't contain
        any image data. Contains at least 1 entry.
    wcs : `list`
        WCSs associated with the processable image data. Will contain at least
        1 WCS.
    bbox : `list`
        Bounding boxes associated with each WCS.
    """

    # Standardizers we don't want to register themselves we leave nameless
    # Since FitsStd isn't usable by itself - we do not register it.
    # name = "MultiExtensionFitsStandardizer"
    # priority = 1
    configClass = FitsStandardizerConfig

    @staticmethod
    def _isImageLikeHDU(hdu):
        """If the given HDU contains an image, returns `True`, otherwise
        `False`.

        HDU is determined to be an image if it's one of the primary, image or
        compressed image types in Astropy and its ``.data`` attribute is not
        empty, but a 2D array with dimensions less than 6000 rows and columns.

        This is a generic best-guess implementation and there are no guarantees
        that the retrieved extensions are science images, i.e. images
        containing the actual sky, and not a small table-like HDU, guider or
        focus chip images.

        Parameters
        ----------
        hdu : `astropy.fits.HDU`
            Header unit to inspect.

        Returns
        -------
        image_like : `bool`
            True if HDU is image-like, False otherwise.
        """
        # This is already a pretty good basic test
        if not any((isinstance(hdu, CompImageHDU), isinstance(hdu, PrimaryHDU), isinstance(hdu, ImageHDU))):
            return False

        # The problem is that all kinds of things are stored as ImageHDUs, say
        # a 120k x 8000k table (I'm looking at you SDSS!). To avoid that we
        # need to check the data shape. The problem is that causes the data to
        # load from disk and that is very expensive
        if len(hdu.shape) != 2:
            return False

        if hdu.shape[0] > 6000 or hdu.shape[1] > 6000:
            return False

        if hdu.data is None:
            return False

        return True

    @classmethod
    def resolveTarget(cls, tgt):
        parentCanStandardize, res = super().resolveTarget(tgt)
        if not parentCanStandardize:
            return False, {}

        canStandardize = parentCanStandardize and len(res["hdulist"]) > 1
        return canStandardize, res

    def __init__(self, location=None, hdulist=None, config=None, set_processable=False, **kwargs):
        super().__init__(location=location, hdulist=hdulist, config=config, **kwargs)

        # do not load images from disk unless requested
        if set_processable:
            for hdu in self.hdulist:
                if self._isImageLikeHDU(hdu):
                    self.processable.append(hdu)
