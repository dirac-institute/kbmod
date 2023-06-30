from astropy.io.fits import CompImageHDU, PrimaryHDU, ImageHDU
from astropy.wcs import WCS

from kbmod.standardizers.fits_standardizer import FitsStandardizer


__all__ = ["MultiExtensionFits",]


class MultiExtensionFits(FitsStandardizer):
    """Suppports processing of a single, multi-extension FITS file.

    Extensions for which it's possible to extract WCS, bounding boxes and masks
    are required to be places in the ``exts`` attribute. For single extension
    FITS files this is the primary header, as it contains the image data.

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
        Bounding boxes associated with each WCS.
    """
    name = "MultiExtensionFitsStandardizer"
    priority = 1

    @staticmethod
    def _isImageLikeHDU(hdu):
        """If the given HDU contains an image, returns True. Otherwise
        returns False.

        HDU is determined to be an image if it's one of the primary,
        image or compressed image types in Astropy and its ``.data``
        attribute is not empty, but a 2D array with dimensions less
        than 6000 rows and columns.

        This is a generic best-guess implementation and there are no
        guarantees that the retrieved extensions are science images,
        i.e. images containing the actual sky, and not a small
        table-like HDU, guider or focus chip images.

        Parameters
        ----------
        hdu : `astropy.fits.HDU`
            Header unit to inspect.

        Returns
        -------
        image_like : `bool`
            True if HDU is image-like, False otherwise.
        """
        if not any((isinstance(hdu, CompImageHDU), isinstance(hdu, PrimaryHDU),
                    isinstance(hdu, ImageHDU))):
            return False

        # People store all kind of stuff even in ImageHDUs, let's make sure we
        # don't crash the server by saving 120k x 8000k table disguised as an
        # image (I'm looking at you SDSS!)
        if hdu.data is None:
            return False

        if len(hdu.data.shape) != 2:
            return False

        if hdu.shape[0] > 6000 or hdu.shape[1] > 6000:
            return False

        return True

    @classmethod
    def canStandardize(cls, location):
        parentCanStandardize, hdulist = super().canStandardize(location)

        if not parentCanStandardize:
            return False, []

        canStandardize = parentCanStandardize and cls._isMultiExtFits(hdulist)
        return canStandardize, hdulist

    def __init__(self, location, set_exts_wcs_bbox=True):
        super().__init__(location)

        self.exts = []
        for hdu in self.hdulist:
            if self._isImageLikeHDU(hdu):
                self.exts.append(hdu)
