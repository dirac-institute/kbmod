from astropy.io.fits import CompImageHDU, PrimaryHDU, ImageHDU
from astropy.wcs import WCS

from kbmod.standardizers.fits_standardizer import FitsStandardizer


__all__ = ["MultiExtensionFits",]


class MultiExtensionFits(FitsStandardizer):

    name = "MultiExtensionFitsStandardizer"
    priority = 1

    def __init__(self, location, set_exts_wcs_bbox=True):
        super().__init__(location)

        if set_exts_wcs_bbox:
            self._set_exts_wcs_bbox()

    def _set_exts_wcs_bbox(self):
        """Sets extensions and WCSs using the default `isImageLikeHDU`
        implementation. Should not be used if the exts and wcs
        attributes are set in the child class explicitly.
        """
        self.exts = []
        for hdu in self.hdulist:
            if self._isImageLikeHDU(hdu):
                self.exts.append(hdu)

        self.wcs = [WCS(hdu.header) for hdu in self.exts]
        self.bbox = [
            self.computeBBox(hdu.header, hdu.header["NAXIS1"], hdu.header["NAXIS2"])
            for hdu in self.exts
        ]

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
