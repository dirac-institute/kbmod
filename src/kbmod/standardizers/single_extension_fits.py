from .fits_standardizer import FitsStandardizer
from astropy.wcs import WCS


class SingleExtensionFits(FitsStandardizer):

    name = "SingleExtensionFits"
    priority = 1

    def __init__(self, location):
        super().__init__(location)
        # this here is a little bit duplicatory (is that a word?)
        # perhaps for single ext fits these could be properties
        # instead, but mostly it's ok because list will nearly always
        # contain references so no copies are actually made
        self.exts = [self.primary, ]
        self.wcs = [WCS(self.primary), ]

    @classmethod
    def canStandardize(cls, uploadedFile):
        parentCanStandardize, hdulist = super().canStandardize(uploadedFile)
        if not parentCanStandardize:
            return False, []
        
        canStandardize = parentCanStandardize and not cls._isMultiExtFits(hdulist)
        return canStandardize, hdulist

    def standardizeBBox(self):
        # I don't have a good 1 frame example that doesn't use NAXIS
        # to store dimensions, better solution is to check if this
        # fails and then load the image and check its shape
        return self.computeBBox(
            self.primary.header,
            self.primary.header["NAXIS1"],
            self.primary.header["NAXIS2"]
        )
