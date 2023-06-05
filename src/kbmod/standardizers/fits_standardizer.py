from abc import abstractmethod
from pathlib import Path
import warnings

import astropy.io.fits as fits
from astropy.wcs import WCS

from kbmod.standardizer import Standardizer


__all__ = ["FitsStandardizer",]


class FitsStandardizer(Standardizer):
    """Suppports processing of a single FITS file.

    Parameters
    ----------
    location : `str`
        Location of the FITS file, can be an URI or local filesystem
        path.

    Attributes
    ----------
    hdulist : `astropy.io.fits.HDUList`
        All HDUs found in the FITS file
    primary : `astropy.io.fits.PrimaryHDU`
        The primary HDU.
    exts : `list`
        Any additional extensions marked by the standardizer
        for further processing, `~astropy.io.fits.CompImageHDU` or
        `~astropy.io.fits.ImageHDU` expected. Does not include the
        primary HDU if it doesn't contain any image data. Will contain
        at least 1 extension.
    _wcs : `list`
        WCSs associated with the processable image data. Will contain
        at least 1 WCS.
    """

    extensions = [".fit", ".fits", ".fits.fz"]
    """File extensions this processor can handle."""

    @staticmethod
    def _isMultiExtFits(hdulist):
        """Returns `True` when given HDUList contains more than 1 HDU.

        Parameters
        ----------
        hdulist : `astropy.io.fits.HDUList`
            An HDUList object.
        """
        return len(hdulist) > 1

    @classmethod
    @abstractmethod
    def canStandardize(cls, tgt):
        # docstring inherited from Standardizer; TODO: check it's True
        canProcess, hdulist = False, None

        # nasty hack, should do better extensions
        fname = Path(tgt)
        extensions = fname.suffixes
        if extensions[0] in cls.extensions:
            try:
                hdulist = fits.open(tgt)
            except OSError:
                # OSError - file is corrupted, or isn't a fits
                # FileNotFoundError - bad file, let it raise
                pass
            else:
                canProcess = True

        return canProcess, hdulist
    
    def __init__(self, location):
        super().__init__(location)
        self.hdulist = fits.open(location)
        self.primary = self.hdulist["PRIMARY"].header
        self.isMultiExt = len(self.hdulist) > 1

        # This feels kind of awkwad because of the
        # the posibility that ext and wcs idx miss-align 
        self.exts = []
        self.wcs = []
    
    def computeBBox(self, header, dimX, dimY):
        """Given an Header containing WCS data and the dimensions of
        an image calculates the values of world coordinates at image
        corner and image center.

        Parameters
        ----------
        header : `object`
            The header, Astropy HDU and its derivatives.
        dimX : `int`
            Image dimension in x-axis.
        dimY : `int`
            Image dimension in y-axis.

        Returns
        -------
        standardizedBBox : `dict`
            Calculated coorinate values, a dict with,
            wcs_center_[ra, dec] and wcs_corner_[ra, dec]

        Notes
        -----
        The center point is assumed to be at the (dimX/2, dimY/2)
        pixel location. Corner is taken to be the (0,0)-th pixel.
        """
        standardizedBBox = {}
        centerX, centerY = int(dimX/2), int(dimY/2)

        with warnings.catch_warnings(record=True) as warns:
            wcs = WCS(header)
            if warns:
                for w in warns:
                    print(w)
                    #logger.warning(w.message)

        # we should have checks here that the constructed WCS
        # isn't the default empy WCS, which can happen
        centerSkyCoord = wcs.pixel_to_world(centerX, centerY)
        cornerSkyCoord = wcs.pixel_to_world(0, 0)

        standardizedBBox["center_ra"] = centerSkyCoord.ra
        standardizedBBox["center_dec"] = centerSkyCoord.dec
        
        standardizedBBox["corner_ra"] = cornerSkyCoord.ra
        standardizedBBox["corner_dec"] = cornerSkyCoord.dec

        # we'll skip doing the to-unit-sphere calculation here
        # preferring rather to let that be standardized in 1 place
        # (prob in region-search?)
#        centerRa = centerSkyCoord.ra.to(u.rad)
#        centerDec = centerSkyCoord.dec.to(u.rad)
#        
#        cornerRa = cornerSkyCoord.ra.to(u.rad)
#        cornerDec = cornerSkyCoord.dec.to(u.rad)
#        
#        unitSphereCenter = np.array([
#            np.cos(centerDec) * np.cos(centerRa),
#            np.cos(centerDec) * np.sin(centerRa),
#            np.sin(centerDec)
#        ])
#        
#        unitSphereCorner = np.array([
#            np.cos(cornerDec) * np.cos(cornerRa),
#            np.cos(cornerDec) * np.sin(cornerRa),
#            np.sin(cornerDec)
#        ])
#        
#        unitRadius = np.linalg.norm(unitSphereCenter - unitSphereCorner)
#        standardizedBBox["radius"] = unitRadius
#        
#        standardizedBBox["center_x"] = unitSphereCenter[0]
#        standardizedBBox["center_y"] = unitSphereCenter[1]
#        standardizedBBox["center_z"] = unitSphereCenter[2]
#        
#        standardizedBBox["corner_x"] = unitSphereCorner[0]
#        standardizedBBox["corner_y"] = unitSphereCorner[1]
#        standardizedBBox["corner_z"] = unitSphereCorner[2]

        return standardizedBBox
