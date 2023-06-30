from abc import abstractmethod
from pathlib import Path
import warnings

import astropy.io.fits as fits
from astropy.wcs import WCS

from kbmod.standardizer import Standardizer


__all__ = ["FitsStandardizer",]


class FitsStandardizer(Standardizer):
    """Suppports processing of a single FITS file.

    The file can be a single or multiple extension FITS file. The FITS file has
    to be openable by Astropy. The resulting `~astropy.io.fits.HDUList` is
    required to be stored in the ``hdulist`` attribute, and the HDU that
    identifies itself as ``PRIMARY`` is required to be in the ``primary``
    attribute.

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
        processing. Does not include the  primary HDU if it doesn't contain any
        image data. Contains at least 1 entry.
    wcs : `list`
        WCSs associated with the processable image data. Will contain
        at least 1 WCS.
    bbox : `list`
        Bounding boxes associated with each WCS.
    """

    extensions = [".fit", ".fits", ".fits.fz"]
    """File extensions this processor can handle."""

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

        self.exts = []
        self._wcs = []
        self._bbox = []

    @property
    def wcs(self):
        if not self._wcs:
            self._wcs = list(self.standardizeWCS())
        return self._wcs

    @property
    def bbox(self):
        if not self._bbox:
            self._bbox = list(self.standardizeBBox())
        return self._bbox

    def _computeBBox(self, wcs, dimX, dimY):
        """Given an WCS and the dimensions of an image calculates the values of
        world coordinates at image corner and image center.

        Parameters
        ----------
        wcs : `object`
            The header, Astropy HDU and its derivatives.
        dimX : `int`
            Image dimension in x-axis.
        dimY : `int`
            Image dimension in y-axis.

        Returns
        -------
        standardizedBBox : `dict`
            Calculated coorinate values, a dict with, ``wcs_center_[ra, dec]``
            and ``wcs_corner_[ra, dec]`` keys.

        Notes
        -----
        The center point is assumed to be at the (dimX/2, dimY/2) pixel
        coordinates, rounded down. Corner is taken to be the (0,0)-th pixel.
        """
        standardizedBBox = {}
        centerX, centerY = int(dimX/2), int(dimY/2)

        centerSkyCoord = wcs.pixel_to_world(centerX, centerY)
        cornerSkyCoord = wcs.pixel_to_world(0, 0)

        standardizedBBox["center_ra"] = centerSkyCoord.ra.deg
        standardizedBBox["center_dec"] = centerSkyCoord.dec.deg

        standardizedBBox["corner_ra"] = cornerSkyCoord.ra.deg
        standardizedBBox["corner_dec"] = cornerSkyCoord.dec.deg

        return standardizedBBox

    def _bestGuessImageDimensions(self):
        """Makes a best guess at the dimensions of the extensions registered
        as processable.

        The best guess methodology assumes that image sizes are encoded in the
        header NAXIS keywords for each extension registered as processable. If
        they are not, the image size is set to the best guess image size.

        The best guess image size is determined by, in order of operations:

        1) assuming the primary header encodes the image size in ``NAXIS1`` and
           ``NAXIS2`` keywords
        2) if the NAXIS keywords are not found, by attempting to read them from
           the header of the first registered processable image
        3) and if they are still not found, by loading the first processable
           extension data and using its shape to determine the dimensions.

        Thus the sizes of all extensions that do not have NAXIS keywords will
        default to what either the primary header NAXIS keywords state, the
        first registered extension NAXIS keywords state, or the dimension of
        the data in the first registered extension.

        This usually reflects the reality because focal planes are not often
        tiled with CCDs of varying sizes, and the registered extensions are
        precisely the images created by the science CCDs.

        The method performs better than determining the exact size for each
        extension from its data attribute as it doesn't load the actual data
        into memory each time.

        Returns
        -------
        sizes : `list`
            List of tuples ``(dimx, dimy)`` of the best guess sizes of the
            extensions.
        """
        # We perhaps do more work than neccessary here, but I don't have a good
        # example of a FITS file without NAXIS1 and NAXIS2 keys in at least
        # one of the headers.
        guessNaxis1 = self.primary.get("NAXIS1", None)
        guessNaxis2 = self.primary.get("NAXIS2", None)

        # NAXIS kwargs don't exist in primary
        if guessNaxis1 is None or guessNaxis2 is None:
            guessNaxis1 = self.exts[0].header.get("NAXIS1", None)
            guessNaxis2 = self.exts[0].header.get("NAXIS2", None)

        # NAXIS kwargs don't exist in primary or first extension
        if guessNaxis1 is None or guessNaxis2 is None:
            guessNaxis1, guessNaxis2 = self.exts[0].data.shape

        return (
            (
                e.header.get("NAXIS1", None) or guessNaxis1,
                e.header.get("NAXIS2", None) or guessNaxis2
            ) for e in self.exts
        )

    def standardizeWCS(self):
        """Creates an WCS for every registered processable extension in
        ``self.exts``

        Returns
        -------
        wcs : `list`
            List of WCSs.
        """
        # we should have checks here that the constructed WCS
        # isn't the default empy WCS, which can happen
        #with warnings.catch_warnings(record=True) as warns:
        #    wcs = WCS(header)
        #    if warns:
        #        for w in warns:
        #            print(w)
        #            #logger.warning(w.message)

        return (WCS(ext.header) for ext in self.exts)

    def standardizeBBox(self):
        """Computes the bounding box of every registered processable extension.

        See `~computeBBox` for more details.

        Returns
        -------
        stdBBox : `dict`
            Calculated coorinate values, a dict with, ``wcs_center_[ra, dec]``
            and ``wcs_corner_[ra, dec]`` keys.
        """
        sizes = self._bestGuessImageDimensions()
        return (
            self._computeBBox(wcs, size[0], size[1]) for wcs, size in zip(self.wcs, sizes)
        )

    def translateHeader(self, *args, **kwargs):
        """Maps the header keywords to the required and optional metadata.

        Is required to return a dictionary containing at least the following
        keys and values:

        ======== ==============================================================
        Key      Description
        ======== ==============================================================
        mjd      Decimal MJD timestamp of the start of the observation
        ra       Right Ascension in ICRS coordinate system of the extracted, or
                 calculated on-sky poisiton of the central pixel, pointing
                 data.
        dec      Declination in ICRS coordinate system, expressed in decimal
                 degrees, of the extracted, or calculated, on-sky poisiton of
                 the data.
        ======== ==============================================================

        Returns
        -------
        metadata : `dict`
            Required and optional metadata.
        """
        raise NotImplementedError("This FitsStandardizer doesn't implement a header standardizer.")

    def standardizeMetadata(self):
        # docs are inherited
        metadata = self.translateHeader()
        metadata.update({"location": self.location})
        metadata.update({"wcs": self.wcs, "bbox": self.bbox})

        # calculate the pointing from the bbox or wcs if they exist?
        # I feel like I've overthought this whole setup, like when will bbox
        # ever not be there if wcs is, what happens if WCS fails to construct
        if "ra" or "dec" not in metadata:
            # delete both?
            metadata.pop("ra", None)
            metadata.pop("dec", None)
            if all(self.bbox):
                metadata["ra"] = [bb["center_ra"] for bb in self.bbox]
                metadata["dec"] = [bb["center_dec"] for bb in self.bbox]
            elif all(self.wcs):
                # like, this is almost a copy of bbox
                sizes = self._bestGuessImageDimensions()
                metadata["ra"], metadata["dec"] = [], []
                for (dimx, dimy), wcs in zip(self.wcs, sizes):
                    centerSkyCoord = wcs.pixel_to_world(dimx/2, dimy/2)
                    metadata["ra"].append(centerSkyCoord.ra.deg)
                    metadata["dec"].append(centerSkyCoord.dec.deg)

        return metadata

    def standardizeScienceImage(self):
        return (ext.data for ext in self.exts)
