"""`FitsStandardizer` standardize local file-system FITS files.

Prefer to use specific FITS standardizers that specify the type of the FITS
file, such as `SingleExtensionFits` or `MultiExtensionFits`, whenever possible.

`FitsStandardizer` is primarily useful to handle shared functionality and
simplify further processing, so there is not much to gain by using it directly.
"""

import abc
from os.path import isfile
from pathlib import Path

from astropy.utils import isiterable
import astropy.io.fits as fits
from astropy.wcs import WCS
import numpy as np

from ..standardizer import Standardizer, StandardizerConfig, ConfigurationError

from kbmod.core.psf import PSF
from kbmod.core.image_stack_py import LayeredImagePy


__all__ = [
    "FitsStandardizer",
    "FitsStandardizerConfig",
]


class FitsStandardizerConfig(StandardizerConfig):
    psf_std = 1
    """Standard deviation of the Point Spread Function.
    When a ``float``, uniform STD applied to all processable items in the
    standardizer. When a list, must be of the equal length as number of
    processable items, an index-matched STD per processable item.
    """

    greedy_export = False
    """When `True`, the FITS standardizer will not remove the data attribute
    from the HDUList objects after standardization. This can be useful when
    the data is needed for further processing, but can lead to increased memory."""


class FitsStandardizer(Standardizer):
    """Supports processing of a single FITS file.

    This is an `Standardizer` stub and can not be used directly. Its intended
    use is to facilitate implementing of new standardizers. If you are
    implementing a `FitsStandardizer`, consider inheriting from
    `SingleExtensionFits` or `MultiExtensionFits`.

    Standardizers inheriting from `FitsStandardizer`` require that FITS file
    is readable by AstroPy.

    Parameters
    ----------
    location : `str` or `None`, optional
        Location of the file (if any) that is standardized. Required if
        ``hdulist`` is not provided.
    hdulist : `astropy.io.fits.HDUList` or `None`, optional
        HDUList to standardize. Required if ``location`` is not provided.
        If provided, and ``location`` is not given, defaults to ``:memory:``.
    config : `StandardizerConfig` or `dict`, optional
        Configuration key-values used when standardizing the file. When not
        provided, uses `configClass` to determine the defaults configuration
        class.

    Attributes
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        All HDUs found in the FITS file
    primary : `~astropy.io.fits.PrimaryHDU`
        The primary HDU.
    processable : `list`
        Any additional extensions marked by the standardizer for further
        processing. Does not include the primary HDU if it doesn't contain any
        image data. Contains at least 1 entry.
    """

    # Standardizers we don't want to register themselves we leave nameless
    # Since FitsStd isn't usable by itself - we do not register it.
    # name = "FitsStandardizer"
    # priority = 0
    configClass = FitsStandardizerConfig

    valid_extensions = [".fit", ".fits", ".fits.fz"]
    """File extensions this processor can handle."""

    @classmethod
    def resolveFromPath(cls, tgt):
        """Successfully resolves FITS files on a local file system, that are
        readable by AstroPy.

        Parameters
        ----------
        tgt : str
            Path to FITS file.

        Returns
        -------
        canStandardize : `bool`
            `True` if target is a FITS file readable by AstroPy. `False` otherwise.
        resources : `dict`
            Empty dict when ``canStandardize`` is `False`, otherwise the
            `"hdulist"`` key contains the constructed `~fits.HDUList` object.

        Raises
        ------
        FileNotFoundError - when file doesn't exist
        """
        resources = {}
        canProcess = False

        # nasty hack, should do better extensions
        fname = Path(tgt)
        extensions = "".join(fname.suffixes)

        # if the extensions are empty, we don't think it's a FITS file
        if not extensions:
            return False, resources

        if extensions in cls.valid_extensions:
            try:
                hdulist = fits.open(tgt)
            except OSError:
                # OSError - file isn't a FITS file
                # FileNotFoundError - bad file, let it raise
                pass
            else:
                # otherwise it's a success!
                resources = {"hdulist": hdulist}
                canProcess = True

        return canProcess, resources

    @classmethod
    def resolveTarget(cls, tgt):
        """Returns ``True`` if the target is a FITS file on a local filesystem,
        that can be read by AstroPy FITS module, or an `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        tgt : str
            Path to FITS file.

        Returns
        -------
        canStandardize : `bool`
            `True` if target is a FITS file readable by AstroPy. `False` otherwise.
        resources : `dict`, optional
            An empty dictionary when ``canStandardize`` is `False`. A dict
            containing `~fits.HDUList` when ``canStandardize`` is `True`.

        Raises
        ------
        FileNotFoundError - when target is path to file that doesn't exist
        TypeError - when target is not HDUList or a filepath.
        """
        if isinstance(tgt, str):
            return cls.resolveFromPath(tgt)
        elif isinstance(tgt, fits.HDUList):
            return True, {"hdulist": tgt}
        return False, {}

    def __init__(self, location=None, hdulist=None, config=None, **kwargs):
        # TODO: oh no, ImageCollection needs to roundtrip from kwargs only.
        # This means either having a "tgt" column (I'm not sure that is such a
        # bad idea) or having STD inits that support different data sources as
        # kwargs, but understand that sometimes the kwargs can be flipped since
        # Standardizer.get can be used without kwarg. I need to improve on this
        if isinstance(location, fits.HDUList):
            hdulist = location
            location = None

        # Failure modes are:
        # - if we have neither location nor HDUList we complain
        # - if location is not a file, but we have no HDUList we complain
        if location is None and hdulist is None:
            raise ValueError("Expected location or HDUList, got neither.")

        if hdulist is None and (location == ":memory:" or not isfile(location)):
            raise FileNotFoundError("Given location is not a file, but no hdulist is given.")

        # Otherwise it's pretty normal
        # - if location is ok and no HDUList exists, read location
        # - if HDUList exists and location doesn't, try to get loc from it, put
        # :memory: otherwise
        # - if hdulist and location exist - nothing to do.
        # The object will attempt to close the hdulist when it gets GC'd
        if location is not None and hdulist is None:
            hdulist = fits.open(location)
        elif location is None and hdulist is not None:
            location = ":memory:" if hdulist.filename() is None else hdulist.filename()

        super().__init__(location, config=config, **kwargs)
        self.hdulist = hdulist
        self.primary = self.hdulist["PRIMARY"].header
        self.isMultiExt = len(self.hdulist) > 1

        self.processable = []
        self._wcs = []
        self._bbox = []

    def __del__(self):
        # Try to close if there's anything to close. Python does not guarantee
        # this method is called, or in what order it's called so various things
        # are potentially already GC'd and attempts fails. There's nothing else
        # that we can do at that point.
        hdulist = getattr(self, "hdulist", None)
        if hdulist is not None:
            try:
                hdulist.close(output_verify="ignore", verbose=False)
            except:
                pass

    def close(self, output_verify="exception", verbose=False, closed=True):
        """Close the associated FITS file and memmap object, if any.

        See `astropy.io.fits.HDUList.close`.

        Parameters
        ----------
        output_verify : `str`
            Output verification option. Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or ``"exception"``. May
            also be any combination of ``"fix"`` or ``"silentfix"`` with
            ``"+ignore"``, ``"+warn"``, or ``"+exception"`` (e.g.
            ``"fix+warn"``). See Astropy Verification Options for more info.
        verbose : bool
            When `True`, print out verbose messages.
        closed : bool
            When `True`, close the underlying file object.
        """
        self.hdulist.close(output_verify=output_verify, verbose=verbose, closed=closed)

    @property
    def wcs(self):
        if not self._wcs:
            self._wcs = list(self.standardizeWCS())
        return self._wcs

    @property
    def bbox(self):
        if not self._bbox:
            self._bbox = self.standardizeBBox()
        return self._bbox

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
            guessNaxis1 = self.processable[0].header.get("NAXIS1", None)
            guessNaxis2 = self.processable[0].header.get("NAXIS2", None)

        # NAXIS kwargs don't exist in primary or first extension
        if guessNaxis1 is None or guessNaxis2 is None:
            guessNaxis1, guessNaxis2 = self.processable[0].data.shape

        return (
            (e.header.get("NAXIS1", None) or guessNaxis1, e.header.get("NAXIS2", None) or guessNaxis2)
            for e in self.processable
        )

    def standardizeWCS(self):
        # we should have checks here that the constructed WCS isn't the default
        # empy WCS, which can happen. It is quite annoying the error is silent
        return (WCS(ext.header) for ext in self.processable)

    def standardizeBBox(self):
        sizes = self._bestGuessImageDimensions()

        # TODO: fix this once you have a BBox abstraction
        bboxes = (self._computeBBox(wcs, size[0], size[1]) for wcs, size in zip(self.wcs, sizes))
        standardizedBboxes = {
            "ra": [],
            "dec": [],
            "ra_tl": [],
            "dec_tl": [],
            "ra_tr": [],
            "dec_tr": [],
            "ra_bl": [],
            "dec_bl": [],
            "ra_br": [],
            "dec_br": [],
        }
        for bbox in bboxes:
            standardizedBboxes["ra"].append(bbox["ra"])
            standardizedBboxes["dec"].append(bbox["dec"])
            standardizedBboxes["ra_tl"].append(bbox["ra_tl"])
            standardizedBboxes["dec_tl"].append(bbox["dec_tl"])
            standardizedBboxes["ra_tr"].append(bbox["ra_tr"])
            standardizedBboxes["dec_tr"].append(bbox["dec_tr"])
            standardizedBboxes["ra_bl"].append(bbox["ra_bl"])
            standardizedBboxes["dec_bl"].append(bbox["dec_bl"])
            standardizedBboxes["ra_br"].append(bbox["ra_br"])
            standardizedBboxes["dec_br"].append(bbox["dec_br"])
        return standardizedBboxes

    @abc.abstractmethod
    def translateHeader(self, *args, **kwargs):
        """Maps the header keywords to the required and optional metadata.

        Is required to return a dictionary containing at least the following
        keys and values:

        ======== ==============================================================
        Key      Description
        ======== ==============================================================
        mjd_mid  Decimal MJD timestamp of the start of the observation (in UTC)
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
        metadata = self.translateHeader()
        metadata["location"] = self.location
        metadata.update(self.bbox)
        return metadata

    def standardizeScienceImage(self):
        # the assumption here is that all Exts are AstroPy HDU objects
        return (ext.data for ext in self.processable)

    def standardizePSF(self):
        stds = self.config["psf_std"]
        if isiterable(stds):
            if len(stds) != len(self.processable):
                raise ConfigurationError(
                    "Number of PSF STDs does not match the "
                    "declared number of processable units "
                    "requiring a PSF instance."
                )
            return (PSF.make_gaussian_kernel(std) for std in stds)
        elif isinstance(stds, (int, float)):
            return (PSF.make_gaussian_kernel(stds) for i in self.processable)
        else:
            raise TypeError("Expected a number or a list, got {type(stds)}: {stds}")

    def toLayeredImage(self):
        """Returns a list of `~LayeredImagePy` objects for each
        entry marked as processable.

        Returns
        -------
        layeredImage : `list`
            Layered image objects.
        """
        meta = self.standardizeMetadata()
        sciences = self.standardizeScienceImage()
        variances = self.standardizeVarianceImage()
        masks = self.standardizeMaskImage()

        psfs = self.standardizePSF()

        # guaranteed to exist, i.e. safe to access, but isn't unravelled here
        # or potentially it is - we can't tell?
        if isinstance(meta["mjd_mid"], (list, tuple)):
            mjds = meta["mjd_mid"]
        else:
            mjds = (meta["mjd_mid"] for e in self.processable)

        imgs = []
        for sci, var, mask, psf, t in zip(sciences, variances, masks, psfs, mjds):
            # Make sure the science and variance layers are float32.
            sci = sci.astype(np.float32)
            var = var.astype(np.float32)

            # Converts nd array mask from bool to np.float32
            mask = mask.astype(np.float32)
            imgs.append(LayeredImagePy(sci, var, mask=mask, psf=psf.astype(np.float32), time=t))

        if not self.config["greedy_export"]:
            for i in self.processable:
                if isinstance(i, (fits.ImageHDU, fits.CompImageHDU)):
                    i.__dict__.pop("data", None)

        return imgs
