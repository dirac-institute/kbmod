"""Class for standardizing Data Products of Vera C. Rubin Science Pipelines
via the Rubin Data Butler.
"""
import importlib

import astropy.nddata as bitmask
from astropy.wcs import WCS
import numpy as np
from scipy.signal import convolve2d

from kbmod.standardizers import Standardizer, StandardizerConfig
from kbmod.search import LayeredImage, RawImage, PSF


__all__ = ["ButlerStandardizer"]


def deferred_import(module, name=None):
    """Defer the import of the some of the stack's functionality until we
    actually need it to be able to import KBMOD before the heat death of the
    universe when we do not intend to use the Rubin stack."""
    if name is None:
        name = module
    if globals().get(name, False):
        return
    try:
        globals()[name] = importlib.import_module(module)
    except ImportError as e:
        raise ImportError("No Rubin Stack found. Please activate Rubin stack.") from e


class ButlerStandardizerConfig(StandardizerConfig):
    do_mask = True
    """Perform masking if ``True``, otherwise return an empty mask."""

    do_bitmask = True
    """Mask ``mask_flags`` from the mask plane in the FITS file."""

    do_threshold = False
    """Mask all pixels above the given count threshold."""

    grow_mask = True
    """Grow mask footprint by ``grow_kernel_shape``"""

    brightness_treshold = 10
    """Pixels with value greater than this threshold will be masked."""

    grow_kernel_shape = (10, 10)
    """Size of the symmetric square kernel by which mask footprints will be
    increased by."""

    mask_flags = ["BAD", "CLIPPED", "CR", "CROSSTALK", "EDGE", "NO_DATA",
                  "SAT", "SENSOR_EDGE", "SUSPECT"]
    """List of flags that will be masked."""

    psf_std = 1
    """Standard deviation of the Point Spread Function."""


class ButlerStandardizer(Standardizer):
    """Standardizer for Vera C. Rubin Data Products, namely the underlying
    ``Exposure`` objects of ``calexp``, ``difim`` and various ``*coadd``
    dataset types.

    This standardizer will volunteer to process ``DatasetRef`` or
    ``DatasetId`` or collections of them. The Rubin Data Butler is expected to
    be provided at instantiation time, in addition to the target we want to
    standardize.

    Parameters
    ----------
    butler : Butler
        Vera C. Rubin Data Butler.
    config : `StandardizerConfig`, `dict` or `None`, optional
        Configuration key-values used when standardizing the file.
    datasetRefs : list[DatasetRef], DatasetRef or `None`, optional
        Dataset reference object(s). Must be provided if `ids` are not.
    ids : list[DatasetId], `list[int]`, DatasetId or `None`, optional
        Dataset ID(s), must be provided if `datasetRefs` are not.

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
    name = "ButlerStandardizer"
    priority = 2
    configClass = ButlerStandardizerConfig

    @classmethod
    def canStandardize(self, tgt):
        # this is pretty hacky - but I'm not importing the entire stack to get
        # a simple isinstance comparison
        if isiterable(tgt) and not isinstance(tgt, str):
            tgttype = str(type(tgt[0])).lower()
        else:
            tgttype = str(type(tgt)).lower()

        if "datasetref" in tgttype or "datasetid" in tgttype:
            return True, tgt

    # Ideally we would require this std to instantiate solely from one thing,
    # a datasetRef or datasetID but the ref is not serializable, and ids are,
    # so we need to support both to roundtrip an ImageCollection. Then on top
    # of that we need to support collections of them, because otherwise we need
    # to resolve collections of ints (ids) into data-refs somewhere and that
    # floats all the fromDatasetRefs, fromDatasetIds factories upwards.
    # Originally the proposal handled both cases here, but this is now changed
    # because it just makes more sense with respect of a definition of Std
    # being "thing that maps to LayeredImage" (and keeps all the data sources
    # in the same place). Opinions on a better way than this??
    def __init__(self, butler, datasetRef=None, id=None, config=None, **kwargs):
        super().__init__(butler.datastore.root, config=config)
        self.butler = butler

        if datasetRef is None and id is None:
            raise TypeError("DatasetRefs or DatasetIds are required "
                            "parameters, got neither.")

        # This is an optimization to avoid importing the whole stack every time
        if datasetRef is not None:
            ref = datasetRef
        else:
            if isinstance(id, int):
                deferred_import("lsst.daf.butler.core", "DatasetId")
                ref = butler.registry.getDataset(DatasetId(id))
            else:  # assume DatasetId
                ref = butler.registry.getDataset(id)

        self.ref = ref
        self.exp = butler.get(ref, collections=[ref.run, ])
        self.processable = [self.exp, ]
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

    def standardizeMetadata(self):
        metadata = {}
        metadata["location"] = self.butler.getURI(self.ref, collections=[self.ref.run, ]).geturl()
        metadata.update({"wcs": self.wcs, "bbox": self.bbox})

        metadata["mjd"] = self.exp.visitInfo.date.toAstropy().mjd
        metadata["filter"] = self.exp.info.getFilter().physicalLabel
        metadata["id"] = str(self.ref.id)
        metadata["exp_id"] = self.exp.info.id

        # Potentially over-engineered; when will bbox be there if wcs isn't,
        # what happens if WCS fails to construct?
        if "ra" not in metadata or "dec" not in metadata:
            # delete both?
            metadata.pop("ra", None)
            metadata.pop("dec", None)
            if all(self.bbox):
                metadata["ra"] = [bb["center_ra"] for bb in self.bbox]
                metadata["dec"] = [bb["center_dec"] for bb in self.bbox]
            elif all(self.wcs):
                dimx, dimy = self.exp.getWidth(), self.exp.getHeight()
                centerSkyCoord = self.wcs[0].pixel_to_world(dimx/2, dimy/2)
                metadata["ra"] = centerSkyCoord.ra.deg
                metadata["dec"] = centerSkyCoord.dec.deg

        return metadata

    # These were expected to be generators, but because we already evaluated
    # the entire exposure, i.e. we already paid the cost of disk IO
    # TODO: Add lazy eval by punting metadata standardization to visit_info
    # table in the registry and thus optimize building of ImageCollection
    def standardizeScienceImage(self):
        return [self.exp.image.array, ]

    def standardizeVarianceImage(self):
        return [self.exp.variance.array, ]

    def standardizeMaskImage(self):
        # Return empty masks if no masking is done
        if not self.config["do_mask"]:
            sizes = self._bestGuessImageDimensions()
            return (np.zeros(size) for size in sizes)

        # Otherwise load the mask extension and process it
        mask = self.exp.image.array

        if self.config["do_bitmask"]:
            # flip_bits makes ignore_flags into mask_these_flags
            mask = bitmask.bitfield_to_boolean_mask(
                bitfield=mask,
                ignore_flags=self.config["mask_flags"],
                flag_name_map=self.config["bit_flag_map"],
                flip_bits=True
            )

        if self.config["do_threshold"]:
            bmask = self.processable[0].data > self.config["brightness_threshold"]
            mask = mask | bmask

        if self.config["grow_mask"]:
            grow_kernel = np.ones(self.config["grow_kernel_shape"])
            mask = convolve2d(mask, grow_kernel, mode="same").astype(bool)

        return [mask, ]

    def standardizePSF(self):
        # TODO: Update when we formalize the PSF, Any of these are available
        # from the stack:
        # self.exp.psf.computeImage
        # self.exp.psf.computeKernelImage
        # self.exp.psf.getKernel
        # self.exp.psf.getLocalKernel
        std = self.config["psf_std"]
        return [PSF(std), ]

    def standardizeWCS(self):
        return [WCS(self.exp.wcs.getFitsMetadata()) if self.exp.hasWCS() else None, ]

    def standardizeBBox(self):
        if self.exp.hasWCS():
            dimx, dimy = self.exp.getWidth(), self.exp.getHeight()
            return [self._computeBBox(self.wcs[0], dimx, dimy), ]
        else:
            return None

    def toLayeredImage(self):
        meta = self.standardizeMetadata()
        sciences = self.standardizeScienceImage()
        variances = self.standardizeVarianceImage()
        masks = self.standardizeMaskImage()
        psfs = self.standardizePSF()

        # guaranteed to exist, i.e. safe to access
        mjds = meta["mjd"]
        imgs = []
        for sci, var, mask, psf, t in zip(sciences, variances, masks, psfs, mjds):
            imgs.append(LayeredImage(RawImage(sci), RawImage(var), RawImage(mask), t, psf))
        return imgs
