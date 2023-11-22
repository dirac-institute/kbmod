"""Class for standardizing Data Products of Vera C. Rubin Science Pipelines
via the Rubin Data Butler.
"""
import importlib
import uuid

import astropy.nddata as bitmask
from astropy.wcs import WCS
import numpy as np
from scipy.signal import convolve2d

from kbmod.standardizers import Standardizer, StandardizerConfig
from kbmod.search import LayeredImage, RawImage, PSF


__all__ = ["ButlerStandardizer", "ButlerStandardizerConfig", ]


def deferred_import(module, name=None):
    """Imports module/class/function/name as  ``name`` into global modules.
    If module/class/function/name already exists does nothing.

    Intended for importing a large module or a library only when needed, as to
    avoid the cost of the import if the functionality depending on the imported
    library is not being used.

    Used to defer the import of Vera C. Rubin Middleware component only during
    `ButlerStandardizer` initialization so that we may be able to import KBMOD
    before the heat death of the universe.

    Parameters
    ----------
    module : `str`
        Fully qualified name of the module/submodule/class/function/name.
    name : `str` or `None`, optional
        Name the target is imported as.

    Raises
    ------
    ImportError :
        Target is not reachable. Ensure package is
        installed and visible int the environment.
    """
    if name is None:
        name = module
    if globals().get(name, False):
        return
    try:
        globals()[name] = importlib.import_module(module)
    except ImportError as e:
        raise ImportError("Module {module} or name {name} not found.") from e


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

    This standardizer will volunteer to process ``DatasetRef`` or ``DatasetId``
    objects. The Rubin Data Butler is expected to  be provided at
    instantiation time, in addition to the target we want to standardize.

    Parameters
    ----------
    tgt : `lsst.daf.butler.core.DatasetId`, `lsst.daf.butler.core.DatasetRef` or `int`
        Target to standardize.
    butler : `lsst.daf.butler.Butler`
        Vera C. Rubin Data Butler.
    config : `StandardizerConfig`, `dict` or `None`, optional
        Configuration key-values used when standardizing the file.

    Attributes
    ----------
    butler : `lsst.daf.butler.Butler`
        Vera C. Rubin Data Butler.
    ref : `lsst.daf.butler.core.DatasetRef`
        Dataset reference to the given target
    exp : `lsst.afw.image.exposure.Exposure`
        The `Exposure` object targeted by the ``ref``
    processable : `list[lsst.afw.image.exposure.Exposure]`
        Items marked as processable by the standardizer. See `Standardizer`.
    """
    name = "ButlerStandardizer"
    priority = 2
    configClass = ButlerStandardizerConfig

    @classmethod
    def resolveTarget(self, tgt):
        # DatasetId is a type alias for UUID's so it'll be like a large int or
        # hex or string of int/hex etc. We try to cast to UUID to check if str
        # is compliant
        # https://github.com/lsst/daf_butler/blob/main/python/lsst/daf/butler/_dataset_ref.py#L265
        if isinstance(tgt, uuid.UUID):
            return True

        if isinstance(tgt, str):
            try:
                uuid.UUID(tgt)
            except ValueError:
                return False
            else:
                return True

        # kinda hacky, but I don't want to import the entire Stack before I
        # absolutely know I need/have it.
        tgttype = str(type(tgt)).lower()
        if "datasetref" in tgttype or "datasetid" in tgttype:
            return True

        return False

    def __init__(self, id, butler, config=None, **kwargs):
        super().__init__(butler.datastore.root, config=config)
        self.butler = butler

        deferred_import("lsst.daf.butler.core.DatasetId", "DatasetId")
        deferred_import("lsst.daf.butler.core.DatasetRef", "DatasetRef")

        if isinstance(id, DatasetRef):
            ref = id
        elif isinstance(id, DatasetId):
            ref = butler.registry.getDataset(id)
        elif isinstance(id, (uuid.UUID, str)):
            ref = butler.registry.getDataset(DatasetId(id))
        else:
            raise TypeError(
                "Expected DatasetRef, DatasetId or an unique integer ID, "
                f"got {id} instead."
            )

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
        mask = self.exp.mask.array.astype(int)

        if self.config["do_bitmask"]:
            # flip_bits makes ignore_flags into mask_these_flags
            bit_flag_map = self.exp.mask.getMaskPlaneDict()
            bit_flag_map = {key: int(2**val) for key, val in bit_flag_map.items()}
            mask = bitmask.bitfield_to_boolean_mask(
                bitfield=mask,
                ignore_flags=self.config["mask_flags"],
                flag_name_map=bit_flag_map,
                flip_bits=True
            )

        if self.config["do_threshold"]:
            bmask = self.exp.image.array > self.config["brightness_threshold"]
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
        return [WCS(self.exp.wcs.getFitsMetadata()) if self.exp.hasWcs() else None, ]

    def standardizeBBox(self):
        if self.exp.hasWcs():
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
