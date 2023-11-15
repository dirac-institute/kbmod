import astropy.nddata as bitmask
from astropy.wcs import WCS

import numpy as np

from scipy.signal import convolve2d

from kbmod.standardizers import Standardizer, StandardizerConfig
from kbmod.search import LayeredImage, RawImage, PSF


__all__ = ["ButlerStandardizer"]


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


class ButlerStandardizer(Standardizer):

    name = "ButlerStandardizer"
    priority = 2
    configClass = ButlerStandardizerConfig

    @classmethod
    def canStandardize(self, tgt):
        # this is pretty hacky - but I'm not importing the entire stack to get
        # a simple isinstance comparison
        if "datasetref" in str(type(tgt)).lower():
            return True, tgt
        return False, []

    def __init__(self, butler, config=None, datasetRefs=None, id=None, **kwargs):
        super().__init__(butler.datastore.root, config=config)
        self.butler = butler

        if datasetRefs is None:
            # This branch is taken when loading the ButlerStandardizer from a
            # serialized image_collection table. DatasetId is imported here
            # for optimization
            from lsst.daf.butler.core import DatasetId
            self.refs = [butler.registry.getDataset(DatasetId(id)), ]
        else:
            self.refs = datasetRefs

        # why is it a data reference if it doesn't reference data?
        self.processable = [butler.get(ref, collections=[ref.run, ]) for ref in self.refs]
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

    def _maskSingleExp(self, exp):
        """Create a mask for the given Exposure object.

        Parameters
        ----------
        hdu : `lsst.afw.image.exposure.Exposure`
            One of the Vera C. Rubin AFW ``Exposure`` objects.

        Returns
        -------
        mask : `np.array`
            Mask image.
        """
        # Return empty masks if no masking is done
        if not self.config["do_mask"]:
            return np.zeros((exp.getHeight(), exp.getWidth()))

        # Otherwise load the mask extension and process it
        # Load the flags as defined by the Stack itself
        bit_flag_map = exp.mask.getMaskPlaneDict()
        bit_flag_map = {key: 2**val for key, val in bit_flag_map.items()}
        mask = exp.mask.array

        if self.config["do_bitmask"]:
            mask = bitmask.bitfield_to_boolean_mask(
                bitfield=mask,
                ignore_flags=self.config["mask_flags"],
                flag_name_map=bit_flag_map
            )

        if self.config["do_threshold"]:
            threshold_mask = exp.image.array > self.config["brigthness_threshold"]
            mask = mask & threshold_mask

        if self.config["grow_mask"]:
            grow_kernel = np.ones(self.config["grow_kernel_shape"])
            mask = convolve2d(mask, grow_kernel, mode="same")

        return [mask, ]

    def standardizeWCS(self):
        # wtf is going on in the stack...
        return (WCS(exp.wcs.getFitsMetadata()) if exp.hasWcs() else None for exp in self.processable)

    def standardizeBBox(self):
        sizes = [(e.getWidth(), e.getHeight()) for e in self.processable]
        return (
            self._computeBBox(wcs, size[0], size[1]) for wcs, size in zip(self.wcs, sizes)
        )

    def standardizeMetadata(self):
        # Hmm, this is somewhat interesting, in FitsStd, because the FITS pack
        # one, or multiple, exposures with some shared metadata together the
        # metadata can be broken down to a list of (ra, dec)'s and singular
        # "primary" header metadata that is shared by every extension.
        # Here, we get a datasetRef, which can be anything from the Data Repo,
        # i.e. nothing needs to be shared between the selected data. Except
        # the butler (since that's the data access handler for the selected
        # datasetRefs).
        # At the moment I unravell the standardizer returns in ImageCollection,
        # by copying non-iterables values for each self.processable. So, if we wanted
        # to, we can have arbitrary, non-list element here that will be
        # copied over for each element in self.processable, for example:
        #     metadata = {"location": str(self.butler.datastore.root)}
        # This makes me wonder though, are there FITS standardizers with
        # completely standalone separate headers we need to worry about? Should
        # it be a standardizer component to unravel itself? How can we keep a
        # natural track of the indices between internal components and the
        # row-index in ImageCollection then?
        metadata = {}
        metadata["location"] = [
            self.butler.getURI(dr, collections=[dr.run, ]).geturl() for dr in self.refs
        ]
        metadata.update({"wcs": self.wcs, "bbox": self.bbox})

        metadata["mjd"] = [e.visitInfo.date.toAstropy().mjd for e in self.processable]
        metadata["filter"] = [e.info.getFilter().physicalLabel for e in self.processable]
        metadata["id"] = [str(r.id) for r in self.refs]
        metadata["exp_id"] = [e.info.id for e in self.processable]

        # it's also like super dificult to extract any information out of the
        # object so I'll stop here. We could just dump all of the
        # exp.getMetadata().toDict() into here, but to parse that would require
        # us to make sure no dataset types have a different keys in the headers
        # than what we expect them to have - and this is basically# FitsStd all
        # over again.

        # I feel like I've overthought this whole setup, like when will bbox
        # ever not be there if wcs is, what happens if WCS fails to construct
        if "ra" not in metadata or "dec" not in metadata:
            # delete both?
            metadata.pop("ra", None)
            metadata.pop("dec", None)
            if all(self.bbox):
                metadata["ra"] = [bb["center_ra"] for bb in self.bbox]
                metadata["dec"] = [bb["center_dec"] for bb in self.bbox]
            elif all(self.wcs):
                sizes = [(e.getWidth(), e.getHeight()) for e in self.processable]
                metadata["ra"], metadata["dec"] = [], []
                for (dimx, dimy), wcs in zip(self.wcs, sizes):
                    centerSkyCoord = wcs.pixel_to_world(dimx/2, dimy/2)
                    metadata["ra"].append(centerSkyCoord.ra.deg)
                    metadata["dec"].append(centerSkyCoord.dec.deg)

        return metadata

    def standardizeScienceImage(self):
        return (exp.image.array for exp in self.processable)

    def standardizeVarianceImage(self):
        return (exp.variance.array for exp in self.processable)

    def standardizeMaskImage(self):
        return (self._maskSingleExp(exp) for exp in self.processable)

    def standardizePSF(self):
        return (PSF(1) for e in self.processable)

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
