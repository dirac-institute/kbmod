import astropy.nddata as bitmask
from astropy.wcs import WCS

import numpy as np

from scipy.signal import convolve2d

from kbmod.standardizer import Standardizer
from kbmod.search import layered_image, raw_image, psf


__all__ = ["ButlerStandardizer"]


# ButlerStd has been inherited from the Standardizer even though at the moment
# it shares (duplicates) a lot of functionality already in the FitsStandardizer
# I suspect we will be replacing much of this functionality later on with
# quicker Butler queries instead of reading all of the exposures as a ExposureF
# objects (which is slow) at which point the two classes should diverge.
class ButlerStandardizer(Standardizer):

    name="ButlerStandardizer"

    mask_flags = ["BAD", "CLIPPED", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT",
                  "SENSOR_EDGE", "SUSPECT"]
    """List of flags that will be masked. The flag map is extracted from the
    Exposure object itself."""

    @classmethod
    def canStandardize(self, tgt):
        # this is pretty hacky - but I'm not importing the entire stack to get
        # a simple isinstance comparison
        if "datasetref" in str(type(tgt)).lower():
            return True, tgt
        return False, []

    def __init__(self, butler, datasetRefs=None, id=None, **kwargs):
        # this requires an invovled logic sequence where
        # 1) we need to check if butler is Butler, datasetRefs a list of
        #    DatasetRef - proceed as is
        # 2) if butler is a string - instantiate butler
        # 3) if the datRefs are a list of DatasetRefs - great
        # 4) else construct DatasetRef's
        #    4.1) They are DataId objects - butler.registry.getDataset(id)
        #    4.2) if the datRefs are strings and UUIDs - b.r.getDataset(id)
        #    4.3) mappings must specify datasetType, instrument, detector,
        #         visit, and collection, then:
        #         butler.registry.queryDatasets(datasetType='goodSeeingDiff_differenceTempExp',
        #                                       dataId= {"instrument":"DECam", "detector":35, "visit":974895},
        #                                       collections=["collection"/datref.run, ])
        #        I guess majority of these we can put in our table by default,
        #        certainly the number of columns we need is growing compared to
        #        metadata columns for the user - maybe separate into two tables
        #        and then flatten before writing and serialize the metadata
        #        about which columns go where into `meta`, but this is getting
        #        a bit too far atm, let's just raise an error in _get_std for now
        # Another question to answer here is the construction and lazy loading
        # of standardizers. FitsStandardizers have 1 input init param - location
        # so in _get_std in image collection we just pass that. A more generic
        # way would be to pass the whole row. The issue here is that __init__
        # is now having to deal with handling a mapping and that makes it less
        # obvious for the end user. Optionally, we can give a mandatory
        # **kwarg to every __init__ and rely on column naming and unpacking.
        # this is not the best, explicit better than implicit, but also now
        # the __init__ method kwargs need to be named the same as columns and
        # be optional too. For example does this init make sense to you?
        # I don't think I would be able to write this without knowing how the
        # whole thing works...
        #breakpoint()
        self.butler = butler
        self.location = butler.datastore.root
        if datasetRefs is None:
            # import this here for now, for optimization
            from lsst.daf.butler.core import DatasetId
            self.refs = [butler.registry.getDataset(DatasetId(id)),]
        else:
            self.refs = datasetRefs
        # wth? why is it a data reference if it doesn't reference data without
        # being explicit about it? Why `collection` and `collections` don't
        # behave the same way? Why is this a thing! AAARRRGGGHH!
        self.exts = [butler.get(ref, collections=[ref.run, ]) for ref in self.refs]
        self._wcs = []
        self._bbox = []

    @property
    def processable(self):
        return self.exts

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
        """Create a mask for the given Exposure objec.

        Mask is a simple edge of detector and 1 sigma treshold mask; grown by
        5 pixels each side.

        Parameters
        ----------
        hdu : `lsst.afw.image.exposure.Exposure`
            One of the Vera C. Rubin AFW ``Exposure`` objects.

        Returns
        -------
        mask : `np.array`
            Mask image.
        """
        bit_flag_map = exp.mask.getMaskPlaneDict()
        bit_flag_map = {key: 2**val for key, val in bit_flag_map.items()}
        bit_mask = bitmask.bitfield_to_boolean_mask(
            bitfield=exp.mask.array,
            ignore_flags=self.mask_flags,
            flag_name_map=bit_flag_map
        )

        brigthness_threshold = exp.image.array.mean() - exp.image.array.std()
        threshold_mask = exp.image.array > brigthness_threshold

        net_mask = bit_mask & threshold_mask

        # this should grow the mask for 5 pixels each side
        grow_kernel = np.ones((11, 11))
        grown_mask = convolve2d(net_mask, grow_kernel, mode="same")

        return grown_mask

    def standardizeWCS(self):
        # wtf is going on in the stack...
        return (WCS(exp.wcs.getFitsMetadata()) if exp.hasWcs() else None for exp in self.exts)

    def standardizeBBox(self):
        sizes = [(e.getWidth(), e.getHeight()) for e in self.exts]
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
        # by copying non-iterables values for each self.exts. So, if we wanted
        # to, we can have arbitrary, non-list element here that will be
        # copied over for each element in self.exts, for example:
        #     metadata = {"location": str(self.butler.datastore.root)}
        # This makes me wonder though, are there FITS standardizers with
        # completely standalone separate headers we need to worry about? Should
        # it be a standardizer component to unravel itself? How can we keep a
        # natural track of the indices between internal components and the
        # row-index in ImageCollection then?
        metadata = {}
        metadata["location"] = [
            self.butler.getURI(dr, collections=[dr.run,]).geturl() for dr in self.refs
        ]
        metadata.update({"wcs": self.wcs, "bbox": self.bbox})

        metadata["mjd"] = [e.visitInfo.date.toAstropy().mjd for e in self.exts]
        metadata["filter"] = [e.info.getFilter().physicalLabel for e in self.exts]
        metadata["id"] = [str(r.id) for r in self.refs]
        metadata["exp_id"] = [e.info.id for e in self.exts]

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
                sizes = [(e.getWidth(), e.getHeight()) for e in self.exts]
                metadata["ra"], metadata["dec"] = [], []
                for (dimx, dimy), wcs in zip(self.wcs, sizes):
                    centerSkyCoord = wcs.pixel_to_world(dimx/2, dimy/2)
                    metadata["ra"].append(centerSkyCoord.ra.deg)
                    metadata["dec"].append(centerSkyCoord.dec.deg)

        return metadata

    def standardizeScienceImage(self):
        return (exp.image.array for exp in self.exts)

    def standardizeVarianceImage(self):
        return (exp.variance.array for exp in self.exts)

    def standardizeMaskImage(self):
        return (self._maskSingleExp(exp) for exp in self.exts)

    def standardizePSF(self):
        return (psf(1) for e in self.exts)

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
            imgs.append(layered_image(raw_image(sci), raw_image(var), raw_image(mask), t, psf))
        return imgs
