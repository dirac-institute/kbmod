"""Class for standardizing Data Products of Vera C. Rubin Science Pipelines
via the Rubin Data Butler.
"""

import importlib
import uuid

import astropy.nddata as bitmask
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u

import numpy as np
from scipy.signal import convolve2d

from .standardizer import Standardizer, StandardizerConfig
from kbmod.search import LayeredImage, RawImage, PSF


__all__ = [
    "ButlerStandardizer",
    "ButlerStandardizerConfig",
]


def deferred_import(module, name=None):
    """Imports module/class/function/name as ``name`` into global modules.
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
        raise ImportError(f"Module {module} or name {name} not found.") from e


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

    mask_flags = ["BAD", "CLIPPED", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SENSOR_EDGE", "SUSPECT"]
    """List of flags that will be masked."""

    psf_std = 1
    """Standard deviation of the Point Spread Function."""

    standardize_metadata = True
    """Fetch and include values from Rubin's Exposure.metadata
    PropertyList such as 'OBSID', 'AIRMASS', 'DIMM2SEE' ec. Typically
    corresponds to ingested header values, ergo the capitalized column
    names.
    """

    standardize_summary_stats = True
    """Fetch and include values from Rubin's SummaryStats DatasetType.
    Includes photometric and astrometric fit metrics like 'psfSigma',
    'psfArea', 'zeroPoint', 'skyBg' etc. Typically camel-case names.
    """

    standardize_effective_summary_stats = False
    """Include the "effective" fit metrics from SummaryStats"""

    standardize_uri = False
    """Include an URL-like path to to the file"""


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

    def __init__(self, dataId, butler, config=None, **kwargs):
        deferred_import("lsst.daf.butler", "dafButler")

        # Somewhere around w_2024_ builds the datastore.root
        # was removed as an attribute of the datastore, not sure
        # it was ever replaced with anything back-compatible
        try:
            super().__init__(str(butler._datastore.root), config=config)
        except AttributeError:
            super().__init__(butler.datastore.root, config=config)

        self.butler = butler

        # including records expands the dataId to include
        # key pieces of information such as filter and band
        # loading datastore_records could be a shortcut to
        # relative path inside the repository
        if isinstance(dataId, dafButler.DatasetRef):
            ref = dataId
        elif isinstance(dataId, dafButler.DatasetId):
            ref = butler.get_dataset(dataId, dimension_records=True)
        elif isinstance(dataId, (uuid.UUID, str)):
            did = dafButler.DatasetId(dataId)
            ref = butler.get_dataset(did, dimension_records=True)
        else:
            raise TypeError(
                "Expected DatasetRef, DatasetId or an unique integer ID, " f"got {dataId} instead."
            )

        self.ref = ref
        self.exp = None
        self.processable = [self.exp]

        # This is for lazy loading
        self._bbox = None
        self._wcs = None

        # This is set for internal use because often we can't share
        # between methods and they all need these values. Centralizing
        # metadatada fetching via butler also helps performance
        self._metadata = None
        self._naxis1 = None
        self._naxis2 = None

    def _fetch_meta(self):
        """Fetch metadata and any dataset components that do not
        load the image or large amount of data.

        This resolves the majority of the metadata, temporal and
        spatial information required by KBMOD, which is stored
        in the attributes and then returned in the standardize
        mehtods. This prevents having to evaluate queries to the
        Butler Registry and Datastore repeatedly, and enables the
        evalution of information that spans multiple datasets without
        forcing the Butler.get of a larger dataset (f.e. the Exposure
        object).
        """
        self._metadata = {}

        # First we standardize the required metadata. Most of this can
        # be extracted from the dataset reference and a visitInfo.
        # This includes i) data required to roundtrip the standardizer
        # ii) timestamp and iii) pointing information
        self._metadata["dataId"] = str(self.ref.id)
        self._metadata["collection"] = self.ref.run
        self._metadata["datasetType"] = self.ref.datasetType.name
        self._metadata["visit"] = self.ref.dataId["visit"]
        self._metadata["detector"] = self.ref.dataId["detector"]
        self._metadata["band"] = self.ref.dataId["band"]
        self._metadata["filter"] = self.ref.dataId["physical_filter"]

        visit_ref = self.ref.makeComponentRef("visitInfo")
        visit = self.butler.get(visit_ref)
        expt = visit.exposureTime
        mjd_start = visit.date.toAstropy()
        half_way = mjd_start + (expt / 2) * u.s + 0.5 * u.s
        self._metadata["exposureTime"] = expt

        # Note the timescales for MJD
        # Name mjd into mjd_mid - make it obvious it's mdidle of exposure
        # and add time scale like mjd_mid_utc
        self._metadata["mjd_start"] = mjd_start.mjd
        self._metadata["mjd"] = half_way.mjd

        self._metadata["object"] = visit.object
        self._metadata["pointing_ra"] = visit.boresightRaDec.getRa().asDegrees()
        self._metadata["pointing_dec"] = visit.boresightRaDec.getDec().asDegrees()
        self._metadata["airmass"] = visit.boresightAirmass

        # Pointing information is hard to standardize because the
        # dimensions of the detector are not easily availible. We get
        # those from BBox (in-pixel bounding box). NAXIS values are
        # required if we reproject, so we must extract them if we can
        bbox_ref = self.ref.makeComponentRef("bbox")
        self._bbox = self.butler.get(bbox_ref)
        self._naxis1 = self._bbox.getWidth()
        self._naxis2 = self._bbox.getHeight()

        wcs_ref = self.ref.makeComponentRef("wcs")
        wcs = self.butler.get(wcs_ref)
        meta = wcs.getFitsMetadata()
        meta["NAXIS1"] = self._naxis1
        meta["NAXIS2"] = self._naxis2
        self._wcs = WCS(meta)

        # TODO: see issue #666
        # this will unroll the entire bbox into columns
        bbox = self._computeBBox(self._wcs, self._naxis1, self._naxis2)
        self._metadata.update(bbox)

        # The rest of the data here is optional, generally metadata
        # is nice to standardize, but keys may change between
        # different instruments, summary stats are useful for
        # photometric analysis of the results, while the effective
        # values are too often NaN. The URI location itself is
        # ultimately not very useful, but helpful for data inspection.
        if self.config.standardize_metadata:
            meta_ref = self.ref.makeComponentRef("metadata")
            meta = self.butler.get(meta_ref)

            # dataId sometimes doesn't have a filter or a band,
            # depending on the way the initial ref is resolved? Why
            # is middleware so complicated! Best-effort attempt,
            # 90% cases?
            self._metadata["OBSID"] = meta["OBSID"]
            self._metadata["DTNSANAM"] = meta["DTNSANAM"]
            self._metadata["AIRMASS"] = meta["AIRMASS"]
            d2s = 0.0 if meta["DIMM2SEE"] == "NaN" else float(meta["DIMM2SEE"])
            self._metadata["DIMM2SEE"] = d2s
            self._metadata["GAINA"] = meta["GAINA"]
            self._metadata["GAINB"] = meta["GAINB"]

        if self.config.standardize_summary_stats:
            summary_ref = self.ref.makeComponentRef("summaryStats")
            summary = self.butler.get(summary_ref)
            self._metadata["psfSigma"] = summary.psfSigma
            self._metadata["psfArea"] = summary.psfArea
            self._metadata["nPsfStar"] = summary.nPsfStar
            self._metadata["zeroPoint"] = summary.zeroPoint
            self._metadata["skyBg"] = summary.skyBg
            self._metadata["skyNoise"] = summary.skyNoise
            self._metadata["meanVar"] = summary.meanVar
            self._metadata["astromOffsetMean"] = summary.astromOffsetMean
            self._metadata["astromOffsetStd"] = summary.astromOffsetStd

            # Will be nan because for VR filter
            if self.config.standardize_effective_summary_stats:
                self._metadata["effTime"] = summary.effTime
                self._metadata["effTimePsfSigmaScale"] = summary.effTimePsfSigmaScale
                self._metadata["effTimeSkyBgScale"] = summary.effTimeSkyBgScale
                self._metadata["effTimeZeroPointScale"] = summary.effTimeZeroPointScale

        if self.config.standardize_uri:
            self._metadata["location"] = self.butler.getURI(
                self.ref,
                collections=[
                    self.ref.run,
                ],
            ).geturl()

    @property
    def wcs(self):
        if self._wcs is None:
            self._fetch_meta()
        return [
            self._wcs,
        ]

    @property
    def bbox(self):
        if self._bbox is None:
            self._fetch_meta()
        return [
            self._bbox,
        ]

    def standardizeMetadata(self):
        if self._metadata is None:
            self._fetch_meta()
        return self._metadata

    def standardizeScienceImage(self):
        self.exp = self.butler.get(self.ref) if self.exp is None else self.exp
        return [
            self.exp.image.array,
        ]

    def standardizeVarianceImage(self):
        self.exp = self.butler.get(self.ref) if self.exp is None else self.exp
        return [
            self.exp.variance.array,
        ]

    def standardizeMaskImage(self):
        self.exp = self.butler.get(self.ref) if self.exp is None else self.exp
        if self._naxis1 is None or self._naxis2 is None:
            self._fetch_meta()

        # Return empty masks if no masking is done
        if not self.config["do_mask"]:
            return (np.zeros((self._naxis1, self._naxis2)) for size in sizes)

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
                flip_bits=True,
            )

        if self.config["do_threshold"]:
            bmask = self.exp.image.array > self.config["brightness_threshold"]
            mask = mask | bmask

        if self.config["grow_mask"]:
            grow_kernel = np.ones(self.config["grow_kernel_shape"])
            mask = convolve2d(mask, grow_kernel, mode="same").astype(bool)

        return [
            mask,
        ]

    def standardizePSF(self):
        # TODO: Update when we formalize the PSF, Any of these are available
        # from the stack:
        # self.exp.psf.computeImage
        # self.exp.psf.computeKernelImage
        # self.exp.psf.getKernel
        # self.exp.psf.getLocalKernel
        std = self.config["psf_std"]
        return [
            PSF(std),
        ]

    # These exist because standardizers promise to return lists
    # for compatiblity for single-data and multi-data sources
    def standardizeWCS(self):
        if self._wcs is None:
            self._fetch_meta()
        return [
            self._wcs,
        ]

    def standardizeBBox(self):
        if self._bbox is None:
            self._fetch_meta()
        return [
            self._bbox,
        ]

    def toLayeredImage(self):
        masks = self.standardizeMaskImage()
        # This is required atm because RawImage can not
        # support different types, TODO: update when fixed
        mask = masks[0].astype(np.float32)
        imgs = [
            LayeredImage(
                self.standardizeScienceImage()[0],
                self.standardizeVarianceImage()[0],
                mask,
                self.standardizePSF()[0],
                self._metadata["mjd"],
            ),
        ]
        return imgs
