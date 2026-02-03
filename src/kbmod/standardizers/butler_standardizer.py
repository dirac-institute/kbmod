"""Class for standardizing Data Products of Vera C. Rubin Science Pipelines
via the Rubin Data Butler.
"""

import importlib
import logging
import uuid

from astropy.coordinates import SkyCoord
import astropy.nddata as bitmask
import astropy.time
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
import astropy.units as u

from networkx import center
import numpy as np

from .standardizer import Standardizer, StandardizerConfig

from kbmod.core.psf import PSF

from kbmod.core.image_stack_py import LayeredImagePy

# Set up logger for this module
logger = logging.getLogger(__name__)

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

    standardize_effective_summary_stats = False
    """Include the "effective" fit metrics from SummaryStats"""

    standardize_uri = False
    """Include an URL-like path to the file"""

    wcs_fallback_points = 1000
    """Number of random points to sample across the detector when
    an astropy WCS cannot be constructed from the Rubin SkyWCS metadata."""

    wcs_fallback_sips_degree = 4
    """Degree of the SIP distortion to fit when creating a fallback WCS when
    an astropy WCS cannot be constructed from the Rubin SkyWCS metadata.
    If ``None``, no SIP distortion is fitted."""

    zero_point = 31
    """Photometric zero point to which all the science and variance will be scaled to."""

    greedy_export = False
    """If True, the standardizer will keep the Exposure object in memory
    after the LayeredImage is created. This is useful for large datasets."""


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
    butler : `lsst.daf.butler.Butler` or `list[lsst.daf.butler.Butler]`
        Vera C. Rubin Data Butler or a list of butlers. The butlers are queried
        to resolve the given target in the given order.
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

    @classmethod
    def __query_butler(self, tgt, butler):
        """Given a target and a butler, which might not contain the target
        queries the butler to resolve it. Butler failures are silenced.

        Has to be called after deffered_import.

        Parameters
        ----------
        tgt : `lsst.daf.butler.core.DatasetId`, `lsst.daf.butler.core.DatasetRef` or `int`
            Target to standardize.
        butler : `lsst.daf.butler.Butler` or `list[lsst.daf.butler.Butler]`
            Vera C. Rubin Data Butler or a list of butlers. The butlers are queried
            to resolve the given target in the given order.

        Raises
        ------
        TypeError : When given target is not a DatasetRef, DatasetId, or unique integer ID"
        """
        # including records expands the dataId to include
        # key pieces of information such as filter and band
        # loading datastore_records could be a shortcut to
        # relative path inside the repository
        if isinstance(tgt, dafButler.DatasetRef):
            ref = tgt
        elif isinstance(tgt, dafButler.DatasetId):
            ref = butler.get_dataset(tgt, dimension_records=True)
        elif isinstance(tgt, (uuid.UUID, str)):
            did = dafButler.DatasetId(tgt)
            ref = butler.get_dataset(did, dimension_records=True)
        else:
            raise TypeError("Expected DatasetRef, DatasetId or an unique integer ID, " f"got {tgt} instead.")

        return ref, butler

    def __init__(self, tgt, butler, config=None, **kwargs):
        deferred_import("lsst.daf.butler", "dafButler")

        # Sometimes we find ourselves having to process data that is
        # in the process of migration between multiple repositories.
        # We want to prioritize one of these repos as the preffered
        # source of data, but it does not yet contain all data.
        # So we check all the given butlers to resolve a target in
        # order and then skip once we get a hit. To cover the more
        # the plain single-butler use case just promote it to a list.
        if isinstance(butler, dafButler.Butler):
            butlers = [
                butler,
            ]
        else:
            butlers = butler

        for b in butlers:
            ref, butler = self.__query_butler(tgt, b)
            if ref is not None:
                continue

        if ref is None:
            raise

        # Now that target was upgraded to a ref and the correct butler
        # is know we can get the info we need from them.
        # Somewhere around w_2024_ builds the datastore.root
        # was removed as an attribute of the datastore, not sure
        # it was ever replaced with anything back-compatible. We simply
        # check for the which _datastore attribute is available for this
        # butler and then check wherther it has a root or roots attribute.
        if hasattr(butler, "datastore"):
            datastore_root = butler.datastore.root
        elif hasattr(butler, "_datastore"):
            if hasattr(butler._datastore, "root"):
                datastore_root = butler._datastore.root
            elif hasattr(butler._datastore, "roots"):
                datastore_root = butler._datastore.roots
            else:
                raise AttributeError("Butler does not have a valid datastore root attribute.")
        else:
            raise AttributeError("Butler does not have a valid datastore attribute.")

        super().__init__(str(datastore_root), config=config)
        self.ref = ref
        self.butler = butler

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

    def _fitWCSFallback(self, wcs, naxis1, naxis2, n_rand_pts, sip_degree, sample_outside_chip=True):
        """Create a simple TAN WCS centered on the detector through sampling random points.

        Parameters
        ----------
        wcs : `SkyWCS`
            Rubin SkyWCS object.
        naxis1: `int`
            naxis1 matching the dimensions of the SkyWCS BBOX
        naxis2: `int`
            naxis2 matching the dimensions of the SkyWCS BBOX
        n_rand_pts : `int`
            Number of random points to sample across the detector.
        sip_degree : `int` or `None`, optional
            Degree of the SIP distortion to fit. If ``None``, no SIP distortion
            is fitted.
        sample_outside_chip: `bool`, optional
            Whether to sample points outside of the chip bounds to interpolate rather
            than extrapolate the WCS fit. Default is `True`.
        Returns
        -------
        wcs : `astropy.wcs.WCS`
            Fitted WCS object.
        """
        if n_rand_pts <= 0:
            raise ValueError("Number of random points must be positive.")
        if sip_degree is not None and sip_degree <= 0:
            raise ValueError("SIP degree must be non-negative or None.")
        if not sample_outside_chip:
            # Sample random X, Y points across this detector
            rand_xy = np.random.rand(n_rand_pts, 2) * [naxis1, naxis2]
            rand_x, rand_y = rand_xy[:, 0], rand_xy[:, 1]
        else:
            # Expand our X, Y grid slightly beyond the bounds of the detector.
            grid_offset_percent = 0.1
            rand_xy = np.random.rand(n_rand_pts, 2) * np.array(
                [[naxis1 * (1 + 2 * grid_offset_percent), naxis2 * (1 + 2 * grid_offset_percent)]]
            )
            # Subtract our offset from each dimension to ensure we do not begin sampling at "0"
            rand_x = rand_xy[:, 0] - (naxis1 * grid_offset_percent)
            rand_y = rand_xy[:, 1] - (naxis2 * grid_offset_percent)

        # Turn our random X, Y points into an RA, Dec SkyCoord
        rand_ra, rand_dec = wcs.pixelToSkyArray(rand_x, rand_y, degrees=True)
        world_coords = SkyCoord(ra=rand_ra * u.deg, dec=rand_dec * u.deg, frame="icrs")

        # For our center point, we just use the center of the detector
        detector_center = wcs.pixelToSky(int(naxis1 // 2), int(naxis2 // 2))
        detector_center_coord = SkyCoord(
            ra=detector_center.getRa().asDegrees() * u.deg,
            dec=detector_center.getDec().asDegrees() * u.deg,
            frame="icrs",
        )

        # Fit a TAN WCS to these points, with optional SIP distortion
        return fit_wcs_from_points(
            (rand_x, rand_y), world_coords, detector_center_coord, sip_degree=sip_degree
        )

    def _computeSkyBBox(self, wcs, dimX, dimY):
        """Given an Rubin SkyWCS object and the dimensions of an image
        calculates the values of world coordinates image center and
        image corners.

        The corners are given by the following indices:

             topleft                 topright
            (0, dimX) ----------  (dimY, dimX)
              |                        |
              |           x            |
              |    (dimY/2, dimX/2)    |
              |         center         |
              |                        |
            (0, 0)    ----------  (dimY, 0)
            botleft               botright

        Parameters
        ----------
        wcs : `object`
            World coordinate system object, must support standard WCS API.
        dimX : `int`
            Maximal index in the NumPy convention x-axis, a "height".
        dimY : `int`
            Maximal index in the NumPy convention y-axis, a "width"
        return_type : `str`, optional
            A 'dict' or an 'array', the type the result is returned as.

        Returns
        -------
        standardizedBBox : `dict` or `array`
            An array of shape ``(5, 2)`` starting with center coordinate,
            then bottom left and progressing clockwise around the detector.
            When a dictionary, ``ra`` and ``dec`` keys mark center coordinates.
            The ``'ra_bl', 'dec_bl'`` mark bottom left. Progressing clocwise
            again, ``tl``, ``tr`` and ``br`` mark top left, top right and bottom
            right mark the edge position, and ``ra_`` and ``dec_`` prefix the
            coordinate.

        Notes
        -----
        The center point is assumed to be at the (dimX/2, dimY/2) pixel
        coordinates, rounded down.
        Bottom left corner is taken to be the (0,0)-th pixel and image lies
        in the first quadrant of a unit circle to match Astropy's convention.
        """
        center = wcs.pixelToSky(int(dimY // 2), int(dimX // 2))
        botleft = wcs.pixelToSky(0, 0)
        topleft = wcs.pixelToSky(0, dimX)
        topright = wcs.pixelToSky(dimY, dimX)
        botright = wcs.pixelToSky(dimY, 0)

        pts = np.array(
            [
                [center.getRa().asDegrees(), center.getDec().asDegrees()],
                [botleft.getRa().asDegrees(), botleft.getDec().asDegrees()],
                [topleft.getRa().asDegrees(), topleft.getDec().asDegrees()],
                [topright.getRa().asDegrees(), topright.getDec().asDegrees()],
                [botright.getRa().asDegrees(), botright.getDec().asDegrees()],
            ]
        )

        return pts

    @staticmethod
    def _mjd_to_obs_day(mjd_mid):
        """Convert MJD to observing day in YYYYMMDD format.

        Parameters
        ----------
        mjd_mid : `float`
            Modified Julian Date at the middle of the exposure.

        Returns
        -------
        obs_day : `int`
            Observing day in YYYYMMDD format.
        """
        observing_date = astropy.time.Time(mjd_mid, format="mjd", scale="tai")
        offset = astropy.time.TimeDelta(12 * 3600, format="sec", scale="tai")
        observing_date -= offset
        return int(observing_date.strftime("%Y%m%d"))

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

        # Note the timescales for MJD. The Butler uses TAI, but we convert
        # time stamps to UTC for consistency.
        # Name mjd into mjd_mid - make it obvious it's middle of exposure.
        self._metadata["mjd_start"] = mjd_start.utc.mjd
        self._metadata["mjd_mid"] = half_way.utc.mjd
        self._metadata["obs_day"] = ButlerStandardizer._mjd_to_obs_day(half_way.utc.mjd)

        self._metadata["object"] = visit.object
        self._metadata["pointing_ra"] = visit.boresightRaDec.getRa().asDegrees()
        self._metadata["pointing_dec"] = visit.boresightRaDec.getDec().asDegrees()
        self._metadata["airmass"] = visit.boresightAirmass
        obs = visit.getObservatory()
        self._metadata["obs_lon"] = obs.getLongitude().asDegrees()
        self._metadata["obs_lat"] = obs.getLatitude().asDegrees()
        self._metadata["obs_elev"] = obs.getElevation()

        # Pointing information is hard to standardize because the
        # dimensions of the detector are not easily availible. We get
        # those from BBox (in-pixel bounding box). NAXIS values are
        # required if we reproject, so we must extract them if we can
        bbox_ref = self.ref.makeComponentRef("bbox")
        bbox = self.butler.get(bbox_ref)
        self._naxis1 = bbox.getWidth()
        self._naxis2 = bbox.getHeight()

        # If the standardizer is re-used, many generators will be
        # depleted, returning None as values. Cast to dict to make
        # a copy.
        wcs_ref = self.ref.makeComponentRef("wcs")
        wcs = self.butler.get(wcs_ref)
        try:
            meta = dict(wcs.getFitsMetadata())
            meta["NAXIS1"] = self._naxis1
            meta["NAXIS2"] = self._naxis2
            self._wcs = WCS(meta)
        except Exception as e:
            logger.debug(f"Could not parse WCS metadata for {self.ref}, got {e}. Creating fallback fit.")
            # Create a simple TAN WCS centered on the detector through sampling random points.
            n_rand_pts = self.config["wcs_fallback_points"]
            sip_degree = self.config["wcs_fallback_sips_degree"]
            self._wcs = self._fitWCSFallback(wcs, self._naxis1, self._naxis2, n_rand_pts, sip_degree)

        center_pt = bbox.getCenter()
        self._metadata["pixel_scale"] = wcs.getPixelScale(center_pt).asArcseconds()

        # calculate the WCS "error" (max difference between edge coordinates
        # from Rubin's more powerful SkyWCS and Atropy's Fits-WCS)
        skyBBox = self._computeSkyBBox(wcs, self._naxis2, self._naxis1)
        apyBBox = self._computeBBoxArray(self._wcs, self._naxis2, self._naxis1)
        self._metadata["wcs_err"] = (skyBBox - apyBBox).max()

        # TODO: see issue #666
        # this will unroll the entire bbox into columns
        # make sky bbox the default, since that one is guaranteed to be correct
        self._bbox = self._bboxArrayToDict(skyBBox)
        self._metadata.update(self._bbox)

        # We need to fetch summary stats for the zero-point, so we
        # might as well extract the rest out of it, exception is
        # only the effective metrics, which may not exists for all filters
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

            # Note that the following metadata keys may not be present in all
            # Rubin butlers, and we only extract them if available.
            if "DTNSANAM" in meta:
                self._metadata["DTNSANAM"] = meta["DTNSANAM"]
            if "AIRMASS" in meta:
                self._metadata["AIRMASS"] = meta["AIRMASS"]
            d2s = 0.0
            if "DIMM2SEE" in meta and meta["DIMM2SEE"] != "NaN":
                self._metadata["DIMM2SEE"] = d2s
            if "GAINA" in meta:
                self._metadata["GAINA"] = meta["GAINA"]
            if "GAINB" in meta:
                self._metadata["GAINB"] = meta["GAINB"]

        # Will be nan for VR filter so it's optional
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
        else:
            # Save the full string representation of ref.
            self._metadata["location"] = str(self.ref)

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
        zp_correct = 10 ** ((self._metadata["zeroPoint"] - self.config.zero_point) / 2.5)
        return [
            self.exp.image.array / zp_correct,
        ]

    def standardizeVarianceImage(self):
        self.exp = self.butler.get(self.ref) if self.exp is None else self.exp
        zp_correct = 10 ** ((self._metadata["zeroPoint"] - self.config.zero_point) / 2.5)
        return [
            self.exp.variance.array / zp_correct**2,
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
            # Only import the scipy module if we actually need it.
            from scipy.signal import convolve2d

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
        return [PSF.make_gaussian_kernel(std)]

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
        mask = masks[0].astype(np.float32)
        imgs = [
            LayeredImagePy(
                self.standardizeScienceImage()[0],
                self.standardizeVarianceImage()[0],
                mask=mask,
                psf=self.standardizePSF()[0],
                time=self._metadata["mjd_mid"],
            ),
        ]
        if not self.config["greedy_export"]:
            self.exp = None
        return imgs
