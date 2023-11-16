"""`Standardizer` converts data from a given data source into a `LayeredImage`
object, applying data-source specific, transformations in the process. A
layered image is a collection containing:
 * science image
 * variance image
 * mask image
along with the:
 * on-sky coordinates of the pixel in the center of the science image
 * observation timestamp
 * location of the data source

When possible, standardizers should attempt to extract a valid WCS and/or
bounding box from the data source.
"""
import os
import abc
import glob
import warnings


__all__ = ["Standardizer", "StandardizerConfig", "ConfigurationError"]


class ConfigurationError(Exception):
    """Error that is raised when configuration parameters contain a logical error."""
    pass


class StandardizerConfig:
    """Standardizer configuration.

    Not all standardizers will (can) use the same parameters so refer to their
    respective documentation for a more complete list.

    Parameters
    ----------
    config : `dict`, `StandardizerConfig` or `None`, optional
        Collection of configuration key-value pairs.
    kwargs : optional
        Keyword arguments, assigned as configuration key-values.
    """
    def __init__(self, config=None, **kwargs):
        # This is a bit hacky, but it makes life a lot easier because it
        # enables automatic loading of the default configuration and separation
        # of default config from instance bound config
        keys = list(set(dir(self.__class__)) - set(dir(StandardizerConfig)))

        # First fill out all the defaults by copying cls attrs
        self._conf = {k: getattr(self, k) for k in keys}

        # Then override with any user-specified values
        if config is not None:
            self._conf.update(config)
        self._conf.update(kwargs)

    # now just shortcut the most common dict operations
    def __getitem__(self, key):
        return self._conf[key]

    def __setitem__(self, key, value):
        self._conf[key] = value

    def __str__(self):
        res = f"{self.__class__.__name__}("
        for k, v in self.items():
            res += f"{k}: {v}, "
        return res[:-2]+")"

    def __len__(self):
        return len(self._conf)

    def __contains__(self, key):
        return key in self._conf

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._conf == other._conf
        elif isinstance(other, dict):
            return self._conf == other
        else:
            return super().__eq__(other)

    def __iter__(self):
        return iter(self._conf)

    def __or__(self, other):
        if isinstance(other, type(self)):
            return self.__class__(config=other._conf | self._conf)
        elif isinstance(other, dict):
            return self.__class__(config=self._conf | other)
        else:
            raise TypeError("unsupported operand type(s) for |: {type(self)} "
                            "and {type(other)}")

    def keys(self):
        """A set-like object providing a view on config's keys."""
        return self._conf.keys()

    def values(self):
        """A set-like object providing a view on config's values."""
        return self._conf.values()

    def items(self):
        """A set-like object providing a view on config's items."""
        return self._conf.items()

    def update(self, conf=None, **kwargs):
        """Update this config from dict/other config/iterable.

        A dict-like update. If ``conf`` is present and has a ``.keys()``
        method, then does:  ``for k in conf: this[k] = conf[k]``. If ``conf``
        is present but lacks a ``.keys()`` method, then does:
        ``for k, v in conf: this[k] = v``.

        In either case, this is followed by:
        ``for k in kwargs:  this[k] = kwargs[k]``
        """
        if conf is not None:
            self._conf.update(conf)
        self._conf.update(kwargs)

    def toDict(self):
        """Return this config as a dict."""
        return self._conf


class Standardizer(abc.ABC):
    """Transforms given data into the format required by KBMOD.

    The given data are FITS images, Vera C. Rubin Science Pipeline objects and
    other data that provides access to the images and observation metadata.
    Standardizers are required to extract certain information from this data.

    The required metadata are the location of the data, timestamp and the
    on-sky coordinates of the observation. The required image data is the
    science exposure, variance and mask.

    The location of the data is a local filesystem path, URI or an URL.
    Timestamp is the decimal MJD of the start of the exposure.
    The on-sky coordinates are a pair of (ra, dec) values in decimal degrees of
    the pointing information extracted from the header or the central pixel
    on-sky coordinates calculated via WCS.

    The science exposure and variance are self-explanatory and the mask is a
    simple bitmask in which pixels with value 1 will not be included in KBMOD
    search.

    The standardizers are allowed to extract any additional information
    desired, f.e. such as filter, but there are no guarantess that this
    information is present for all availible standardizers.

    Of the optional metadata two properties are of particular interest - the
    WCS and the bounding box of each science exposure. They underpin the way
    the exposures of interest are grouped together.
    KBMOD does not insist on providing them. however, requiring instead just a
    more general "pointing" information. For these reason the WCS and BBOX have
    their own standardize methods, and should be provided whenever possible.

    Data is expected to be unravelled on a per-science exposure level. Data
    that shares some metadata, such as timestamp or filter for example, for
    multiple science exposures is expected to unravel that metadata and
    associate it with each of the science exposures individually.

    The data for which this standardization and unravelling will occur needs to
    be appended to the items in ``processable`` attribute of the class.

    This class is an abstract base class to serve as a recipe for implementing
    a Standardizer specialized for processing an particular dataset. Do not
    attempt to instantiate this class directly.

    Parameters
    ----------
    location : `str`
        Location of the file to standardize, a filepath or URI.

    Attributes
    ----------
    location : `str`
        Location of the file being standardized.
    processable : `list`
        List of standardizers internal units that can be processed by KBMOD.
        For example, for FITS files on local filesystem these could be the
        AstroPy header units (`~astropy.io.fits.HDUList` elements), for Vera C.
        Rubin their internal `Exposure*` objects etc. The number of processable
        units will match the number of returned  standardized images,
        variances, masks and PSFs.
    wcs : `list`
        WCSs associated with the processable image data.
    bbox : `list`
        Bounding boxes associated with each processable image data.
    """

    registry = dict()
    """All registered upload standardizer classes."""

    name = None
    """Processor's name. Only named standardizers will be registered."""

    priority = 0
    """Priority. Standardizers with high priority are prefered over
    standardizers with low priority when processing an upload.
    """

    configClass = StandardizerConfig
    """Standardizer's configuration. A class whose attributes set the behavior
    of the standardization."""

    @classmethod
    def get(cls, tgt=None, standardizer=None):
        """Get the standardizer class that can handle given file.

        When the standardizer is registered, it can be requested by its name.
        See ``self.registry`` for a list of registered Standardizers.

        When the correct standardizer is not known, the the target can be
        provided. The Standardizer with the highest priority, that also marks
        the target as processable, will be returned.

        At least one of either the target or the standardizer parameters have
        to be given.

        Parameters
        ----------
        tgt : any
            The target to be standardized.
        standardizer : `str` or `cls`
            Force the use of the given Standardizer. The given name must be a
            part of the registered Standardizers. If a class reference is
            given, returns it (no-op).

        Returns
        -------
        standardizer : `cls`
            Standardizer class that can process the given upload.

        Raises
        ------
        ValueError
            When neither target or standardizer are given.
        ValueError
            When no standardizer that marks the target as processable is found.
        TypeError
            When no standardizer that marks the target as processable is found.
        KeyError
            When the given standardizer name is not a part of the standardizer
            registry.
        """
        if standardizer is None and tgt is None:
            raise ValueError("Standardizer or a target are required.")

        # A particular standardizer was requested here, or is being forced
        if standardizer is not None and isinstance(standardizer, type):
            return standardizer
        elif standardizer is not None and isinstance(standardizer, str):
            try:
                return cls.registry[standardizer]
            except KeyError as e:
                raise KeyError(
                    "Standardizer must be a registered standardizer name or a "
                    f"class reference. Expected {', '.join([std for std in cls.registry])} "
                    f"got '{standardizer}' instead. "

                ) from e

        # The standardizer is unknown, check which standardizers volunteer and
        # return the highest priority one. The canStandardize rule is that the
        # first element has to be canStd bool, but more values can be returned
        # for optimization purposes; f..e. acquisition of expensive resources
        # like hdulist. We will immediately consume these resources in one of
        # the factory functions later; f.e. fromFile, fromHDUList etc. We do
        # not want to close the resource and pay the acquisition cost again, so
        # ignore the warnings.
        standardizers = []
        for standardizer in cls.registry.values():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                canStandardize, *_ = standardizer.canStandardize(tgt)
            if canStandardize:
                standardizers.append(standardizer)

        def get_priority(standardizer):
            """Return standardizers priority."""
            return standardizer.priority
        standardizers.sort(key=get_priority, reverse=True)

        if standardizers:
            if len(standardizers) > 1:
                names = [std.name for std in standardizers]
                warnings.warn("Multiple standardizers declared ability to "
                              f"standardize; using {names[0]}.")
            return standardizers[0]
        else:
            raise ValueError("None of the registered standardizers are able "
                             "to process this source. You can provide your "
                             "own. Refer to  Standardizer documentation for "
                             "further details.")

    @classmethod
    def fromHDUList(cls, hdul, forceStandardizer=None, config=None, **kwargs):
        """Return a Standardizer(s) that can standardize the given HDUList(s).

        Parameters
        ----------
        hdul : `astropy.io.fits.HDUList`
            ``HDUList`` to standardize
        forceStandardizer : `str`, `class` or `None`
            Standardizer class to use when mapping the file content. When
            ``None`` standardizer will automatically be determined from the
            provided file.
        stdConfig : `StandardizerConfig` or `dict`
            Standardization configuration.
        **kwargs : `dict`
            Passed onto the matching Standardizer.

        Returns
        -------
        standardizer : `object`
            Standardizer that can process the given source.

        Raises
        ------
        ValueError
            None of the registered processors can process the upload.
        ValueError
            When given standardizer name is not a registered Standardizer.
        """
        standardizer = cls.get(tgt=hdul, standardizer=forceStandardizer)
        return standardizer(hdulist=hdul, config=config, **kwargs)

    @classmethod
    def fromHDULists(cls, hduls, forceStandardizer=None, config=None, **kwargs):
        """Get a list of standardizers that can standardize the list of
        ``HDUList``."""
        return [cls.fromHDUList(hdul=hdul, forceStandardizer=forceStandardizer, config=config, **kwargs)
                for hdul in hduls]

    # Punting file, path, files, dir, dirs to lsst.resources would remove the
    # need to actually check for all this, we just use resource.open
    @classmethod
    def fromFile(cls, path, forceStandardizer=None, config=None, **kwargs):
        """Return a single Standardizer that can standardize the given file.

        Parameters
        ----------
        path : `str`
            File path.
        forceStandardizer : `str`, `class` or `None`
            Standardizer class to use when mapping the file content. When
            ``None`` standardizer will be determined automatically.
        config : `StandardizerConfig` or `dict`
            Standardization configuration.
        **kwargs : `dict`
            Passed onto the matching Standardizer.

        Returns
        -------
        standardizer : `object`
            Standardizer that can process the given source.

        Raises
        ------
        FileNotFoundError
            No such file exists, or the given path is not a file.
        ValueError
            None of the registered processors can process the upload.
        ValueError
            When given standardizer name is not a registered Standardizer.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("No such file {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError("Path does not point to a a file {path}")
        standardizer = cls.get(tgt=path, standardizer=forceStandardizer)
        return standardizer(location=path, config=config, **kwargs)

    @classmethod
    def fromFiles(cls, paths, forceStandardizer=None, config=None, **kwargs):
        """Get a list of standardizers that can standardize the files."""
        return [cls.fromFile(path=path, forceStandardizer=forceStandardizer, config=config, **kwargs)
                for path in paths]

    @classmethod
    def fromDir(cls, dirpath, recursive=False, forceStandardizer=None, config=None, **kwargs):
        """Get a list of standardizers that can standardize all FITS files
        found in the given directory."""
        if not os.path.isdir(dirpath):
            raise FileNotFoundError("Path does not point to a a directory {path}")

        # return empty or raise?
        fits_files = glob.glob(os.path.join(dirpath, "*fits*"), recursive=recursive)
        if len(fits_files) == 0:
            return []

        return cls.fromFiles(fits_files, forceStandardizer, config, **kwargs)

    # The proliferation of Butler related factory methods is a little bit
    # concerning because the Butler has many (perhaps too many) ways to get a
    # Dataset. Whereas location (URI or posix filepath) and HDUList cover all
    # eventualities in the case of file-based formats. It is possible this will
    # eventually have to be refactored into a factory that would live next to
    # the butlerstd, providing the, currently lacking, context.
    @classmethod
    def fromDatasetRef(cls, butler, refs config=None, **kwargs):
        """Get a single ButlerStandardizer for the ``DatasetRef``

        Parameters
        ----------
        butler : `~lsst.daf.butler.Butler`
            Vera C. Rubin Data Butler.
        refs : `lsst.daf.butler.core.DatasetRef`
            ``DatasetRef`` to standardize.
        **kwargs : `dict`
            Keyword arguments passed onto `ButlerStandardizer`.

        Returns
        -------
        standardizer : `ButlerStandardizer`
            Butler standardizer
        """
        standardizer = cls.get(standardizer="ButlerStandardizer")
        return standardizer(butler=butler, ref=ref, config=config, **kwargs)

    @classmethod
    def fromDatasetRefs(cls, butler, refs, **kwargs):
        """Get a list of `ButlerStandardizer`s for a collection of ``DatasetRef``s."""
        return [cls.fromDatasetRef(butler=butler, ref=ref, **kwargs) for ref in refs]

    @classmethod
    def fromDatasetId(cls, butler, id, config=None, **kwargs):
        """Get a single `ButlerStandardizer` for the given ``id``.

        Parameters
        ----------
        butler : `lsst.daf.butler.Butler`
            Vera C. Rubin Data Butler.
        id : `lsst.daf.butler.core.DatasetId` or `int`
            ``DatasetId`` or the unique integer identifier of a ``Dataset``
            reachable by the given ``Butler``
        **kwargs : `dict`
            Keyword arguments passed onto `ButlerStandardizer`.

        Returns
        -------
        standardizer : `ButlerStandardizer`
            Standardizer
        """
        standardizer = Standardizer.get(standardizer="ButlerStandardizer")
        return standardizer(butler, id=id, config=config, **kwargs)

    @classmethod
    def fromDatasetIds(cls, butler, ids, config=None, **kwargs):
        """Get a list of `ButlerStandardizer`s for a collection of ``ids``."""
        return [cls.fromDatasetId(butler=butler, id=id, config=config, **kwargs)
                for id in ids]

    @classmethod
    @abc.abstractmethod
    def canStandardize(self, tgt):
        """Returns ``True`` when the standardizer knows how to process
        the given upload (file) type.

        Parameters
        ----------
        tgt : data-source
            Some data source, URL, URI, POSIX filepath, Rubin Data Butler Data
            Repository, an Python object, callable, etc.

        Returns
        -------
        canStandardize : `bool`
            `True` when the processor knows how to handle uploaded
            file and `False` otherwise.

        Notes
        -----
        Implementation is instrument-specific.
        """
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        # Registers subclasses of this class with set `name` class
        # parameters as availible standardizers. Note the specific
        # standardizer has to be imported before this class since the
        # mechanism is triggered at definition time.
        name = getattr(cls, "name", False)
        if name and name is not None:
            super().__init_subclass__(**kwargs)
            Standardizer.registry[cls.name] = cls

    @abc.abstractmethod
    def __init__(self, location, config=None, **kwargs):
        self.location = location
        self.processable = []
        self.config = self.configClass(config)

    def __str__(self):
        return f"{self.name}({self.location}, {self.processable})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.location})"

    # all these should probably be named in plural - but lord allmighty that's
    # a lot of references to update and it feels so unnatural - help?
    @abc.abstractproperty
    def wcs(self):
        """A list of WCS's or `None` for each entry marked as processable.

        Expected to be an WCS object, but when it is not possible to construct
        one `None` will be returned in its place.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def bbox(self):
        """A list of bounding boxes or `None` for each entry marked as
        processable.

        ========== ============================================================
        Key        Description
        ========== ============================================================
        center_ra  On-sky (ICRS) right ascension coordinate, expressed in
                   decimal degrees, of the center of an rectangular bounding
                   box.
        center_dec On-sky (ICRS) declination coordinate, expressed in decimal
                   degrees, of the center of an rectangular bounding box.
        corner_ra  On-sky (ICRS) right ascension coordinate, expressed in
                   decimal degrees, of an corner of a rectangular bounding box.
        corner_dec On-sky (ICRS) declination coordinate, expressed in decimal
                   degrees, of an corner of a rectangular bounding box.
        ========== ============================================================

        Often, when WCS can not be constructed, the BBOX can not be
        constructed. Then `None` is expected to be returned in it's place.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeWCS(self):
        """Creates an WCS for every processable science image.

        Returns
        -------
        wcs : `list`
            List of `~astropy.wcs.WCS` objects.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeBBox(self):
        """Calculate the standardized bounding box, the world coordinates at
        image corner and image center.

        Returns
        -------
        standardizedBBox : `dict`
            Calculated coordinate values, a dict with, ``wcs_center_[ra, dec]``
            and ``wcs_corner_[ra, dec]`` keys.

        Notes
        -----
        Center can be assumed to be at the (dimX/2, dimY/2) pixel coordinates,
        rounded down. Corner is taken to be the (0,0)-th pixel.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeMetadata(self):
        """Standardizes required and optional metadata from the given data.

        The standardized metadata is expected to be a dict containing the
        following keys and values:

        ======== ==============================================================
        Key      Description
        ======== ==============================================================
        location Path, URL or URI to the data.
        mjd      Decimal MJD timestamp of the start of the observation
        ra       Right Ascension in ICRS coordinate system of the extracted, or
                 calculated on-sky poisiton of the central pixel, pointing
                 data.
        dec      Declination in ICRS coordinate system, expressed in decimal
                 degrees, of the extracted, or calculated, on-sky poisiton of
                 the data.
        wcs      Result of `~standardizeWCS`, a list of `~astropy.wcs.WCS`
                 objects or a list of `None`s if they can not be constructed.
        bbox     Result of `standardizeBBox`, a list of standardized bounding
                 boxes or a list of `None`s if they can not be calculated.
        ======== ==============================================================

        Returns
        -------
        metadata : `dict`
            Standardized metadata.

        Notes
        -----
        If a target being standardized contains only 1 image that can be
        standardized and processed by KBMOD, the values in the dictionary can
        be strings, integers, floats, and other such single-values non-iterable
        types, or an iterable type containing only 1 element.
        When the target being standardized contains multiple images/extensions
        that are processable by KBMOD the single-valued dict entries will be
        considered as shared by all the processable images/extension/units of
        the target. The iterable values in the dictionary must match the number
        and order of the units marked as processable.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeScienceImage(self):
        """Standardizes the science image data.

        For FITS files, this is usually a trivial no-op operation that returns
        the `data` attribute of the correct HDU. For different data sources
        this is a type casting, or a download operation.

        Returns
        -------
        image : `list[~np.array]`
            Science images.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeVarianceImage(self):
        """Standardizes the variance data.

        For FITS files, this is sometimes a trivial no-op operation returning
        the correct FITS extension. In other cases, this has to be calculated
        if sufficient information is provided in the header or a different
        file needs to be downloaded/read.

        .. note::
           Refer to the manual for the originating dataset, whether the
           instrument or processing pipeline reference, as some variance planes
           store the inverse of variance.

        Returns
        -------
        variance : `list[`np.array]`
            Variance images.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeMaskImage(self):
        """Standardizes the mask data as an simple 0 (not masked) and 1
        (masked) bitmap.

        For FITS files, this is sometimes a trivial no-op operation returning
        the correct FITS extension. In other cases, the mask has to be
        constructed from external data, such as pixel masks and catalogs,
        or downloaded/read from a different file at the data source.

        Returns
        -------
        mask : `list[~np.array]`
            Mask images.
        """
        raise NotImplementedError()

    # no idea really what to do bout this one? AFAIK almost no images come
    # with PSFs in them
    @abc.abstractmethod
    def standardizePSF(self):
        """Returns PSF for each extension marked as processable.

        Returns
        -------
        psfs : `list[~kbmod.search.psf]`
            List of `~kbmod.search.psf` objects.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def toLayeredImage(self):
        """Run metadata standardization methods. These include header
        and bounding box standardization.

        Notes
        -----
        Implementation is standardizer-specific.
        """
        raise NotImplementedError()

    def standardize(self):
        """Invokes standardize metadata, image, variance, mask and PSF and
        returns the results as a dictionary.

        Returns
        -------
        standardizedData : `dict`
            Dictionary with standardized ``meta``, ``science``, ``variance``,
            ``mask`` and ``psf`` values.
        """
        std = {"meta": self.standardizeMetadata()}
        std["science"] = self.standardizeScienceImage()
        std["variance"] = self.standardizeVarianceImage()
        std["mask"] = self.standardizeMaskImage()
        std["psf"] = self.standardizePSF()

        return std
