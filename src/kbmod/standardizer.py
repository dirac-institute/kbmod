import abc


__all__ = ["Standardizer",]


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
    the exposures of interest are grouped together and furhter results analysis.
    KBMOD does not insist on providing them however, requiring instead just a
    more general "pointing" information. For these reason the WCS and BBOX have
    their own standardize methods, and should be provided whenever possible.

    Data is expected to be unravelled on a per-science exposure level. Data
    that shares some metadata, such as timestamp or filter for example, for
    multiple science exposures is expected to unravel that metadata and
    associate it with each of the science exposures individualy.

    This class is an abstract base class to serve as a recipe for implementing
    a Standardizer specialized for processing an particular dataset. Do not
    attempt to instantiate this class directly.

    Parameters
    ----------
    location : `str`
        Location of the file to standardize, a filepath or URI.

    Attributes
    ----------
    loc : `str`
        Location of the file being standardized.
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

    @abc.abstractmethod
    def __init__(self, location, *args, **kwargs):
        self.location = location

    def __init_subclass__(cls, **kwargs):
        # Registers subclasses of this class with set `name` class
        # parameters as availible standardizers. Note the specific
        # standardizer has to be imported before this class since the
        # mechanism is triggered at definition time.
        name = getattr(cls, "name", False)
        if name and name is not None:
            super().__init_subclass__(**kwargs)
            Standardizer.registry[cls.name] = cls

    def __str__(self):
        return f"{self.name}({self.location}, {self.exts})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.location})"

    @classmethod
    def getStandardizer(cls, location):
        """Get the standardizer class that can handle given file. If
        multiple standardizers declare the ability to process the
        given file the standardizer with highest prirority is
        selected.

        Parameters
        ----------
        location : `str`
            Source of the metadata to standardize.

        Returns
        -------
        standardizer : `cls`
            Standardizer class that can process the given upload.
        """
        standardizers = []
        for standardizer in cls.registry.values():
            # the rule of thumb here is that the first element has to
            # be canStd, but more stuff can be returned for
            # optimization purposes, here we only care about the first
            # value so we throw away the rest
            canStandardize, *_ = standardizer.canStandardize(location)
            if canStandardize:
                standardizers.append(standardizer)

        def get_priority(standardizer):
            """Return standardizers priority."""
            return standardizer.priority
        standardizers.sort(key=get_priority, reverse=True)

        if standardizers:
            if len(standardizers) > 1:
                names = [std.name for std in standardizers]
                # this should never be an issue, but just in case
                #logger.info("Multiple standardizers declared ability to process "
                #            f"the given upload: {names}. Using {names[0]} "
                #            "to process FITS.")
            return standardizers[0]
        else:
            raise ValueError("None of the known standardizers can handle this source.\n "
                             f"Known standardizers: {list(cls.registry.keys())}")

    @classmethod
    def fromFile(cls, path, forceStandardizer=None, **kwargs):
        """Return a single Standardizer that can standardize the given
        metadata location.

        Parameters
        ----------
        location : `str`
            Source of the metadata to standardize.
        forceStandardizer : `callable` or `None`
            Standardizer class to use when mapping the file content. When
            ``False` standardizer will automatically be determined from the
            provided file.
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
        """
        standardizerCls = forceStandardizer if forceStandardizer else cls.getStandardizer(path)
        return standardizerCls(path, **kwargs)

    @classmethod
    @abc.abstractmethod
    def canStandardize(self, tgt):
        """Returns ``True`` when the standardizer knows how to process
        the given upload (file) type.

        Parameters
        ----------
        tgt : anything
            Say we expect only HDULists from imageinfoset but really
            can be more generic than that - I just can't come up with
            something right now.

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
            Calculated coorinate values, a dict with, ``wcs_center_[ra, dec]``
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
        following keys in data:

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
                 boxes or a list of `None`s if they can not be calcualted.
        ======== ==============================================================

        Returns
        -------
        metadata : `dict`
            Standardized metadata.
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
        image : `np.array`
            Science image.
        """
        raise NotImplemented()

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
        variance : `np.array`
            Variance image.
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
        mask : `np.array`
            Mask image.
        """
        raise NotImplementedError()

    # no idea really what to do bout this one? AFAIK almost no images come
    # with PSFs in them
    #@abc.abstractmethod
    #def standardizePSF(self):
    #    raise NotImplementedError()

    def standardize(self):
        """Invokes standardize metadata, image, variance, mask and PSF and
        returns the results as a dictionary.

        Returns
        -------
        standardizedData : `dict`
            Dictionary with standardized ``meta``, ``science``, ``variance``,
            ``mask`` and ``psf`` values.
        """
        std = {"meta": self.standardizeStddata()}
        std.update({"science": self.standardizeScienceImage()})
        std.update({"variance": self.standardizeVarianceImage()})
        std.update({"mask": self.standardizeMaskImage()})
        std.update({"psf": self.standardizePSF()})

        return std

    def toLayeredImage(self):
        """Run metadata standardization methods. These include header
        and bounding box standardization.

        Notes
        -----
        Implementation is processor-specific.
        """
        # note that these will be lists so we need a new constructor
        # They will be equal length(maybe?) and each is a different
        # detector - i.e. not an image stack. Because we don't use
        # ndarray this will be a copy operation and not very fast.
        header = self.standardizeHeader()
        masks = self.standardizeMask()
        variances = self.standardizeVariance()
        #psfs = self.standardizePSF()

        return LayeredImage(self.loc, self.exts, masks, variances, psfs, header["mjd"])
