import abc


__all__ = ["Standardizer",]


class Standardizer(abc.ABC):
    """Supports standardization of data targeted by KBMOD interface.

    Parameters
    ----------
    location : `str`
        Location of the file to standardize, a filepath or URI.
    
    Attributes
    ----------
    loc : `str`
        Location of the file being standardized.
    """
    
    standardizers = dict()
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
            Standardizer.standardizers[cls.name] = cls

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
        for standardizer in cls.standardizers.values():
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
                # I think this should never be an issue really, but just in case
                names = [std.name for std in standardizers]
                #logger.info("Multiple standardizers declared ability to process "
                #            f"the given upload: {names}. Using {names[0]} "
                #            "to process FITS.")
            return standardizers[0]
        else:
            raise ValueError("None of the known standardizers can handle this source.\n "
                             f"Known standardizers: {list(cls.standardizers.keys())}")

    @classmethod
    def fromFile(cls, path, **kwargs):
        """Return a single Standardizer that can standardize the given
        metadata location.

        Parameters
        ----------
        location : `str`
            Source of the metadata to standardize.
        
        Returns
        -------
        standardizer : `object`
            Standardizer that can process the given source.

        Raises
        ------
        ValueError
            None of the registered processors can process the upload.
        """
        processorCls = cls.getStandardizer(path)
        return processorCls(path, **kwargs)

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
    def standardizeHeader(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeVariance(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeMask(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def standardizeBBox(self):
        raise NotImplementedError()
    
    # no idea really what to do about this one?
    # AFAIK almost no images come with PSFs in them
    #@abc.abstractmethod
    #def standardizePSF(self):
    #    raise NotImplementedError()

    def standardizeMetadata(self):
        """Run metadata standardization methods. These include header
        and bounding box standardization.

        Notes
        -----
        Implementation is processor-specific.
        """
        metadata = self.standardizeHeader()
        metadata["bbox"] = self.standardizeBBox()
        
        return metadata

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
