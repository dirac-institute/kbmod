"""Classes for working with the input files for KBMOD.

The ``ImageCollection`` class stores additional information for the
input FITS files that is used during a variety of analysis.
"""

import logging
import os
import glob
import json
import warnings

from astropy.table import Table, Column, vstack
from astropy.wcs import WCS
from astropy.utils import isiterable

import numpy as np

from kbmod.search import ImageStack
from .standardizers import Standardizer
from .work_unit import WorkUnit


__all__ = [
    "ImageCollection",
]


logger = logging.getLogger(__name__)


class ImageCollection:
    """A collection of basic pointing, file paths, names, timestamps and other
    metadata that facilitate, and make easier the construction of ImageStack
    and execution of KBMOD search.

    It is recommended to construct this object by using one of its factory
    methods, avoiding instantiating the object directly. When constructed by
    using one of the ``from*`` methods the class interprets the data source
    formats and determines the appropriate way of extracting, at least, the
    required metadata from it. This behaviour can be modified by supplying a
    callable as the `forceStandardizer` argument to the factory method. The
    provided callable has to be an instance of `Standardizer`. See the factory
    method's and `Standardizer` documentation for more information.

    Columns listed in the `ImageCollection.required_metadata` are required to
    exist and be non-zero. Additional columns may or may not exist. The names
    of the standardizers used to extract the metadata are also guaranteed to
    exist. The `Standardizer` objects may or may not be availible, depending on
    whether the data is accessible by the user. The same, by extension, applies
    to properties such as bbox and WCS.
    Attributes which values may or may not be availible have their own getter
    methods (f.e. `get_standardizer`, `get_wcs` etc.).

    Some standardizers, such as `MultiExtensionfits`, map a single location and
    timestamp to potentially many individual extensions each with a slightly
    different on-sky location. Therefore the list of the standardizers does
    not neccessarily match the length of the metadata table. Metadata table is
    the metadata per extension, unravelled into a flat table.

    Parameters
    ----------
    metadata : `~astropy.table.Table`
        A table of exposure metadata properties
    serializers : `list`
        A list of valid serializer names, used to extract the metadata in its
        row.

    Attributes
    ----------
    data : `~astropy.table.Table`
        Table of exposure metadata properties.
    standardizers : `list`
        Standardizer names used to standardize the rows of the metadata.
        The names are guaranteed to exist, but the standardizer object may not
        be availible.
    wcs : `list`
        List of `~astropy.wcs.WCS` objects that correspond to the rows
        of the metadadata.
    bbox : `list`
        List of `dict` objects containing the pixel and on-sky coordinates of
        the central and corner (0, 0) pixel.

    Raises
    ------
    ValueError :
        when instantiated from a Table which does not have the required
        columns, or has null-values in the required columns.
    """

    # Both are required, but supporting metadata is mostly handled internally
    required_metadata = ["mjd", "ra", "dec", "wcs"]
    _supporting_metadata = ["std_name", "std_idx", "ext_idx", "config"]

    ########################
    # CONSTRUCTORS
    ########################
    def _validate(self, metadata):
        """Validates the required metadata exist and is not-null.

        Required metadata is the ``location`` of the target, ``mjd``, ``ra``,
        ``dec`` and the standardizer-row lookup indices ``std_idx`` and
        ``ext_idx``.

        Parameters
        ----------
        metadata : `~astropy.table.Table`
            Astropy Table containing the required metadata.

        Returns
        -------
        valid : `bool`
            When ``True`` the metadata is valid, when ``False`` it isn't.
        message: `str`
            Validation failure message, if any.
        """
        if not isinstance(metadata, Table):
            return False, "not an Astropy Table object."

        # if empty table
        if not metadata:
            return False, "an emtpy table."

        cols = metadata.columns
        missing_keys = [key for key in self.required_metadata if key not in cols]
        if missing_keys:
            return False, f"missing required columns: {missing_keys}"

        # check that none of the actual required column entries are empty in
        # some way. perhaps we should be checking np.nan too?
        for rc in self.required_metadata:
            if None in metadata[rc] or "" in metadata[rc]:
                return False, "missing required metadata values."

        # check that standardizer to row lookup exists
        missing_keys = [key for key in self._supporting_metadata if key not in cols]
        if missing_keys:
            return False, (f"missing required standardizer-row lookup indices: {missing_keys}")

        return True, ""

    def __init__(self, metadata, standardizers=None, enable_lazy_loading=True):
        valid, explanation = self._validate(metadata)
        if valid:
            metadata.sort("mjd")
        else:
            raise ValueError(f"Metadata is {explanation}")

        # If standardizers are already instantiated, keep them. This keeps any
        # resources they are holding onto alive, and enables in-memory stds.
        # These are impossible to instantiate, but since they are already
        # in-memory we don't need to; and lazy-loading will skip attempts to.
        # If standardizers are not instantiated, figure out how many we have
        # from the metadata. If metadata doesn't say, guess how many there are.
        # If lazy loading is not enabled, assume they round-trip from row data.
        self._standardizers = None
        if standardizers is not None:
            self._standardizers = np.array(standardizers)
            metadata.meta["n_std"] = len(standardizers)
        else:
            n_stds = metadata.meta.get("n_stds", None)
            if n_stds is None:
                n_stds = metadata["std_idx"].max()
                self.data.meta["n_stds"] = n_stds

            if enable_lazy_loading:
                self._standardizers = np.full((n_stds,), None)

        self.data = metadata
        self._userColumns = [col for col in self.data.columns if col not in self._supporting_metadata]

    @classmethod
    def read(cls, *args, format="ascii.ecsv", units=None, descriptions=None, **kwargs):
        """Create ImageCollection from a file containing serialized image
        collection.

        Parameters
        ----------
        filepath : `str`
            Path to the file containing the serialized image collection.

        Returns
        -------
        ic : `ImageCollection`
            Image Collection
        """
        metadata = Table.read(*args, format=format, units=units, descriptions=descriptions, **kwargs)
        meta = json.loads(
            metadata.meta["comments"][0],
        )
        metadata.meta = meta
        return cls(metadata)

    @classmethod
    def fromStandardizers(cls, standardizers, meta=None):
        """Create ImageCollection from a collection `Standardizers`.

        The `Standardizer` is "unravelled", i.e. the shared metadata is
        duplicated for each entry marked as processable by the standardizer. On
        an practical example - MJD timestamps are shared by all 62 science
        exposures created by DECam in a single exposure, but pointing of each
        one isn't. So each pointing is a new row in the metadata table for each
        of the individual pointings.

        Parameters
        ----------
        standardizers : `iterable`
            Collection of `Standardizer` objects.

        Returns
        -------
        ic : `ImageCollection`
            Image Collection
        """
        logger.info(f"Creating ImageCollection from {len(standardizers)} standardizers.")

        unravelledStdMetadata = []
        for i, std in enumerate(standardizers):
            # needs a "validate standardized" method here or in standardizers
            stdMeta = std.standardizeMetadata()

            # unravel all standardized keys whose values are iterables unless
            # they are a string. "Unraveling" means that each processable item
            # of standardizer gets its own row and each non-iterable standardized
            # item is copied into that row. F.e. "a.fits" with 3 images becomes
            # has the same location which is then duplicated across rows:
            # location    std_vals
            #  a.fits     ...1
            #  a.fits     ...2
            #  a.fits     ...3
            unravelColumns = [
                key for key, val in stdMeta.items() if isiterable(val) and not isinstance(val, str)
            ]
            for j, ext in enumerate(std.processable):
                row = {}
                for key in stdMeta.keys():
                    if key in unravelColumns:
                        row[key] = stdMeta[key][j]
                    else:
                        row[key] = stdMeta[key]
                    row["std_idx"] = i
                    row["ext_idx"] = j
                    row["std_name"] = std.name

                # config and WCS are serialized in a more complicated way
                # than most literal values. Both are stringified dicts, but
                # WCS must construct its metadata as a header object before it
                # can be serialized. Its important to save every character here
                row["config"] = json.dumps(std.config.toDict(), separators=(",", ":"))

                header = std.wcs[j].to_header(relax=True)
                h, w = std.wcs[j].pixel_shape
                header["NAXIS1"] = h
                header["NAXIS2"] = w
                header_dict = {k: v for k, v in header.items()}
                row["wcs"] = json.dumps(header_dict, separators=(",", ":"))
                unravelledStdMetadata.append(row)

        # We could even track things like `whoami`, `uname`, `time` etc.
        meta = meta if meta is not None else {}
        meta["n_stds"] = len(standardizers)
        metadata = Table(rows=unravelledStdMetadata, meta=meta)
        return cls(metadata=metadata, standardizers=standardizers)

    @classmethod
    def fromTargets(cls, tgts, force=None, config=None, **kwargs):
        """Instantiate a ImageCollection class from a collection of targets
        recognized by the standardizers, for example file paths, integer id,
        dataset reference objects etc.

        Parameters
        ----------
        locations : `str` or `iterable`
            Collection of file-paths, a path to a directory, pathor URIs, to
            FITS files or a path to a directory of FITS files or a butler
            repository.
        recursive : `bool`
            If the location is a local filesystem directory, scan it
            recursively including all sub-directories.
        forceStandardizer : `Standardizer` or `None`
            If `None`, when applicable, determine the correct `Standardizer` to
            use automatically. Otherwise force the use of the given
            `Standardizer`.
        **kwargs : `dict`
            Remaining kwargs, not listed here, are passed onwards to
            the underlying `Standardizer`.

        Raises
        ------
        ValueError:
            when location is not recognized as a file, directory or an URI
        """
        standardizers = [Standardizer.get(tgt, force=force, config=config, **kwargs) for tgt in tgts]
        return cls.fromStandardizers(standardizers)

    @classmethod
    def fromDir(cls, dirpath, recursive=False, force=None, config=None, **kwargs):
        """Instantiate ImageInfoSet from a path to a directory
        containing FITS files.

        Parameters
        ----------
        locations : `str` or `iterable`
            Collection of file-paths, a path to a directory, pathor URIs, to
            FITS files or a path to a directory of FITS files or a butler
            repository.
        recursive : `bool`
            If the location is a local filesystem directory, scan it
            recursively including all sub-directories.
        forceStandardizer : `Standardizer` or `None`
            If `None`, when applicable, determine the correct `Standardizer` to
            use automatically. Otherwise force the use of the given
            `Standardizer`.
        **kwargs : `dict`
            Remaining kwargs, not listed here, are passed onwards to
            the underlying `Standardizer`.
        """
        logger.debug(f"Building ImageCollection from FITS filtes in: {dirpath}")
        fits_files = glob.glob(os.path.join(dirpath, "*fits*"), recursive=recursive)
        logger.debug(f"Found {len(fits_files)} matching files:\n{fits_files}")
        return cls.fromTargets(fits_files, force=force, config=config, **kwargs)

    ########################
    # PROPERTIES (type operations and invariants)
    ########################
    def __str__(self):
        return str(self.data[self._userColumns])

    def __repr__(self):
        return repr(self.data).replace("Table", "ImageCollection")

    # Jupyter notebook hook for rendering output as HTML
    def _repr_html_(self):
        return self.data[self._userColumns]._repr_html_().replace("Table", "ImageCollection")

    def __getitem__(self, key):
        if isinstance(key, (int, str, np.integer)):
            return self.data[self._userColumns][key]
        elif isinstance(key, (list, np.ndarray, slice)):
            # current data table has standardizer idxs with respect to current
            # list of standardizers. Sub-selecting them resets the count to 0
            meta = self.data[key]
            stds = [self._standardizers[idx] for idx in meta["std_idx"]]
            meta["std_idx"] = np.arange(len(stds))
            return self.__class__(meta, standardizers=stds)
        else:
            return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        # start with cheap comparisons
        if not isinstance(other, ImageCollection):
            return False

        if not self.meta == other.meta:
            return False

        if not self.columns.keys() == other.columns.keys():
            return False

        # before we compare the entire tables (minus WCS, not comparable)
        cols = [col for col in self.columns if col != "wcs"]
        return (self.data[cols] == other.data[cols]).all()

    @property
    def meta(self):
        return self.data.meta

    @property
    def wcs(self):
        for i in range(len(self.data)):
            # the warnings that some keywords might be ignored are expected
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield WCS(json.loads(self.data["wcs"][i]), relax=True)

    def get_wcs(self, idxs):
        # select column before indices, because a copy of the data
        # will be made, same for bbox. It pays off not to copy the
        # whole row nearly every-time
        selected = self.data["wcs"][idxs]
        # the warnings that some keywords might be ignored are expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(selected, Column):
                return [WCS(json.loads(row), relax=True) for row in selected]
            return WCS(json.loads(selected), relax=True)

    @property
    def bbox(self):
        # what we return here depends on what region search needs
        # best probably to return a BBox dataclass that has some useful
        # functionality for region search or something, maybe bbox
        # even needs a timestamp or something...
        cols = ["ra", "dec", "ra_tl", "dec_tl", "ra_tr", "dec_tr", "ra_bl", "dec_bl", "ra_br", "dec_br"]
        for i in range(len(self.data)):
            yield self.data[cols][i]

    def get_bbox(self, idxs):
        # again, we can return an BBox collection object with additional methods
        cols = ["ra", "dec", "ra_tl", "dec_tl", "ra_tr", "dec_tr", "ra_bl", "dec_bl", "ra_br", "dec_br"]
        selected = self.data[cols][idxs]
        return selected

    @property
    def columns(self):
        """Return metadata columns."""
        return self.data[self._userColumns].columns

    @property
    def standardizers(self):
        """Standardizer generator."""
        for i in range(len(self.data)):
            yield self.get_standardizer(i)

    def get_standardizer(self, index, **kwargs):
        """Get the standardizer and extension index for the selected row of the
        unravelled metadata table.

        A helper function that allows for access to non-required common
        properties such as standardizer object, WCS, bounding boxes etc.

        Parameters
        ----------
        index : `int`
            Index, as it appears in the unravelled table of metadata
            properties.
        **kwargs : `dict`
            Keyword arguments are passed onto the Standardizer constructor.

        Returns
        -------
        std : `dict`
            A dictionary containing the standardizer (``std``) and the
            extension (``ext``) that maps to the given metadata row index.
        """
        row = self.data[index]
        std_idx = row["std_idx"]

        def load_std():
            # we want the row, because rows have to contain all values required
            # to init a standardizer, but std config is written in that row as
            # just a string and we want a dict. Pluck it out, make a dict.
            std_cls = Standardizer.registry[row["std_name"]]
            no_conf_cols = list(self.data.columns.keys())
            no_conf_cols.remove("config")
            config = json.loads(row["config"])
            return std_cls(**kwargs, **row[no_conf_cols], config=config)

        # I don't think a 65k long standardizer list will work. But if a list
        # of _standardizers exists, then we can do to lazy loading, since the
        # implication is, it isn't too long. Keep in mind some standardizers
        # keep their resources alive, including images - which can be memory
        # intensive.
        if self._standardizers is None:
            # no lazy loading
            std = load_std()
        elif self._standardizers[index] is None:
            # lazy load and store
            std = load_std()
            self._standardizers[std_idx] = std
        else:
            # already loaded
            std = self._standardizers[std_idx]

        # maybe a clever dataclass to shortcut the idx lookups on the user end?
        return {"std": std, "ext": self.data[index]["ext_idx"]}

    def get_standardizers(self, idxs, **kwargs):
        """Get the standardizers used to extract metadata of the selected
        rows.

        Parameters
        ----------
        idxs : `int` or `iterable`
            Index of the row for which to retrieve the Standardizer.
        **kwargs : `dict`
            Keyword arguments are passed onto the constructors of the retrieved
            Standardizer.

        Returns
        -------
        std : `list``
            A list of dictionaries containing the standardizer (``std``) and
            the extension (``ext``) that maps to the given metadata row index.
        """
        if isinstance(idxs, int):
            return [
                self.get_standardizer(idxs, **kwargs),
            ]
        else:
            return [self.get_standardizer(idx, **kwargs) for idx in idxs]

    ########################
    # FUNCTIONALITY (object operations, transformative functionality)
    ########################
    def write(self, *args, format="ascii.ecsv", serialize_method=None, **kwargs):
        """Write the ImageCollection to a file or file-like object.

        A light wrapper around the underlying AstroPy's Table ``write``
        functionality. See `astropy/io.ascii.write`
        `documentation <https://docs.astropy.org/en/stable/io/ascii/write.html#parameters-for-write>`_
        """
        logger.info(f"Writing ImageCollection to {args[0]}")
        tmpdata = self.data.copy()

        # some formats do not officially support comments, like CSV, others
        # have no problems with comments, some provide a workaround like
        # dumping only meta["comment"] section if comment="#" kwarg is given.
        # Because of these inconsistencies we'll just package everything into
        # "comments" tag and then unpack at read time.
        stringified = json.dumps(tmpdata.meta)
        current_comments = tmpdata.meta.get("comments", None)
        if current_comments is not None:
            tmpdata.meta = {}
        tmpdata.meta["comments"] = [
            stringified,
        ]

        tmpdata.write(*args, format=format, serialize_method=serialize_method, **kwargs)

    def get_zero_shifted_times(self):
        """Returns a list of timestamps such that the first image
        is at time 0.

        Returns
        -------
        List of floats
            A list of zero-shifted times (JD or MJD).
        """
        # The images do not have to be sorted, but we treat the first
        # image as timestep 0.
        return self.data["mjd"] - self.data["mjd"][0]

    def toImageStack(self):
        """Return an `~kbmod.search.image_stack` object for processing with
        KBMOD.
        Returns
        -------
        imageStack : `~kbmod.search.image_stack`
            Image stack for processing with KBMOD.
        """
        logger.info("Building ImageStack from ImageCollection")
        layeredImages = [img for std in self._standardizers for img in std.toLayeredImage()]
        return ImageStack(layeredImages)

    def toWorkUnit(self, config=None):
        """Return an `~kbmod.WorkUnit` object for processing with
        KBMOD.

        Parameters
        ----------
        config : `~kbmod.SearchConfiguration` or None, optional
            Search configuration. Default ``None``.

        Returns
        -------
        work_unit : `~kbmod.WorkUnit`
            A `~kbmod.WorkUnit` object for processing with KBMOD.
        """
        image_locations = [str(s) for s in self.data["location"].data]
        logger.info("Building WorkUnit from ImageCollection")
        layeredImages = [img for std in self.standardizers for img in std["std"].toLayeredImage()]
        imgstack = ImageStack(layeredImages)
        if None not in self.wcs:
            return WorkUnit(imgstack, config, constituent_images=image_locations, per_image_wcs=self.wcs)
        return WorkUnit(imgstack, config, constituent_images=image_locations)
