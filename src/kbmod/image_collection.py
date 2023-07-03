"""Classes for working with the input files for KBMOD.

The ``ImageInfo`` class stores additional information for the
input FITS files that is used during a variety of analysis.
"""
import os
import glob
import json

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS
from astropy.utils import isiterable

import numpy as np

from kbmod.file_utils import FileUtils
from kbmod.search import pixel_pos, layered_image
from kbmod.standardizer import Standardizer


class ImageCollection:
    """A collection of basic pointing, file paths, names, timestamps and other
    metadata that facilitate, and make easier the construction of ImageStack
    and execution of KBMOD search.

    It is recommended to construct this object by using one of its factory
    methods, avoiding instantiating the object directly. When constructed by
    using one of the ``from*`` methods the class interprets the data source
    formats and determines the appropriate way of extracting, at least, the
    required metadata from it. This behaviour can be modified by supplying a
    callable as the `forceStandardizer` argument to the facotry method. The
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
    required_metadata = ["location", "mjd", "ra", "dec"]
    _supporting_metadata = ["std_name", "std_idx", "ext_idx", "wcs", "bbox"]

    ########################
    # CONSTRUCTORS
    ########################
    def _validate(self, metadata):
        """Validates the required metadata exist and is not-null.

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

        return True, ""

    def __init__(self, metadata, standardizers=None):
        valid, explanation = self._validate(metadata)
        if valid:
            self.data = metadata
        else:
            raise ValueError(f"Metadata is {explanation}")

        if "std_name" in metadata.columns:
            self._standardizer_names = metadata["std_name"]
        elif standardizers is not None:
            self._standardizer_names = [std.name for std in standardizers]
            self.data["std_names"] = self._standardizer_names

        if standardizers is not None:
            self.data.meta["n_entries"] = len(standardizers)
            self._standardizers = standardizers
        elif metadata.meta and "n_entries" in metadata.meta:
            n_entries = metadata.meta["n_entries"]
            self._standardizers = [None]*n_entries
        else:
            n_entries = len(np.unique(metadata["location"]))
            self.data.meta["n_entries"] = n_entries
            self._standardizers = [None]*n_entries

        # hidden indices that track the unravelled lookup to standardizer
        # extension index. I should imagine there's a better than double-loop
        # solution and a flat lookup table.
        # If they weren't adde when standardizers were unravelled it's
        # basically not possible to reconstruct them. Guess attempt is no good?
        no_std_map = False
        if "std_idx" not in metadata.columns:
            no_std_map = True
            self.data["std_idx"] = [None]*len(self.data)

        if "ext_idx" not in metadata.columns:
            no_std_map = True
            self.data["ext_idx"] = [None]*len(self.data)

        if standardizers and no_std_map:
            std_idxs, ext_idxs = [], []
            for i, stdFits in enumerate(standardizers):
                for j, ext in enumerate(stdFits.exts):
                    std_idxs.append(i)
                    ext_idxs.append(j)
            self.data["std_idx"] = std_idxs
            self.data["ext_idx"] = ext_idxs

        self._userColumns = [
            col for col in self.data.columns
            if col not in self._supporting_metadata
        ]

    @classmethod
    def read(cls, *args, format=None, units=None, descriptions=None, **kwargs):
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
        metadata = Table.read(*args, format=format, units=units,
                              descriptions=descriptions, **kwargs)
        metadata["wcs"] = [WCS(w) for w in metadata["wcs"] if w is not None]
        metadata["bbox"] = [json.loads(b) for b in metadata["bbox"]]
        meta = json.loads(metadata.meta["comments"][0],)
        metadata.meta = meta
        return cls(metadata)

    @classmethod
    def _fromStandardizers(cls, standardizers, meta=None):
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
        unravelledStdMetadata = []
        for i, stdFits in enumerate(standardizers):
            # needs a "validate standardized" method here or in standardizers
            stdMeta = stdFits.standardizeMetadata()
            # how can we tell what we need to unravel? (ra, dec, wcs, bbox) but
            # sometimes mjd and other keys too. See comment in
            # ButlerStd.stdMeta. Everything that is an iterable, except for a
            # string because that could be a location key?
            unravelColumns = [key for key, val in stdMeta.items() if isiterable(val) and not isinstance(val, str)]
            for j, ext in enumerate(stdFits.exts):
                row = {}
                for key in stdMeta.keys():
                    if key in unravelColumns:
                        row[key] = stdMeta[key][j]
                    else:
                        row[key] = stdMeta[key]
                    row["std_idx"] = i
                    row["ext_idx"] = j
                    row["std_name"] = stdFits.name
                unravelledStdMetadata.append(row)

        # We could even track things like `whoami`, `uname` etc. as a metadata
        # to the imagecollection in order to truly pinpoint where the data
        # came from. For now, this is more of a test.
        meta = meta if meta is not None else {"source": "fromStandardizers",
                                              "n_entries": len(standardizers)}
        metadata = Table(rows=unravelledStdMetadata, meta=meta)
        return cls(metadata=metadata, standardizers=standardizers)

    @classmethod
    def _fromFilepaths(cls, filepaths, forceStandardizer, **kwargs):
        """Create ImageCollection from a collection of local system
        filepaths to FITS files.

        Parameters
        ----------
        filepaths : `iterable`
            Collection of paths to fits files.

        Returns
        -------
        ic : `ImageCollection`
            Image Collection
        """
        standardizers = [
            Standardizer.fromFile(path=path, forceStandardizer=forceStandardizer, **kwargs)
            for path in filepaths
        ]
        return cls._fromStandardizers(standardizers)

    @classmethod
    def _fromDir(cls, path, recursive, forceStandardizer, **kwargs):
        """Instantiate ImageInfoSet from a path to a directory
        containing FITS files.

        Parameters
        ----------
        path : `str`
            Path to directory containing fits files.

        Returns
        -------
        ic : `ImageCollection`
            Image Collection
        """
        # imagine only dir of FITS files
        fits_files = glob.glob(os.path.join(path, "*fits*"), recursive=recursive)
        return cls._fromFilepaths(filepaths=fits_files, forceStandardizer=forceStandardizer, **kwargs)

    @classmethod
    def fromLocations(cls, locations, recursive=False, forceStandardizer=None,
                      **kwargs):
        """Instantiate a ImageInfoSet class from a collection of system
        file paths, URLs, URIs to FITS files or a path to a system directory
        containing FITS files.

        .. warning::

           Currently supports only a list of local POSIX compliant filesystem
           paths or a path to a directory.

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
        # somewhere here we also need a check for an URI schema, or punt the
        # whole thing to lsst.resources. Then we wouldn't need all the hidden
        # _fromURI/DIR/FLOCATION constructors anymore (hopefully)
        if os.path.isdir(locations):
            return cls._fromDir(locations, recursive=recursive,
                                forceStandardizer=forceStandardizer, **kwargs)

        if os.path.isfile(locations):
            return cls._fromFilepaths([locations,], forceStandardizer=forceStandardizer,
                                      **kwargs)
        elif isiterable(locations) and all([os.path.isfile(p) for p in locations]):
            return cls._fromFilepaths(locations, recursive=recursive,
                                      forceStandardizer=forceStandardizer,
                                      **kwargs)

        raise ValueError(f"Unrecognized local filesystem path: {locations}")

    @classmethod
    def fromDatasetRefs(cls, butler, refs, **kwargs):
        """Construct an `ImageCollection` from an instantiated butler and a
        collection of ``DatasetRef``s.

        Parameters
        ----------
        butler : `~lsst.daf.butler.Butler`
            Vera C. Rubin Data Butler.
        refs : `list`
            List of `~lsst.daf.butler.core.DatasetRef` objects.
        **kwargs : `dict`
            Keyword arguments passed onto `ButlerStandardizer`.

        Returns
        -------
        ic : `ImageCollection`
            Image collection.
        """
        standardizer_cls = Standardizer.get(standardizer="ButlerStandardizer")
        standardizer = standardizer_cls(butler, refs, **kwargs)
        meta = {"root": butler.datastore.root.geturl(), "n_entries": len(list(refs))}
        return cls._fromStandardizers([standardizer, ], meta=meta)

    def fromAQueryTable(self):  # ? TBD
        pass

    ########################
    # PROPERTIES (type operations and invariants)
    ########################
    def __str__(self):
        return str(self.data[self._userColumns])

    def __repr__(self):
        return repr(self.data).replace("Table", "ImageInfoSet")

    def __getitem__(self, key):
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
        return self.data[cols] == other.data[cols]

    @property
    def meta(self):
        return self.data.meta

    @property
    def wcs(self):
        return self.data["wcs"].data

    @property
    def bbox(self):
        return self.data["bbox"].data

    @property
    def columns(self):
        """Return metadata columns."""
        return self.data[self._userColumns].columns

    @property
    def standardizers(self):
        """A list of used standardizer names."""
        return self._standardizer_names

    def _get_standardizer(self, index):
        """Get the standardizer and extension index for the selected row of the
        unravelled metadata table.

        A helper function that allows for access to non-required common
        properties such as standardizer object, WCS, bounding boxes etc.

        Parameter
        --------
        index : `int`
            Index, as it appears in the unravelled table of metadata
            properties.

        Returns
        -------
        std : `dict`
            A dictionary containing the standardizer (``std``) and the
            extension (``ext``) that maps to the given metadata row index.
        """
        row = self.data[index]
        std_idx = row["std_idx"]
        ext_idx = row["ext_idx"]
        if self._standardizers[std_idx] is None:
            std_cls = Standardizer.registry[row["std_name"]]
            self._standardizers[std_idx] = std_cls(row["location"])

        # maybe a clever dataclass to shortcut the idx lookups on the user end?
        return {"std": self._standardizers[std_idx],
                "ext": self.data[index]["ext_idx"]}

    def get_standardizers(self, idxs):
        """ Get the standardizers used to extract metadata of the selected
        rows.

        Parameters
        ----------
        idx : `int` or `iterable`
            Index of the row for which to retrieve the Standardizer.

        Returns
        -------
        std : `list``
            A list of dictionaries containing the standardizer (``std``) and
            the extension (``ext``) that maps to the given metadata row index.
        """
        if isinstance(idxs, int):
            return self._get_standardizer(idxs)
        else:
            return [self._get_standardizer(idx) for idx in idxs]

    ########################
    # FUNCTIONALITY (object operations, transformative functionality)
    ########################
    def toImageStack(self):
        pass

    def plot_onsky(self):
        pass

    def write(self, *args, format=None, serialize_method=None, **kwargs):
        tmpdata = self.data.copy()

        wcs = [w.to_header_string(relax=True) for w in self.wcs]
        tmpdata["wcs"] = wcs

        bbox = [json.dumps(b) for b in self.bbox]
        tmpdata["bbox"] = bbox

        # some formats do not officially support comments, like CSV, others
        # have no problems with comments, some provide a workaround like
        # dumping only meta["comment"] section if comment="#" kwarg is given.
        # Because of these inconsistencies we'll just package everything and
        # unpack at read. Comments are not not expected to be complicated
        # structures
        stringified = json.dumps(tmpdata.meta)
        current_comments = tmpdata.meta.get("comments", None)
        if current_comments is not None:
            tmdata.meta = {}
        tmpdata.meta["comments"] = [stringified,]

        tmpdata.write(*args, format=format, serialize_method=serialize_method, **kwargs)

    def get_zero_shifted_times(self):
        """Returns a list of timestamps such that the first image
        is at time 0.

        Returns
        -------
        List of floats
            A list of zero-shifted times (JD or MJD).
        """
        # what's the deal here - are we required to be sorted?
        return self.data["mjd"] - self.data["mjd"][0]

    def get_duration(self):
        """Returns a list of timestamps such that the first image
        is at time 0.

        Returns
        -------
        List of floats
            A list of zero-shifted times (JD or MJD).
        """
        # maybe timespan?
        return self.data["mjd"][-1] - self.data["mjd"][0]
