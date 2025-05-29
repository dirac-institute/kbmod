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
from astropy.io import fits as fitsio
from astropy.wcs import WCS
from astropy.utils import isiterable

import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from .standardizers import Standardizer


from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized


__all__ = [
    "ImageCollection",
]


logger = logging.getLogger(__name__)


def pack_table(data):
    """Given a `Table`, find columns containing the same values and pack them
    into the `Table` metadata as keys.

    Parameters
    ----------
    data : `Table`
        Table to pack.

    Returns
    -------
    packed : `Table`
        Packed table.
    """
    shared_values = {}
    for col in data.columns:
        vals = np.unique(data[col])
        if len(vals) == 1:
            # for some reason yaml can't serialize np.str_
            if isinstance(vals[0], np.str_):
                shared_values[col] = str(vals[0])
            else:
                shared_values[col] = vals[0]

    data.meta["shared_cols"] = list(shared_values.keys())
    data.meta.update(shared_values)
    data.meta["is_packed"] = True

    data.remove_columns(data.meta["shared_cols"])
    return data


def unpack_table(data):
    """Given a packed `Table`, unpack the shared data as columns of the table.

    If the given table does not contain an ``is_packed`` metadata entry, this
    is a no-op.

    Parameters
    ----------
    data : `Table`
        Table to unpack.

    Returns
    -------
    packed : `Table`
        Unpacked table.
    """
    is_packed = data.meta.get("is_packed", False)
    if not is_packed:
        return data

    n_rows = 1 if len(data) == 0 else len(data)
    for col in data.meta["shared_cols"]:
        data[col] = np.full((n_rows,), data.meta[col])

    for col in data.meta["shared_cols"]:
        data.meta.pop(col)
    data.meta.pop("shared_cols")
    data.meta["is_packed"] = False

    return data


class ImageCollection:
    """A collection of metadata extracted by standardizers.

    Columns listed in the `ImageCollection.required_metadata` are required to
    exist and be non-zero. Additional columns may or may not exist. The names
    of the standardizers used to extract the metadata are also guaranteed to
    exist.

    Generating the exact `Standardizer` objects that created the row may or may
    not be possible, depending on whether the data is accessible to the user.

    Avoid constructing this object directly. Use one of the provided factory
    methods instead. Their behaviour can be modified by supplying a callable
    as the `forceStandardizer` argument in order to change the mechanism or
    modify the extracted data. The provided callable has to be an instance of
    `Standardizer` class, see the factory method's documentation and
    `Standardizer` documentation for more information.

    Parameters
    ----------
    metadata : `~astropy.table.Table`
        A table of exposure metadata properties
    serializers : `list`
        A list of valid serializer names, used to extract the metadata in its
        row.
    enable_lazy_loading : `bool`
        Enable lazy loading of the standardizers, `True` by default. When
        enabled, and if possible, a reference to the constructed `Standardizer`
        objects that built the table are kept. Further calls to
        `get_standardizer` or `get_standardizers` methods then will not pay the
        price of the resource acquisition when building the metadata table.
        When that price is low, f.e. FITS files on a local SSD drive, turning
        lazy loading off will reduce memory footprint. When data acquisition
        is large, f.e. a non-local Butler, that space is traded in favor of
        avoiding the cost of accessing the file.
    validate : `bool`
        Validate the given metadata during initialization.

    Attributes
    ----------
    data : `~astropy.table.Table`
        Table of exposure metadata and internal metadata book-keeping properties.
        Should not be directly modified.
    _standardizers : `list` or `None`
        The current list of loaded `Standardizer`s. Entry will be none if the
        standardizer was not previously loaded and lazy loading is enabled.
        When `None`, lazy loading is disabled. Should not be directly modified.

    Raises
    ------
    ValueError :
        when instantiated from a Table which does not have the required
        columns, or has null-values in the required columns.
    """

    # Both are required, but supporting metadata is mostly handled internally
    required_metadata = [
        "mjd_mid",
        "ra",
        "dec",
        "wcs",
        "obs_lon",
        "obs_lat",
        "obs_elev",
    ]
    _supporting_metadata = ["std_name", "std_idx", "ext_idx", "config"]

    ########################
    # CONSTRUCTORS
    ########################
    def __init__(self, metadata, standardizers=None, enable_std_caching=True, validate=True):
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
            metadata.meta["n_stds"] = len(standardizers)
        else:
            n_stds = metadata.meta.get("n_stds", None)
            if n_stds is None:
                n_stds = metadata["std_idx"].max()
                self.data.meta["n_stds"] = n_stds

            if enable_std_caching:
                self._standardizers = np.full((n_stds,), None)

        self.data = metadata
        self._userColumns = [col for col in self.data.columns if col not in self._supporting_metadata]
        if validate:
            self.validate()

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
                    row["std_name"] = str(std.name)

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
        recognized by at least one of the standardizers.

        Parameters
        ----------
        tgts : `iterable`
            Collection of file-paths, a path to a directory, URIs, a butler and
            dataset ids or reference objects, or any other data targets that
            are supported by the standardizers.
        force : `Standardizer`, `str` or `None`
            If `None`, all available `Standardizer`s are tested to find the
            appropriate one. When multiple `Standardizer`s are found, the one
            with the highest priority is selected. A name of one
            of the registered standardizers can be provided. Optionally,
            provide the `Standardizer` class itself in which case it will be
            called for each target in the iterable.
        config : `~StandardizerConfig`, `dict` or `None`, optional
            Standardizer configuration or dictionary containing the config
            parameters for standardization. When `None` default values for the
            appropriate `Standardizer` will be used.
        **kwargs : `dict`
            Remaining keyword arguments are passed to the `Standardizer`.

        Raises
        ------
        KeyError:
            When a name of a non-registered standardizer is given.
        ValueError:
            When none of the registered standardizers volunteer to process the
            given target.
        """
        standardizers = [Standardizer.get(tgt, force=force, config=config, **kwargs) for tgt in tgts]
        return cls.fromStandardizers(standardizers)

    @classmethod
    def fromDir(cls, dirpath, recursive=False, force=None, config=None, **kwargs):
        """Instantiate ImageInfoSet from a path to a directory
        containing FITS files.

        Parameters
        ----------
        dirpath : `path-like`
            Path to a directory containing FITS files.
        recursive : `bool`
            If the location is a local filesystem directory, scan it
            recursively including all sub-directories.
        force : `Standardizer` or `None`
            If `None`, when applicable, determine the correct `Standardizer` to
            use automatically. Otherwise force the use of the given
            `Standardizer`.
        config : `~StandardizerConfig`, `dict` or `None`, optional
            Standardizer configuration or dictionary containing the config
            parameters for standardization. When `None` default values for the
            appropriate `Standardizer` will be used.
        **kwargs : `dict`
            Remaining kwargs, not listed here, are passed onwards to
            the underlying `Standardizer`.
        """
        logger.debug(f"Building ImageCollection from FITS filtes in: {dirpath}")
        fits_files = glob.glob(os.path.join(dirpath, "*fits*"), recursive=recursive)
        logger.debug(f"Found {len(fits_files)} matching files:\n{fits_files}")
        return cls.fromTargets(fits_files, force=force, config=config, **kwargs)

    @classmethod
    def fromBinTableHDU(cls, hdu):
        """Create an image collection out of a `BinTableHDU` object.

        Parameters
        ----------
        hdu : `BinTableHDU`
            A fits table object containing the metadata required to make an
            image collection.

        Returns
        -------
        ic : `ImageCollection`
            Image collection
        """
        metadata = Table(hdu.data)
        metadata.meta["n_stds"] = hdu.header["N_STDS"]
        return cls(metadata)

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

    def reset_lazy_loading_indices(self):
        """Resets the internal index lookup table and standardizer list used
        for lazy loading to a contiguous array starting at 0.

        Image collection tracks `Standardizers` that were used to create the
        metadata table rows on a per-row basis. Selecting rows or columns from
        the image collection does not sub-select the requested standardizers or
        reset these internal counters as often this can rather time-consuming.

        Instead, the full list of already loaded standardizers is carried over
        and original lookup indices remain unchanged. While faster than
        recalculating the indices at every selection, this can leave a
        fragmented index lookup table and a longer list of standardizers
        compared to the number of rows in the table.

        Calling this method will reset the index lookup table to a new
        zero-based contiguous state while trimming all unused lazy-loaded
        standardizers from the list. Loaded standardizers will not be
        un-loaded.

        In practical use-case the standardizer indices rarely have to be reset
        because the cost of carrying even few-thousand item long list of `None`
        entries carries an insignificant memory footprint. Nominally, the
        use-case is the situation when creating small, few hundreds rows, image
        collections from a very large image collection containing >10 000+ rows.
        """
        if self._standardizers is None:
            return

        counter = 0
        seen = {}
        new_idxs, stds = [], []
        for i, idx in enumerate(self.data["std_idx"]):
            if idx in seen:
                new_idxs.append(seen[idx])
            else:
                stds.append(self._standardizers[idx])
                seen[idx] = counter
                new_idxs.append(counter)
                counter += 1
        self._standardizers = stds
        self.data["std_idx"] = new_idxs
        self.data.meta["n_stds"] = counter

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._userColumns:
                raise KeyError(f"{key}")
            return self.data[key]
        elif isinstance(key, int):
            return self.data[key][self._userColumns]
        elif isinstance(key, (tuple, list)) and isinstance(key[0], str):
            noexist = [k for k in key if k not in self._userColumns]
            if len(noexist) > 0:
                raise KeyError(f"{noexist}")
            return self.data[key]
        else:
            # key is slice, array, list of idxs, boolean mask etc...
            new_meta = self.data[key]
            # Since we're constucting an ImageCollection from a slice of the
            # original, we need to reset the standardizer indices and the
            # standardizer list to match the number of rows in the new table.
            new_meta.meta["n_stds"] = min(len(new_meta), self.meta["n_stds"])
            new_meta.meta["std_idx"] = range(len(new_meta))
            return self.__class__(new_meta, standardizers=self._standardizers)

    def __setitem__(self, key, val):
        self.data[key] = val

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        # start with cheap comparisons
        if not isinstance(other, ImageCollection):
            return False

        if not self.data.columns.keys() == other.data.columns.keys():
            return False

        if len(self.data) != len(other.data):
            return False

        # before we compare the entire tables
        # WCS not comparable, BBox not compared
        cols = [col for col in self.columns if col not in ("wcs", "bbox")]
        # I think it's a bug in AstropyTables, but this sometimes returns
        # a boolean instead of an array of booleans (only when False)
        equal = self.data[cols] == other.data[cols]
        if isinstance(equal, bool):
            return equal
        return equal.all()

    @property
    def meta(self):
        """Image collection metadata.

        Contains ``shared_cols`` and values when collection is in packed state.
        """
        return self.data.meta

    @property
    def is_packed(self):
        """Values shared by all rows are packed as table metadata to save space."""
        if "is_packed" in self.data.meta:
            return self.data.meta["is_packed"]
        return False

    @property
    def wcs(self):
        """Iterate through `WCS` of each row."""
        for i in range(len(self.data)):
            # the warnings that some keywords might be ignored are expected
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield WCS(json.loads(self.data[i]["wcs"]), relax=True)

    def reflex_correct(self, guess_distance, earth_loc):
        """
        Generate reflex-corrected coordinates for each image in the ImageCollection.

        This adds inplace new columns in the ImageCollection with the suffix
        ``_{guess_distance}`` for each coordinate column. If a list of distances
        is provided, the correction will be applied for each distance in the list.

        The helper function `reflex_corrected_col` is used to generate the new column names.

        Parameters
        ----------
        guess_distance : `float` or `list`
            The guess distance in au. If a list is provided, the correction will be
            applied for each distance in the list.
        earth_loc : `EarthLocation`
            The location of the Earth in which the parallax correction is calculated.
        """
        guess_dists = [guess_distance] if not isinstance(guess_distance, list) else guess_distance
        for guess_dist in guess_dists:
            # Calculate the parallax correction for each RA, Dec in the ImageCollection
            corrected_ra_dec, _ = correct_parallax_geometrically_vectorized(
                self.data["ra"],
                self.data["dec"],
                self.data["mjd_mid"],
                guess_dist,
                earth_loc,
            )
            # Add the corrected coordinates to the ImageCollection
            self.data[self.reflex_corrected_col("ra", guess_dist)] = corrected_ra_dec.ra.deg
            self.data[self.reflex_corrected_col("dec", guess_dist)] = corrected_ra_dec.dec.deg

            # Now we want to reflex-correct the corners for each image in the collection.
            for box_corner in ["tl", "tr", "bl", "br"]:
                corrected_ra_dec_corner, _ = correct_parallax_geometrically_vectorized(
                    self.data[f"ra_{box_corner}"],
                    self.data[f"dec_{box_corner}"],
                    self.data["mjd_mid"],
                    guess_dist,
                    earth_loc,
                )

                ra_col = self.reflex_corrected_col(f"ra_{box_corner}", guess_dist)
                self.data[ra_col] = corrected_ra_dec_corner.ra.deg

                dec_col = self.reflex_corrected_col(f"dec_{box_corner}", guess_dist)
                self.data[dec_col] = corrected_ra_dec_corner.dec.deg

    def reflex_corrected_col(self, col_name, guess_dist):
        """Get the name of the reflex-corrected column for a given guess distance.

        These columns may be added by calling `ImageCollection.reflex_correct`.

        Parameters
        ----------
        guess_dist : `float`
            The guess distance in parsecs.

        Returns
        -------
        col_name : `str`
            The name of the reflex-corrected column.
        """
        if col_name not in self.data.columns:
            raise ValueError(f"Column {col_name} not in ImageCollection")
        if not isinstance(guess_dist, float):
            raise ValueError("Reflex-corrected guess distance must be a float")
        if guess_dist == 0.0:
            return col_name
        return f"{col_name}_{guess_dist}"

    def filter_by_mjds(self, mjds, time_sep_s=0.001):
        """
        Filter the visits in the ImageCollection by the given MJDs. Is performed in-place.

        Note that the comparison is made against "mjd_mid"

        Parameters
        ----------
        ic : ImageCollection
            The ImageCollection to filter.
        timestamps : list of floats
            List of timestamps to keep.
        time_sep_s : float, optional
            The maximum separation in seconds between the timestamps in the ImageCollection and the timestamps to keep.

        Returns
        -------
        None
        """
        if len(self.data) < 1:
            return
        if time_sep_s < 0:
            raise ValueError("time_sep_s must be positive")
        mask = np.zeros(len(self.data), dtype=bool)
        for mjd in mjds:
            mjd_diff = abs(self.data["mjd_mid"] - mjd)
            mask = mask | (mjd_diff <= time_sep_s / (24 * 60 * 60))
        self.data = self.data[mask]

    def filter_by_time_range(self, start_mjd=None, end_mjd=None):
        """
        Filter the ImageCollection by the given time range. Is performed in-place.

        Note that it uses the "mjd_mid" column to filter.

        Parameters
        ----------
        start_mjd : float, optional
            The start of the time range in MJD. Optional if `end_mjd` is provided.
        end_mjd : float, optional
            The end of the time range in MJD. Optional if `start_mjd` is provided.
        """
        if start_mjd is None and end_mjd is None:
            raise ValueError("At least one of start_mjd or end_mjd must be provided.")
        if start_mjd is not None and end_mjd is not None and start_mjd > end_mjd:
            raise ValueError("start_mjd must be less than end_mjd.")
        if start_mjd is not None:
            self.data = self.data[self.data["mjd_mid"] >= start_mjd]
        if end_mjd is not None:
            self.data = self.data[self.data["mjd_mid"] <= end_mjd]

    def get_wcs(self, idxs):
        """Get a list of WCS objects for selected rows.

        Parameters
        ----------
        idxs : `int`, `slice`, `list[int]`
            Indices of rows for which to get WCS objects.

        Returns
        -------
        wcss : `list[WCS]`
            WCS object for the selected rows.
        """
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
        """Iterate through `BBox` of each row."""
        # what we return here depends on what region search needs
        # best probably to return a BBox dataclass that has some useful
        # functionality for region search or something, maybe bbox
        # even needs a timestamp or something...
        cols = [
            "ra",
            "dec",
            "ra_tl",
            "dec_tl",
            "ra_tr",
            "dec_tr",
            "ra_bl",
            "dec_bl",
            "ra_br",
            "dec_br",
        ]
        for i in range(len(self.data)):
            yield self.data[cols][i]

    def get_bbox(self, idxs):
        """Get a list of BBOX objects for selected rows.

        Parameters
        ----------
        idxs : `int`, `slice`, `list[int]`
            Indices of rows for which to get WCS objects.

        Returns
        -------
        bboxes : `list[BBox]`
            BBox object for the selected rows.
        """
        # again, we can return an BBox collection object with additional methods
        cols = [
            "ra",
            "dec",
            "ra_tl",
            "dec_tl",
            "ra_tr",
            "dec_tr",
            "ra_bl",
            "dec_bl",
            "ra_br",
            "dec_br",
        ]
        selected = self.data[cols][idxs]
        return selected

    @property
    def columns(self):
        """Return metadata columns."""
        # interesting, in python 3.10.9  using unpacking operator * inside bracket
        # operator is considered SyntaxError. But casting the columns into a tuple
        # (basically what unpacking operator would've done) is a-ok. TODO: update
        # to unpacking operator when 3.10 stops being supported.
        return self.data.columns[tuple(self._userColumns)]

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
        # of _standardizers exists, then we can cache, since the
        # implication is it isn't too long. Keep in mind some standardizers
        # keep their resources alive, including images - which can be memory
        # intensive.
        if self._standardizers is None:
            # no caching
            std = load_std()
        elif self._standardizers[std_idx] is None:
            # lazy load and cache
            std = load_std()
            self._standardizers[std_idx] = std
        else:
            # already loaded
            std = self._standardizers[std_idx]

        # maybe a clever dataclass to shortcut the idx lookups on the user end?
        return {"std": std, "ext": self.data[index]["ext_idx"]}

    def get_standardizers(self, idxs=None, **kwargs):
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
        if idxs is None:
            return [self.get_standardizer(idx, **kwargs) for idx in range(self.data["std_idx"].max() + 1)]
        # this keeps happening to me, despite having a get_standardizer method
        # See Issue #543
        if isinstance(idxs, int):
            return [
                self.get_standardizer(idxs, **kwargs),
            ]
        return [self.get_standardizer(idx, **kwargs) for idx in idxs]

    ########################
    # IO
    ########################
    @classmethod
    def read(
        cls,
        *args,
        format="ascii.ecsv",
        units=None,
        descriptions=None,
        unpack=True,
        validate=True,
        **kwargs,
    ):
        """Create ImageCollection from a file containing serialized image
        collection.

        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to data reader. If supplied the
            first argument is the input filename.
        format : `str`
            File format specified, one of AstroPy IO formats that must support
            comments.  Default: `ascii.ecsv`
        units : `list`
            List or dict of units to apply to columns.
        descriptions : `list`
            List or dict of descriptions to apply to columns
        unpack : `bool`
            If reading a packed image collection, unpack the shared values.
        validate : `bool`
            Validate that all required metadata exists.
        kwargs: `dict`
            Other keyword arguments passed onwards to AstroPy's `Table.read`.

        Returns
        -------
        ic : `ImageCollection`
            Image Collection
        """
        metadata = Table.read(*args, format=format, units=units, descriptions=descriptions, **kwargs)
        if unpack and metadata.meta.get("is_packed", False):
            metadata = unpack_table(metadata)
        return cls(metadata, validate=validate)

    def write(
        self,
        *args,
        format="ascii.ecsv",
        serialize_method=None,
        pack=True,
        validate=True,
        **kwargs,
    ):
        """Write the ImageCollection to a file or file-like object.

        A light wrapper around the underlying AstroPy's Table ``write``
        functionality. See `astropy/io.ascii.write`
        `documentation <https://docs.astropy.org/en/stable/io/ascii/write.html#parameters-for-write>`_

        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to data writer. If supplied the
            first argument is the output filename.
        format : `str`
            File format specified, one of AstroPy IO formats that must support
            comments.  Default: `ascii.ecsv`
        serialize_method : `str`, `dict`, optional
            Serialization method specifier for columns.
        pack : `bool`
            Pack the values shared by all rows into the table metadata.
        validate : `bool`
            Validate that all required metadata exists before writing it.
        kwargs: `dict`
            Other keyword arguments passed onwards to AstroPy's `Table.write`.
        """
        logger.info(f"Writing ImageCollection to {args[0]}")
        if validate:
            self.validate()
        tmpdata = self.data.copy()
        if pack:
            tmpdata = pack_table(tmpdata)
        tmpdata.write(*args, format=format, serialize_method=serialize_method, **kwargs)

    ########################
    # FUNCTIONALITY (object operations, transformative functionality)
    ########################
    def _validate(self):
        """See `validate`.

        Returns
        -------
        valid : `bool`
            Metadata is valid.
        explanation : `str`
            Explanation of reason why metadata is not valid. Emtpy string when
            valid.
        """
        if not isinstance(self.data, Table):
            return False, "not an Astropy Table object."

        # if empty table
        if not self.data:
            return False, "an empty table."

        # create a list of table columns, columns with shared
        # value and the join of the two
        tbl_cols = self.data.columns
        shared_cols = []
        all_cols = [n for n in self.data.columns]
        if "shared_cols" in self.data.meta:
            shared_cols = self.data.meta["shared_cols"]
            all_cols.extend(self.data.meta["shared_cols"])

        # check no required keys are left out of anywhere
        missing_keys = [key for key in self.required_metadata if key not in all_cols]
        if missing_keys:
            return False, f"missing required columns: {missing_keys}"

        missing_keys = [key for key in self._supporting_metadata if key not in all_cols]
        if missing_keys:
            return False, (f"missing required standardizer-row lookup indices: {missing_keys}")

        # finally check that no values are empty in some way.
        # Perhaps we should be checking np.nan too?
        for col in tbl_cols:
            # Future astropy versions promise to make 'in' operator an elementwise
            # comparator. Until then it's a column-wise operation. We check what
            # type we get and silence the warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                test1 = None in self.data[col]
                test1 = test1 if isinstance(test1, bool) else test1.any()
                if test1:
                    return False, f"missing required self.data values: {col}"

        for col in shared_cols:
            if self.data.meta[col] is None or self.data.meta[col] == "":
                return False, f"missing required self.data values: {col}"

        return True, ""

    def validate(self):
        """Validate the metadata table has all the required values
        and that none of them are false.

        Requires all columns in ``required_cols`` and ``_supporting_cols``
        attributes exist.

        Returns
        -------
        valid : `bool`
            `True` if valid

        Raises
        ------
        ValueError:
            When not valid, raises a value error with explanation
            of condition that wasn't satisfied.
        """
        valid, explanation = self._validate()
        if not valid:
            raise ValueError(f"Metadata is {explanation}")
        return valid

    def copy(self, copy_data=True):
        """Return a copy of ImageCollection.

        Parameters
        ----------
        copy_data = True
            If `True` (default) copies the underlying `Table`
            data and creates a deep copy of `meta` attribute.
        """
        return self.__class__(self.data.copy(copy_data=copy_data))

    def pack(self):
        """Identifies columns containing the same repeated value and
        stores it as a key in the `meta` attribute.

        Lists all the stored keys under the ``shared_cols`` value of
        `meta`. Reduce the size of the final serialized Table on disk.
        """
        self.data = pack_table(self.data)
        self._userColumns = [col for col in self.data.columns if col not in self._supporting_metadata]

    def unpack(self, data=None):
        """Unpacks the shared data from `meta` into columns."""
        self.data = unpack_table(self.data)
        self._userColumns = [col for col in self.data.columns if col not in self._supporting_metadata]

    def vstack(self, ics):
        """Stack multiple image collections vertically (along rows) into a new,
        larger, image collection.

        .. note::
           Modifies the ImageCollection in place.

        Parameters
        ----------
        ics : `list[ImageCollection]`
            List of image collections that will be stacked.

        Returns
        -------
        ic : `ImageCollection`
            Extended image collection.
        """
        self.unpack()
        std_offset = self.meta["n_stds"]

        old_metas, old_offsets = [], []
        data = []
        for ic in ics:
            n_stds = ic.data["std_idx"].max()
            old_metas.append(ic.meta.copy())
            old_offsets.append(std_offset)
            ic.data["std_idx"] += std_offset
            ic.data.meta = None
            data.append(ic.data)
            if self._standardizers is not None:
                if ic._standardizers is not None:
                    self._standardizers.extend(ic._standardizers)
                else:
                    self._standardizers.extend([None] * n_stds)
            std_offset += n_stds

        self.data = vstack([self.data, *data], metadata_conflicts="silent")
        self.data.meta["n_stds"] = self.data["std_idx"].max()

        for meta, offset, ic in zip(old_metas, old_offsets, ics):
            ic.data["std_idx"] -= offset
            ic._meta = meta

        self.reset_lazy_loading_indices()
        return self

    def get_zero_shifted_times(self):
        """Returns a list of timestamps such that the earliest time is treated
        as 0.

        Returns
        -------
        List of floats
            A list of zero-shifted times (JD or MJD).
        """
        return self.data["mjd"] - self.data["mjd"].min()

    def toBinTableHDU(self):
        """Writes the image collection as a `BinTableHDU` object.

        If image collection was packed, it is unpacked before the table is
        created.

        Returns
        -------
        bintbl : `astropy.io.fits.BinTableHDU`
            Image collection as a flattened table HDU.
        """
        if self.is_packed:
            self.unpack()
            self.meta.pop("is_packed", None)
        return fitsio.hdu.BinTableHDU(self.data, name="IMGCOLL")

    def toWorkUnit(self, search_config=None, **kwargs):
        """Return an `~kbmod.WorkUnit` object for processing with
        KBMOD.

        Parameters
        ----------
        search_config : `~kbmod.SearchConfiguration` or None, optional
            Search configuration. Default ``None``.

        Returns
        -------
        work_unit : `~kbmod.WorkUnit`
            A `~kbmod.WorkUnit` object for processing with KBMOD.
        """
        from .work_unit import WorkUnit

        logger.info("Building WorkUnit from ImageCollection")

        # Extract data from each standardizer and each LayeredImagePy within
        # that standardizer.
        layered_images = []
        for std in self.get_standardizers(**kwargs):
            for img in std["std"].toLayeredImage():
                layered_images.append(img)

        # Extract all of the relevant metadata from the ImageCollection.
        metadata = Table(self.toBinTableHDU().data)
        if None not in self.wcs:
            metadata["per_image_wcs"] = list(self.wcs)

        # WorkUnit expects a 'data_loc' column, so we rename 'location' to 'data_loc'.
        if "data_loc" not in metadata.columns and "location" in metadata.columns:
            metadata.rename_column("location", "data_loc")

        # Create the basic WorkUnit from the ImageStackPy.
        imgstack = ImageStackPy()
        for layimg in layered_images:
            imgstack.append_layered_image(layimg)
        work = WorkUnit(imgstack, search_config, org_image_meta=metadata)

        return work
