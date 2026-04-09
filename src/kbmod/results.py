"""Results is a column-based data structure for tracking results with additional global data
and helper functions for filtering and maintaining consistency between different attributes in each row.
"""

import copy
import csv
import logging
import numpy as np
import re
import uuid

from astropy.io import fits
from astropy.io.misc.yaml import load as astropy_yaml_load
from astropy.table import Column, Table, vstack
from astropy.time import Time
from pathlib import Path
import pyarrow.parquet as pq

from kbmod.search import (
    extract_all_trajectory_flux,
    extract_all_trajectory_lh,
    extract_all_trajectory_obs_count,
    extract_all_trajectory_vx,
    extract_all_trajectory_vy,
    extract_all_trajectory_x,
    extract_all_trajectory_y,
    Trajectory,
)
from kbmod.wcs_utils import deserialize_wcs, serialize_wcs

logger = logging.getLogger(__name__)


class Results:
    """This class stores a collection of related data from all of the kbmod results.

    At a minimum it contains columns for the trajectory information:
    (x, y, vx, vy, likelihood, flux, obs_count)
    but additional columns can be added as needed.

    Attributes
    ----------
    table : `astropy.table.Table`
        The stored results data.
    wcs : `astropy.wcs.WCS`
        A global WCS for all the results. This is optional and primarily used when saving
        the results to a file so as to preserve the WCS for future analysis.
    mjd_mid : `np.ndarray`
        An array of the times (mid-MJD) for each observation in UTC. This is optional
        and primarily used when saving the results to a file so as to preserve the times
        for future analysis.
    track_filtered : `bool`
        Whether to track (save) the filtered trajectories. This will use
        more memory and is recommended only for analysis.
    filtered : `dict`
        A dictionary mapping a string of filtered name to a table
        of the results removed by that filter.
    filtered_stats : `dict`
        A dictionary mapping a string of filtered name to a count of how
        many results were removed by that filter.
        This is maintained even if ``track_filtered`` is ``False``.
    """

    # The required columns list gives a list of tuples containing
    # (column name, dype, default value) for each required column.
    required_cols = [
        ("x", int, 0),
        ("y", int, 0),
        ("vx", float, 0.0),
        ("vy", float, 0.0),
        ("likelihood", float, 0.0),
        ("flux", float, 0.0),
        ("obs_count", int, 0),
    ]
    _required_col_names = set([rq_col[0] for rq_col in required_cols])

    # We only support a few output formats since we need to save metadata.
    _supported_formats = [".ecsv", ".parq", ".parquet", ".hdf5"]

    def __init__(self, data=None, track_filtered=False, wcs=None):
        """Create a ResultTable class.

        Parameters
        ----------
        data : `dict`, `astropy.table.Table`
            The data for the results table.
        track_filtered : `bool`
            Whether to track (save) the filtered trajectories. This will use
            more memory and is recommended only for analysis.
        wcs : `astropy.wcs.WCS`, optional
            A global WCS for the results.
        """
        self.wcs = wcs
        self.mjd_mid = None

        # Set up information to track which row is filtered at which round.
        self.track_filtered = track_filtered
        self.filtered = {}
        self.filtered_stats = {}

        if data is None:
            # Set up the basic table meta data.
            self.table = Table(
                names=[col[0] for col in self.required_cols],
                dtype=[col[1] for col in self.required_cols],
            )
        elif isinstance(data, dict):
            self.table = Table(data)
        elif isinstance(data, Table):
            self.table = data
        else:
            raise TypeError(f"Incompatible data type {type(data)}")

        # Check if there is a uuid column and, if not, generate one. We set it as
        # a Column object so we can set the dtype even in the case of empty results.
        if "uuid" not in self.table.colnames:
            col = Column(
                data=[uuid.uuid4().hex for i in range(len(self.table))],
                name="uuid",
                dtype="str",
            )
            self.table.add_column(col)

        # Check that we have the correct columns.
        for col in self.required_cols:
            if col[0] not in self.table.colnames:
                raise KeyError(f"Column {col[0]} missing from input data.")

    def __len__(self):
        return len(self.table)

    def __str__(self):
        return str(self.table)

    def __repr__(self):
        return repr(self.table).replace("Table", "Results")

    def _repr_html_(self):
        return self.table._repr_html_().replace("Table", "Results")

    def __getitem__(self, key):
        return self.table[key]

    @property
    def mjd_utc_mid(self):
        return self.mjd_mid

    @property
    def mjd_tai_mid(self):
        return Time(self.mjd_mid, format="mjd", scale="utc").tai.mjd

    def set_mjd_utc_mid(self, times):
        """Set the midpoint times in UTC MJD."""
        self.mjd_mid = times

    @property
    def colnames(self):
        return self.table.colnames

    def get_num_times(self):
        """Get the number of observations times in the data as computed
        from the lightcurves or the marked valid observations. Returns 0 if
        there is no time series information.

        Returns
        -------
        result : `int`
            The number of time steps. Returns 0 if there is no such information.
        """
        if self.mjd_mid is not None:
            return len(self.mjd_mid)
        if "psi_curve" in self.table.colnames:
            return self.table["psi_curve"].shape[1]
        if "phi_curve" in self.table.colnames:
            return self.table["phi_curve"].shape[1]
        if "obs_valid" in self.table.colnames:
            return self.table["obs_valid"].shape[1]
        return 0

    def copy(self):
        """Return a deep copy of the current Results object."""
        return copy.deepcopy(self)

    @classmethod
    def from_trajectories(cls, trajectories, track_filtered=False):
        """Extract data from a list of Trajectory objects.

        Parameters
        ----------
        trajectories : `list[Trajectory]`
            A list of trajectories to include in these results.
        track_filtered : `bool`
            Indicates whether to track future filtered points.
        """
        # Create a table object from the Trajectories.
        input_table = Table()
        input_table["x"] = extract_all_trajectory_x(trajectories)
        input_table["y"] = extract_all_trajectory_y(trajectories)
        input_table["vx"] = extract_all_trajectory_vx(trajectories)
        input_table["vy"] = extract_all_trajectory_vy(trajectories)
        input_table["likelihood"] = extract_all_trajectory_lh(trajectories)
        input_table["flux"] = extract_all_trajectory_flux(trajectories)
        input_table["obs_count"] = extract_all_trajectory_obs_count(trajectories)

        # Check for any missing columns and fill in the default value.
        for col in cls.required_cols:
            if col[0] not in input_table.colnames:
                input_table[col[0]] = [col[2]] * len(trajectories)

        # Create the table and add the unfiltered (and filtered) results.
        results = Results(input_table, track_filtered=track_filtered)
        return results

    @classmethod
    def read_table(cls, filename, track_filtered=False, load_aux_files=False):
        """Read the Results from a table file. The file format is automatically
        determined from the file name's suffix which must be one of ".ecsv",
        ".parquet", ".parq", or ".hdf5".

        Parameters
        ----------
        filename : `str`
            The name of the file to load.
        track_filtered : `bool`
            Indicates whether the object should track future filtered points.
        load_aux_files : `bool`
            If True the code will check the file path for any auxiliary files
            that share the same base name as the main file and load them.
            Default: False

        Raises
        ------
        Raises a FileNotFoundError if the file is not found.
        Raises a KeyError if any of the columns are missing.
        """
        logger.info(f"Reading results from {filename}")

        filepath = Path(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filename} not found.")
        if filepath.suffix not in cls._supported_formats:
            raise ValueError(
                f"Unsupported file type '{filepath.suffix}' " f"use one of {cls._supported_formats}."
            )
        data = Table.read(filename)

        # Parse metadata (WCS, times, image column shapes)
        wcs, mjd_mid, image_column_shapes = cls._parse_table_metadata(data.meta)

        # Create the results object.
        results = Results(data, track_filtered=track_filtered, wcs=wcs)

        # Attach the mjd_mid if found (already np.array from _parse_table_metadata)
        if mjd_mid is not None:
            results.set_mjd_utc_mid(mjd_mid)

        if load_aux_files:
            # Check for any auxiliary files that share the same base name,
            # but have _colname in the stem.
            aux_files = filepath.parent.glob(f"{filepath.stem}_*")
            for aux_file in aux_files:
                colname = aux_file.stem.replace(f"{filepath.stem}_", "")
                logger.info(f"Loading column {colname} results from {aux_file}")
                results.load_column(aux_file, colname=colname)

        # Reshape image columns if we have stored shape metadata
        # (parquet flattens multi-dimensional arrays to 1D)
        # This must happen after loading aux files since they may contain image columns.
        results._reshape_image_columns(data.meta.get("image_column_shapes"))

        return results

    @classmethod
    def read_table_chunks(cls, filename, chunk_size=10000):
        """Read the Results from a table file in chunks.

        This is a generator that yields `Results` objects for each chunk of data.
        It is designed to handle very large result files that cannot fit into memory.

        It also does not load any auxiliary files for this result file.

        Filenme must be one of  ".parquet", ".parq", since chunked reading is
        currently only supported for parquet files.

        Parameters
        ----------
        filename : `str`
            The name of the file to load. Must be a parquet file.
        chunk_size : `int`
            Number of rows to include in each chunk. Default is 10000.

        Yields
        ------
        results : `Results`
            A Results object containing a chunk of the data.
        """
        logger.info(f"Reading results from {filename} in chunks of {chunk_size}")
        filepath = Path(filename)

        if not filepath.is_file():
            raise FileNotFoundError(f"File {filename} not found.")
        if ".parquet" not in filepath.suffix and ".parq" not in filepath.suffix:
            raise ValueError("Chunked reading currently only supported for parquet files.")

        # Open the parquet file
        pf = pq.ParquetFile(filename)

        # Extract metadata from the parquet file without reading all data
        meta_dict = cls._extract_parquet_metadata(pf)

        # Parse metadata (WCS, times, image column shapes)
        wcs, mjd_mid, image_column_shapes = cls._parse_table_metadata(meta_dict)

        # Iterate over batches
        for batch in pf.iter_batches(batch_size=chunk_size):
            # Convert pyarrow batch to Astropy Table
            # Converting to pandas first is often more robust for astropy conversion
            batch_table = Table.from_pandas(batch.to_pandas())

            # Create a mini-Results object
            # Note: We don't use track_filtered for chunks usually as it's for streaming analysis
            results = Results(batch_table, track_filtered=False, wcs=wcs)

            # Attach the global mjd_mid if found
            if mjd_mid is not None:
                results.set_mjd_utc_mid(mjd_mid)

            # Reshape image columns if we have stored shape metadata
            # (parquet flattens multi-dimensional arrays to 1D)
            results._reshape_image_columns(image_column_shapes)

            yield results

    @staticmethod
    def _extract_parquet_metadata(pf):
        """Extract Astropy table metadata from a parquet file without reading all data.

        Astropy stores table metadata in a YAML-encoded format within the parquet
        schema's key-value metadata under the key 'table_meta_yaml'.

        Parameters
        ----------
        pf : `pyarrow.parquet.ParquetFile`
            An open parquet file object.

        Returns
        -------
        meta_dict : `dict`
            A dictionary containing the extracted metadata. Returns an empty dict
            if no metadata is found or parsing fails.
        """
        arrow_metadata = pf.schema_arrow.metadata
        if not arrow_metadata or b"table_meta_yaml" not in arrow_metadata:
            return {}

        try:
            yaml_bytes = arrow_metadata[b"table_meta_yaml"]
            yaml_str = yaml_bytes.decode("utf-8")
            meta_obj = astropy_yaml_load(yaml_str)

            # Extract the 'meta' field from the YAML structure
            # Astropy serializes table.meta as a list of tuples (OrderedDict format)
            meta_dict = {}
            if isinstance(meta_obj, dict) and "meta" in meta_obj:
                raw_meta = meta_obj["meta"]
                # Convert list of tuples to dict
                if isinstance(raw_meta, list):
                    for item in raw_meta:
                        if isinstance(item, (tuple, list)) and len(item) == 2:
                            key, value = item
                            meta_dict[key] = value
                        elif isinstance(item, dict):
                            meta_dict.update(item)
                elif isinstance(raw_meta, dict):
                    meta_dict = raw_meta

            return meta_dict
        except Exception as e:
            logger.warning(f"Failed to parse astropy metadata from parquet: {e}")
            return {}

    @staticmethod
    def _parse_table_metadata(meta_dict):
        """Parse common metadata fields from a table's metadata dictionary.

        Extracts WCS, observation times, and image column shape information
        from the metadata. This is used by both `read_table` and `read_table_chunks`.

        Parameters
        ----------
        meta_dict : `dict`
            A dictionary containing table metadata (e.g., from `table.meta` or
            extracted from parquet metadata).

        Returns
        -------
        wcs : `astropy.wcs.WCS` or `None`
            The deserialized WCS object, or None if not present.
        mjd_mid : `np.ndarray` or `None`
            The observation midpoint times in MJD UTC as a numpy array,
            or None if not present.
        image_column_shapes : `dict` or `None`
            A dictionary mapping column names to their original shapes,
            or None if not present.
        """
        if not meta_dict:
            return None, None, None

        # Extract WCS
        wcs = None
        if "wcs" in meta_dict:
            wcs = deserialize_wcs(meta_dict["wcs"])

        # Extract observation times (check mjd_utc_mid first for compatibility)
        # Normalize to np.array for consistent type across read_table and read_table_chunks
        mjd_mid = None
        if "mjd_utc_mid" in meta_dict:
            mjd_mid = np.array(meta_dict["mjd_utc_mid"])
        elif "mjd_mid" in meta_dict:
            mjd_mid = np.array(meta_dict["mjd_mid"])

        # Extract image column shapes
        image_column_shapes = meta_dict.get("image_column_shapes")

        logger.debug(
            f"Parsed metadata: wcs={wcs is not None}, "
            f"mjd_mid len={len(mjd_mid) if mjd_mid is not None else 0}, "
            f"image_columns={list(image_column_shapes.keys()) if image_column_shapes else []}"
        )

        return wcs, mjd_mid, image_column_shapes

    def _reshape_image_columns(self, image_column_shapes):
        """Reshape image columns back to their original dimensions.

        Parquet serialization (via Astropy) flattens multi-dimensional arrays to 1D.
        This method uses stored shape metadata to restore the original dimensions.

        Parameters
        ----------
        image_column_shapes : `dict` or `None`
            A dictionary mapping column names to their original shapes.
            If None or empty, no reshaping is performed.
        """
        if not image_column_shapes or len(self) == 0:
            return

        for colname, shape in image_column_shapes.items():
            if colname in self.colnames:
                shape = tuple(shape) if isinstance(shape, list) else shape
                logger.debug(f"Reshaping column '{colname}' to {shape}")
                try:
                    new_shape = tuple([len(self)] + list(shape))
                    self.table[colname] = np.reshape(self.table[colname], new_shape)
                except ValueError as e:
                    logger.warning(
                        f"Could not reshape column '{colname}' to {shape}: {e}. "
                        "Shape may have changed or data is incompatible."
                    )

    def remove_column(self, colname):
        """Remove a column from the results table.

        Parameters
        ----------
        colname : `str`
            The name of column to drop.

        Raises
        ------
        Raises a ``KeyError`` if the column does not exist or
        is a required column.
        """
        if colname not in self.table.colnames:
            raise KeyError(f"Column {colname} not found.")
        if colname in self._required_col_names:
            raise KeyError(f"Unable to drop required column {colname}.")
        self.table.remove_column(colname)

    def extend(self, results2):
        """Append the results in a second `Results` object to the current one.

        Parameters
        ----------
        results2 : `Results`
            The data structure containing additional results to add.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.

        Raises
        ------
        Raises a ValueError if the columns of the results do not match.
        """
        # Confirm that, if the current table is non-empty, both tables have matching columns.
        if len(self) > 0 and set(self.colnames) != set(results2.colnames):
            raise ValueError("Column mismatch when merging results")

        self.table = vstack([self.table, results2.table])

        # Combine the statistics (even if track_filtered is False).
        for key in results2.filtered_stats.keys():
            if key in self.filtered:
                self.filtered_stats[key] += results2.filtered_stats[key]
            else:
                self.filtered_stats[key] = results2.filtered_stats[key]

        # When merging the filtered results extend lists with the
        # same key and create new lists for new keys.
        for key in results2.filtered.keys():
            if key in self.filtered:
                self.filtered[key] = vstack([self.filtered[key], results2.filtered[key]])
            else:
                self.filtered[key] = results2.filtered[key]

        return self

    def make_trajectory_list(self):
        """Create a list of ``Trajectory`` objects.

        Returns
        -------
        trajectories : `list[Trajectory]`
            The ``Trajectory`` objects.
        """
        trajectories = [
            Trajectory(
                x=row["x"],
                y=row["y"],
                vx=row["vx"],
                vy=row["vy"],
                flux=row["flux"],
                lh=row["likelihood"],
                obs_count=row["obs_count"],
            )
            for row in self.table
        ]
        return trajectories

    def sort(self, colname, descending=True):
        """Sort the results by a given column.

        Parameters
        ----------
        colname : `str`
            The name of the column to sort by.
        reversdescendinge : `bool`
            Whether to sort in descending order. By default this is true,
            so that the highest likelihoods and num obs are the the top.
            Default: True.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.

        Raises
        ------
        Raises a KeyError if the column is not in the data.
        """
        if colname not in self.table.colnames:
            raise KeyError(f"Column {colname} not found.")
        self.table.sort(colname, reverse=descending)
        return self

    def compute_likelihood_curves(self, filter_obs=True, mask_value=0.0):
        """Create a matrix of likelihood curves where each row has a likelihood
        curve for a single trajectory.

        Parameters
        ----------
        filter_obs : `bool`
            Filter any indices marked as invalid in the 'obs_valid' column.
            Substitutes the value of ``mask_value`` in their place.
        mask_value : `float`
            A floating point value to substitute into the masked entries.
            Commonly used values are 0.0 and np.NAN, which allows filtering
            for some numpy operations.

        Returns
        -------
        lh_matrix : `numpy.ndarray`
            The likleihood curves for each trajectory.

        Raises
        ------
        Raises an IndexError if the necessary columns are missing.
        """
        if "psi_curve" not in self.table.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")
        if "phi_curve" not in self.table.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")

        psi = self.table["psi_curve"]
        phi = self.table["phi_curve"]

        # Create a mask of valid data.
        valid = (phi != 0) & np.isfinite(psi) & np.isfinite(phi)
        if filter_obs and "obs_valid" in self.table.colnames:
            valid = valid & self.table["obs_valid"]

        lh_matrix = np.full(psi.shape, mask_value, dtype=np.float32)
        lh_matrix[valid] = psi[valid] / np.sqrt(phi[valid])
        return lh_matrix

    def _update_likelihood(self):
        """Update the likelihood related trajectory information from the
        psi and phi information. Requires the existence of the columns
        'psi_curve' and 'phi_curve' which can be set with add_psi_phi_data().
        Uses the (optional) 'valid_indices' if it exists.

        This should be called any time that the psi_curve, phi_curve, or
        obs_valid columns are modified.

        Raises
        ------
        Raises an IndexError if the necessary columns are missing.
        """
        num_rows = len(self.table)
        if num_rows == 0:
            return  # Nothing to do for an empty table

        if "psi_curve" not in self.table.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")
        if "phi_curve" not in self.table.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")

        num_times = len(self.table["phi_curve"][0])
        if "obs_valid" in self.table.colnames:
            phi_sum = (self.table["phi_curve"] * self.table["obs_valid"]).sum(axis=1)
            psi_sum = (self.table["psi_curve"] * self.table["obs_valid"]).sum(axis=1)
            num_obs = self.table["obs_valid"].sum(axis=1)
        else:
            phi_sum = self.table["phi_curve"].sum(axis=1)
            psi_sum = self.table["psi_curve"].sum(axis=1)
            num_obs = np.full(num_rows, num_times)

        non_zero = phi_sum != 0
        self.table["likelihood"] = np.zeros((num_rows))
        self.table["likelihood"][non_zero] = psi_sum[non_zero] / np.sqrt(phi_sum[non_zero])
        self.table["flux"] = np.zeros((num_rows))
        self.table["flux"][non_zero] = psi_sum[non_zero] / phi_sum[non_zero]
        self.table["obs_count"] = num_obs

    def add_psi_phi_data(self, psi_array, phi_array, obs_valid=None):
        """Append columns for the psi and phi data and use this to update the
        relevant trajectory information.

        Parameters
        ----------
        psi_array : `numpy.ndarray`
            An array of psi_curves with one for each row.
        phi_array : `numpy.ndarray`
            An array of psi_curves with one for each row.
        obs_valid : `numpy.ndarray`, optional
            An optional array of obs_valid arrays with one for each row.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.

        Raises
        ------
        Raises a ValueError if the input arrays are not the same size as the table
        or a given pair of rows in the arrays are not the same length.
        """
        if len(psi_array) != len(self.table):
            raise ValueError(
                f"Wrong number of psi curves provided. Expected {len(self.table)} rows."
                f" Found {len(psi_array)} rows."
            )
        if len(phi_array) != len(self.table):
            raise ValueError(
                f"Wrong number of phi curves provided. Expected {len(self.table)} rows."
                f" Found {len(phi_array)} rows."
            )
        self.table["psi_curve"] = np.asanyarray(psi_array, dtype=np.float32)
        self.table["phi_curve"] = np.asanyarray(phi_array, dtype=np.float32)

        if obs_valid is not None:
            # Make the data to match.
            if len(obs_valid) != len(self.table):
                raise ValueError(
                    f"Wrong number of obs_valid provided. Expected {len(self.table)} rows."
                    f" Found {len(obs_valid)} rows."
                )
            self.table["obs_valid"] = obs_valid

        # Update the track likelihoods given this new information.
        self._update_likelihood()

        return self

    def update_obs_valid(self, obs_valid, drop_empty_rows=True):
        """Updates or appends the 'obs_valid' column.

        Parameters
        ----------
        obs_valid : `numpy.ndarray`
            An array with one row per results and one column per timestamp
            with Booleans indicating whether the corresponding observation
            is valid.
        drop_empty_rows : `bool`
            Filter the rows without any valid observations.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.

        Raises
        ------
        Raises a ValueError if the input array is not the same size as the table
        or a given pair of rows in the arrays are not the same length.
        """
        if len(obs_valid) != len(self.table):
            raise ValueError(
                f"Wrong number of obs_valid lists provided. Expected {len(self.table)} rows"
                f" Found {len(obs_valid)} rows"
            )
        self.table["obs_valid"] = obs_valid

        # Update the count of valid observations and filter any rows without valid observations.
        self.table["obs_count"] = self.table["obs_valid"].sum(axis=1)
        row_has_obs = self.table["obs_count"] > 0
        if drop_empty_rows and not np.all(row_has_obs):
            self.filter_rows(row_has_obs, "no valid observations")

        # Update the track likelihoods given this new information.
        if "psi_curve" in self.colnames and "phi_curve" in self.colnames:
            self._update_likelihood()
        return self

    def is_empty_value(self, colname):
        """Create a Boolean vector indicating whether the entry in each row
        is an 'empty' value (None or anything of length 0). Used to mark or
        filter missing values.

        Parameters
        ----------
        colname : str
            The name of the column to check.

        Returns
        -------
        result : `numpy.ndarray`
            An array of Boolean values indicating whether the result is
            one of the empty values.
        """
        if colname not in self.table.colnames:
            raise KeyError(f"Querying unknown column {colname}")

        # Skip numeric types (integers, floats, etc.)
        result = np.full(len(self.table), False)
        if np.issubdtype(self.table[colname].dtype, np.number):
            return result

        # Go through each entry and check whether it is None or something of length=0.
        for idx, val in enumerate(self.table[colname]):
            if val is None:
                result[idx] = True
            elif hasattr(val, "__len__") and len(val) == 0:
                result[idx] = True
        return result

    def is_image_like(self, colname, max_rows=10):
        """Check whether the column contains image-like data (numpy arrays
        with at least 2 dimensions).

        This method first checks the table metadata for stored image column
        shape information (which preserves shape info through parquet round-trips).
        If not found in metadata, it inspects the actual data.

        Parameters
        ----------
        colname : `str`
            The name of the column to check.
        max_row : `int`
            The maximum number of rows to check before concluding
            that the column is not image-like. Default: 10.
        Returns
        -------
        result : `bool`
            True if the column contains image-like data.
        """
        if colname not in self.table.colnames:
            raise KeyError(f"Querying unknown column {colname}")

        # First check if we have metadata about image column shapes (from a previous save)
        if "image_column_shapes" in self.table.meta:
            if colname in self.table.meta["image_column_shapes"]:
                return True

        # Otherwise, check the actual data for 2D+ arrays
        max_rows = len(self.table) if max_rows is None else min(max_rows, len(self.table))

        # If no rows to check, we can't determine it's image-like
        if max_rows < 1:
            return False

        for idx in range(max_rows):
            entry = self.table[colname][idx]
            if isinstance(entry, np.ndarray) and entry.ndim >= 2:
                return True
        return False

    def filter_rows(self, rows, label=""):
        """Filter the rows in the `Results` to only include those indices
        that are provided in a list of row indices (integers) or marked
        ``True`` in a mask.

        Parameters
        ----------
        rows : `numpy.ndarray`
            Either a Boolean array of the same length as the table
            or list of integer row indices to keep.
        label : `str`
            The label of the filtering stage to use.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.
        """
        logger.info(f"Applying filter={label} to results of size {len(self.table)}.")
        if len(self.table) == 0 or len(rows) == 0:  # Nothing to filter
            self.filtered_stats[label] = self.filtered_stats.get(label, 0)
            return

        # Check if we are dealing with a mask of a list of indices.
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if len(rows) != len(self.table):
                raise ValueError(
                    f"Mask length mismatch. Expected {len(self.table)} rows, but found {len(rows)}."
                )
            mask = rows
        else:
            mask = np.full((len(self.table),), False)
            mask[rows] = True

        # Track the data we have filtered.
        filtered_table = self.table[~mask]
        self.filtered_stats[label] = self.filtered_stats.get(label, 0) + len(filtered_table)
        logger.debug(f"Filter={label} removed {len(filtered_table)} entries.")

        if self.track_filtered:
            if label in self.filtered:
                self.filtered[label] = vstack([self.filtered[label], filtered_table])
            else:
                self.filtered[label] = filtered_table

        # Do the actual filtering.
        self.table = self.table[mask]

        # Return a reference to the current object to allow chaining.
        return self

    def get_filtered(self, label=None):
        """Get the results filtered at a given stage or all stages.

        Parameters
        ----------
        label : `str`
            The filtering stage to use. If no label is provided,
            return all filtered rows.

        Returns
        -------
        results : `astropy.table.Table` or ``None``
            A table with the filtered rows or ``None`` if there are no entries.
        """
        if not self.track_filtered:
            raise ValueError("ResultTable filter tracking not enabled.")

        result = None
        if label is not None:
            # Check if anything was filtered at this stage.
            if label in self.filtered:
                result = self.filtered[label]
        else:
            result = vstack([x for x in self.filtered.values()])

        return result

    def revert_filter(self, label=None, add_column=None):
        """Revert the filtering by re-adding filtered ResultRows.

        Notes
        -----
        Filtered rows are appended to the end of the list. Does not return
        the results to the original ordering.

        Parameters
        ----------
        label : `str`
            The filtering stage to use. If no label is provided,
            revert all filtered rows.
        add_column : `str`
            If not ``None``, add a tracking column with the given name
            that includes the original filtering reason.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.

        Raises
        ------
        ValueError if filtering is not enabled.
        KeyError if label is unknown.
        """
        if not self.track_filtered:
            raise ValueError("ResultTable filter tracking not enabled.")

        # Make a list of labels to revert
        if label is not None:
            if label not in self.filtered:
                raise KeyError(f"Unknown filtered label {label}")
            to_revert = [label]
        else:
            to_revert = list(self.filtered.keys())

        # If we don't have the tracking column yet, add it.
        if add_column is not None and add_column not in self.table.colnames:
            self.table[add_column] = np.full(len(self.table), "", dtype=str)

        # Make a list of tables to merge.
        table_list = [self.table]
        for key in to_revert:
            logger.info(f"Reverting filter={key} with {self.filtered_stats[key]} entries.")

            filtered_table = self.filtered[key]
            if add_column is not None and len(filtered_table) > 0:
                filtered_table[add_column] = [key] * len(filtered_table)
            table_list.append(filtered_table)
            del self.filtered[key]
            del self.filtered_stats[key]
        self.table = vstack(table_list)

        return self

    def _detect_image_columns(self, image_columns=None, max_rows=10):
        """Detect image-like columns and their shapes.

        This method always auto-detects columns with 2D+ numpy arrays by scanning
        up to `max_rows` rows to find a representative non-empty entry for each
        column. Additionally, any columns explicitly listed in `image_columns`
        will be included even if they don't meet the 2D+ criteria (e.g., if they
        are already flattened from a parquet round-trip).

        Parameters
        ----------
        image_columns : `list` of `str`, optional
            Explicit list of column names that should be treated as image-like,
            in addition to any auto-detected columns. Useful for columns that
            may already be flattened from parquet serialization.
        max_rows : `int`
            Maximum number of rows to check for auto-detection.

        Returns
        -------
        image_col_shapes : `dict`
            Dictionary mapping column name to tuple of (original_shape).
        """
        image_col_shapes = {}

        if len(self.table) == 0:
            return image_col_shapes

        # Convert explicit image_columns to a set for fast lookup
        explicit_image_cols = set(image_columns) if image_columns is not None else set()

        max_rows = min(max_rows, len(self.table)) if max_rows is not None else len(self.table)

        # Always check all columns for auto-detection
        for colname in self.table.colnames:
            # Skip required trajectory columns
            if colname in self._required_col_names or colname == "uuid":
                continue

            is_explicit = colname in explicit_image_cols

            # Scan rows until we find a representative non-empty entry
            for idx in range(max_rows):
                entry = self.table[colname][idx]
                if not isinstance(entry, np.ndarray):
                    # Not a numpy array, skip this column
                    break

                # Check if this entry is "empty" (size 0) - if so, try next row
                if entry.size == 0:
                    continue

                # Found a representative entry
                if entry.ndim >= 2:
                    # This is an image-like column (auto-detected)
                    # Convert numpy int types to Python ints for pyarrow serialization
                    image_col_shapes[colname] = tuple(int(s) for s in entry.shape)
                    logger.debug(f"Detected image column '{colname}' with shape {entry.shape}")
                elif is_explicit:
                    # User explicitly specified this as image column
                    # Store shape even if 1D (may be already flattened from parquet)
                    # Convert numpy int types to Python ints for pyarrow serialization
                    image_col_shapes[colname] = tuple(int(s) for s in entry.shape)
                    logger.debug(f"User-specified image column '{colname}' with shape {entry.shape}")
                break  # Found representative entry, move to next column

        return image_col_shapes

    def write_table(
        self,
        filename,
        overwrite=True,
        extra_meta=None,
    ):
        """Write the unfiltered results to a single file.  The file format is automatically
        determined from the file name's suffix which must be one of ".ecsv", ".parquet",
        ".parq", or ".hdf5".  We recommend ".parquet".

        Parameters
        ----------
        filename : `str`
            The name of the result file.  Must have a suffix matching one of ".ecsv",
            ".parquet", ".parq", or ".hdf5".
        overwrite : `bool`
            Overwrite the file if it already exists. [default: True]
        extra_meta : `dict`, optional
            Any additional meta data to save with the table.
        """
        logger.info(f"Saving results to {filename}")

        # Check that we are using a valid file format.
        filepath = Path(filename)
        if filepath.suffix not in self._supported_formats:
            raise ValueError(
                f"Unsupported file type '{filepath.suffix}' " f"use one of {self._supported_formats}."
            )

        # Include optional arguments for hdf5 to avoid a warning.
        if filepath.suffix == ".hdf5":
            kwargs = {
                "path": "__astropy_table__",
                "serialize_meta": True,
            }
        else:
            kwargs = {}

        # Add global meta data that we can retrieve.
        if self.wcs is not None:
            logger.debug("Saving WCS to Results table meta data.")
            self.table.meta["wcs"] = serialize_wcs(self.wcs)
        if self.mjd_mid is not None:
            # Save different format time stamps.
            self.table.meta["mjd_mid"] = self.mjd_mid
            self.table.meta["mjd_utc_mid"] = self.mjd_mid
            self.table.meta["mjd_tai_mid"] = self.mjd_tai_mid

        # Detect and store image column metadata for parquet round-trip
        image_col_shapes = self._detect_image_columns()
        if image_col_shapes:
            self.table.meta["image_column_shapes"] = {
                col: list(shape) for col, shape in image_col_shapes.items()
            }
            logger.debug(f"Storing image column metadata: {list(image_col_shapes.keys())}")

        if extra_meta is not None:
            for key, val in extra_meta.items():
                logger.debug(f"Saving {key} to Results table meta data.")
                self.table.meta[key] = val

        # Write out the table.
        self.table.write(filename, overwrite=overwrite, **kwargs)

    def write_column(self, colname, filename, overwrite=True, is_image=None):
        """Save a single column's data as its own data file. The file
        type is inferred from the filename suffix. Supported formats include
        numpy (.npy), ecsv (.ecsv), parquet (.parq or .parquet), or
        fits (.fits).

        Parameters
        ----------
        colname : `str`
           The name of the column to save.
        filename : `str` or `Path`
            The file name for the ouput file. Must have a suffix matching one of ".npy",
            ".ecsv", ".parquet", ".parq", or ".fits".
        overwrite : `bool`
            Overwrite the file if it already exists.
            Default: True
        is_image : `bool`, optional
            Explicitly specify whether this column contains image-like data.
            If None, auto-detection via `is_image_like()` is used.

        Raises
        ------
        Raises a KeyError if the column is not in the data.
        """
        logger.info(f"Writing {colname} column data to {filename}")
        if colname not in self.table.colnames:
            raise KeyError(f"Column {colname} missing from data.")

        filename = Path(filename)
        if filename.exists() and not overwrite:
            raise FileExistsError(f"File {filename} arleady exists.")

        if filename.suffix == ".npy":
            # Extract and save the column.
            data = np.asarray(self.table[colname])
            np.save(filename, data, allow_pickle=False)
        elif filename.suffix in [".ecsv", ".parq", ".parquet"]:
            # Create a table with just this column.
            single_table = Table({colname: self.table[colname].data})

            # Parquet might fail on some output types, so we
            # try to convert it into a string in that case.
            try:
                single_table.write(filename, overwrite=overwrite)
            except Exception as e:
                logger.debug(f"Failed to write {colname}. Retrying as a string: {e}")
                data = [str(x) for x in single_table[colname].data]
                single_table = Table({colname: data})
                single_table.write(filename, overwrite=overwrite)
        elif filename.suffix == ".fits":
            # Check if this the data is looks like images.
            # Use explicit parameter if provided, otherwise auto-detect.
            if is_image is not None:
                is_img = is_image
            else:
                is_img = self.is_image_like(colname)

            # Create a HDU List and primary header with basic meta data.
            hdul = fits.HDUList()
            pri = fits.PrimaryHDU()
            pri.header["NUMRES"] = len(self.table)
            pri.header["ISIMG"] = is_img
            pri.header["COLNAME"] = colname
            hdul.append(pri)

            if is_img:
                # Create a separate HDU for each entry.
                for idx in range(len(self.table)):
                    img_hdu = fits.CompImageHDU(
                        self.table[colname][idx],
                        compression_type="RICE_1",
                        quantize_level=-0.01,
                    )

                    # If we have the UUID, save that as the meta data.
                    if "uuid" in self.table.colnames:
                        img_hdu.header["uuid"] = self.table["uuid"][idx]

                    img_hdu.name = f"IMG_{idx}"
                    hdul.append(img_hdu)
            else:
                # Create one bin table for the data.
                single_table = Table({colname: self.table[colname].data})
                data_hdu = fits.BinTableHDU(single_table)
                data_hdu.name = "DATA"
                hdul.append(data_hdu)

            hdul.writeto(filename, overwrite=overwrite)
        else:
            raise ValueError(f"Unsupported suffix {filename.suffix}")

    def load_image_row_by_uuid(self, aux_filepath, target_uuid):
        """Lazy load a single image row by UUID from an auxiliary FITS image column file.

        This is a convenience method that wraps the module-level function
        `load_image_row_by_uuid()`. It loads only the requested image without
        reading the entire image column into memory.

        Parameters
        ----------
        aux_filepath : `str` or `Path`
            The path to the auxiliary FITS file containing the image column.
            Must be a FITS file (.fits) written by `write_column()` with
            image-like data.
        target_uuid : `str`
            The UUID of the row to load.

        Returns
        -------
        image : `numpy.ndarray`
            The image data for the row with the matching UUID.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not an image column FITS file (ISIMG=False).
        KeyError
            If no row with the given UUID is found in the file.

        See Also
        --------
        load_image_row_by_uuid : Module-level function with same functionality.

        Examples
        --------
        >>> results = Results.read_table("results.parquet")
        >>> # Load a single stamp image by UUID without loading the entire column
        >>> image = results.load_image_row_by_uuid("results_stamps.fits", results["uuid"][0])
        >>> print(image.shape)
        (21, 21)
        """
        return load_image_row_by_uuid(aux_filepath, target_uuid)

    def load_column(self, filename, colname=None):
        """Read in a file containing a single column as its own data file.
        The file type is inferred from the filename suffix. Supported formats
        include numpy (.npy), ecsv (.ecsv), parquet (.parq or .parquet), or
        fits (.fits).

        Parameters
        ----------
        filename : `str` or `Path`
            The file name for the ouput file. Must have a suffix matching one of ".npy",
            ".ecsv", ".parquet", ".parq", or ".fits".
        colname : `str`
           The name of the column to save. If None this is automatically
           inferred from the data.

        Raises
        ------
        Raises a FileNotFoundError if the file does not exist.
        Raises a ValueError if column is of the wrong length.
        """
        logger.info(f"Loading column data from {filename} as {colname}")
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"{filename} not found for load.")

        if filename.suffix == ".npy":
            data = np.load(filename, allow_pickle=False)
        elif filename.suffix in [".ecsv", ".parq", ".parquet"]:
            # Create a table with just this column.
            single_table = Table.read(filename)

            if len(single_table.colnames) != 1:
                raise ValueError(f"Expected one column. Found: {single_table.colnames}")
            single_col = single_table.colnames[0]
            if colname is None:
                colname = single_col

            data = single_table[single_col].data
        elif filename.suffix == ".fits":
            with fits.open(filename) as hdul:
                num_rows = hdul[0].header["NUMRES"]
                is_img = hdul[0].header["ISIMG"]
                if colname is None:
                    colname = hdul[0].header["COLNAME"]

                if is_img:
                    # Extract each image from its own layer.
                    data = []
                    for idx in range(num_rows):
                        img_layer = hdul[f"IMG_{idx}"]
                        data.append(img_layer.data.astype(np.single))
                else:
                    # Extract the column from the data layer.
                    single_table = Table(hdul["DATA"].data)
                    data = single_table[hdul[0].header["COLNAME"]].data

        if len(data) != len(self.table):
            raise ValueError(
                f"Error loading {filename}: expected {len(self.table)} entries, but found {len(data)}."
            )

        self.table[colname] = data

    def write_filtered_stats(self, filename):
        """Write out the filtering statistics to a human readable CSV file.

        Parameters
        ----------
        filename : `str`
            The name of the file to write.
        """
        logger.info(f"Saving results filter statistics to {filename}.")
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["unfiltered", len(self.table)])
            for key, value in self.filtered_stats.items():
                writer.writerow([key, value])

    @classmethod
    def from_trajectory_file(cls, filename, track_filtered=False):
        """Load the results from a saved Trajectory file.

        Parameters
        ----------
        filename : `str`
            The file name for the input file.
        track_filtered : `bool`
            Whether to track (save) the filtered trajectories. This will use
            more memory and is recommended only for analysis.

        Raises
        ------
        Raises a FileNotFoundError if the file does not exist.
        """
        logger.info(f"Loading result trajectories from {filename}")
        if not Path(filename).is_file():
            raise FileNotFoundError(f"{filename} not found for load.")

        trj_list = Results.load_trajectory_file(filename)
        return cls.from_trajectories(trj_list, track_filtered)


def load_image_row_by_uuid(filename, target_uuid):
    """Lazy load a single image row by UUID from an auxiliary FITS image column file.

    This function loads only the requested image without reading the entire
    image column into memory. It is designed for on-demand access to individual
    images stored in auxiliary FITS files created by `Results.write_column()`.

    Parameters
    ----------
    filename : `str` or `Path`
        The path to the auxiliary FITS file containing the image column.
        Must be a FITS file (.fits) written by `Results.write_column()` with
        image-like data.
    target_uuid : `str`
        The UUID of the row to load.

    Returns
    -------
    image : `numpy.ndarray`
        The image data for the row with the matching UUID.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not an image column FITS file (ISIMG=False).
    KeyError
        If no row with the given UUID is found in the file.

    Examples
    --------
    >>> # Load a single image by UUID without loading the entire column
    >>> image = load_image_row_by_uuid("results_stamps.fits", "abc123def456")
    >>> print(image.shape)
    (21, 21)
    """
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"{filename} not found.")

    if filename.suffix != ".fits":
        raise ValueError(f"Expected a FITS file, got {filename.suffix}")

    with fits.open(filename, memmap=True) as hdul:
        # Verify this is an image column file
        if not hdul[0].header.get("ISIMG", False):
            raise ValueError(
                f"{filename} is not an image column FITS file (ISIMG=False). "
                "Use load_column() for non-image data."
            )

        num_rows = hdul[0].header["NUMRES"]

        # Search through image HDUs for the matching UUID
        for idx in range(num_rows):
            hdu_name = f"IMG_{idx}"
            if hdu_name not in hdul:
                continue

            img_hdu = hdul[hdu_name]
            hdu_uuid = img_hdu.header.get("uuid")

            if hdu_uuid == target_uuid:
                # Found the matching row - load and return the image data
                return img_hdu.data.astype(np.single)

        # UUID not found
        raise KeyError(f"UUID '{target_uuid}' not found in {filename}")


def write_results_to_files_destructive(
    filename,
    results,
    extra_meta=None,
    separate_col_files=None,
    drop_columns=None,
    overwrite=True,
    image_columns=None,
):
    """Write the results to one or more files.

    Note
    ----
    This function modifies the `results` object in-place to drop columns
    as they are written out. This avoid creating unnecessary copies of the data, but
    destroys the original data.

    Parameters
    ----------
    filename : `str` or `Path`
        The name of the file to output the results to.
    results : `Results`
        The results to output. These results are modified in-place to drop
        columns as they are written out.
    extra_meta : `dict`, optional
        Any additional meta data to save with the table. This is saved in the
        table's meta data and can be retrieved later.
    separate_col_files : `list` of `str`, optional
        A list of column name regex patterns to write to separate files. If None, no separate files
        are written.
    drop_columns : `list` of `str`, optional
        A list of column names to skip when outputting results. If None, no columns are skipped.
    overwrite : `bool`, optional
        If True, overwrite existing files. If False, do not overwrite existing files.
        Defaults to True.
    image_columns : `list` of `str`, optional
        A list of column names that contain image-like data. These columns will be saved
        as FITS files when written separately. If None, auto-detection via `is_image_like()`
        is used.
    """
    if not filename:
        raise ValueError("No filename provided for outputting results.")
    filepath = Path(filename)
    if filepath.exists() and not overwrite:
        raise ValueError(f"File {filepath} already exists. Not overwriting.")

    # Capture image column shapes BEFORE removing any columns.
    # This ensures the metadata is preserved even after columns are written to aux files.
    image_col_shapes = results._detect_image_columns(image_columns)

    # Write out the auxiliary columns to their own files and drop them from the main table.
    if separate_col_files is not None:
        # Treat the separate_col_files as a list of regex and find all matching columns.
        all_separate_cols = []
        for pattern in separate_col_files:
            regex = re.compile(pattern)
            matching_cols = [col for col in results.colnames if regex.fullmatch(col)]
            all_separate_cols.extend(matching_cols)
        separate_col_files = all_separate_cols

        # For each column that matched, write it out to its own file and drop it from the main table.
        for col in separate_col_files:
            # If the column is an image-like data type, save it as a FITS file. Otherwise, save
            # it using the same extension as the main file.
            is_image = col in image_col_shapes
            if is_image:
                # If the column is an image, save it as a FITS file.
                col_file = filepath.with_name(filepath.stem + f"_{col}.fits")
            else:
                col_file = filepath.with_name(filepath.stem + f"_{col}" + filepath.suffix)

            logger.info(f"Saving column {col} to {col_file}")
            results.write_column(col, col_file, overwrite=overwrite, is_image=is_image)
            results.remove_column(col)

    # Drop any other columns specified.
    if drop_columns is not None:
        for col in drop_columns:
            if col not in results.colnames:
                logger.debug(f"Column {col} not found in results. Skipping.")
            else:
                results.remove_column(col)

    # Add the dropped column information to the meta data.
    if extra_meta is None:
        extra_meta = {}
    extra_meta["separate_col_files"] = separate_col_files
    extra_meta["dropped_columns"] = drop_columns

    # Preserve the image_column_shapes captured before columns were removed.
    # This allows the shapes to be stored in metadata even for columns now in aux files.
    if image_col_shapes:
        extra_meta["image_column_shapes"] = {col: list(shape) for col, shape in image_col_shapes.items()}

    # Write the remaining data from the results to the main file.
    logger.info(f"Saving results table to {filepath}")
    results.write_table(filepath, overwrite=overwrite, extra_meta=extra_meta)


def compute_trajectory_rates(
    results, wcs=None, default_pixel_scale=5.5555555555556e-05, requested_columns=None
):
    """Compute RA and Dec rates (arcsec/day) for trajectories.

    Parameters
    ----------
    results : `Results` or `astropy.table.Table`
        The results data containing at least `x`, `y`, `vx`, `vy`.
    wcs : `astropy.wcs.WCS`, optional
        The WCS to use for pixel-to-world conversion. If not provided,
        it will attempt to extract it from `results.wcs`. If neither is
        available, it falls back to a simple pixel scale approximation.
    default_pixel_scale : `float`
        The default pixel scale in deg/pixel to use if WCS is absent.
        Default is ~0.2"/pixel.
    requested_columns : `list`, optional
        A list of column names to include in the output table. If None,
        only `uuid` and the computed rate columns will be returned.

    Returns
    -------
    rates_table : `astropy.table.Table`
        A table containing the requested columns along with
        `vx_ra_rate_arcsec_per_day` and `vy_dec_rate_arcsec_per_day`.
    """
    if hasattr(results, "table"):
        table = results.table
        wcs = wcs if wcs is not None else results.wcs
    else:
        table = results

    # ensure requested_columns exist
    if requested_columns is None:
        requested_columns = ["uuid"]
    else:
        requested_columns = list(set(requested_columns + ["uuid"]))

    out_dict = {}
    for col in requested_columns:
        if col in table.colnames:
            out_dict[col] = table[col]

    if len(table) == 0:
        # Return empty table with correct columns
        out_dict["vx_ra_rate_arcsec_per_day"] = np.array([], dtype=float)
        out_dict["vy_dec_rate_arcsec_per_day"] = np.array([], dtype=float)
        return Table(out_dict)

    x = table["x"]
    y = table["y"]
    vx = table["vx"]
    vy = table["vy"]

    # time step for velocity projection
    dt = 1.0  # days

    if wcs is not None:
        # Use proper WCS forward projection
        ra1, dec1 = wcs.pixel_to_world_values(x, y)
        ra2, dec2 = wcs.pixel_to_world_values(x + vx * dt, y + vy * dt)

        # d_ra needs to be multiplied by cos(mean_dec)
        mean_dec_rad = np.radians(0.5 * (dec1 + dec2))
        d_ra = (ra2 - ra1) * np.cos(mean_dec_rad)  # degrees on sky
        d_dec = dec2 - dec1  # degrees

        ra_rate = d_ra * 3600.0 / dt
        dec_rate = d_dec * 3600.0 / dt
    else:
        # Fallback to simple scaling (assumes Global coordinate system +X -> -RA)
        pixel_scale_arcsec = default_pixel_scale * 3600.0
        # If no global_dec is available, cos_dec=1.0. This is an approximation.
        ra_rate = -vx * pixel_scale_arcsec
        dec_rate = vy * pixel_scale_arcsec

    out_dict["vx_ra_rate_arcsec_per_day"] = ra_rate
    out_dict["vy_dec_rate_arcsec_per_day"] = dec_rate

    # Method 1 & 2: Endpoint rates for global/img RA/Dec curves
    mjd_mid = None
    if hasattr(results, "mjd_mid") and results.mjd_mid is not None:
        mjd_mid = np.array(results.mjd_mid)
    elif "mjd_utc_mid" in table.meta:
        mjd_mid = np.array(table.meta["mjd_utc_mid"])
    elif "mjd_mid" in table.meta:
        mjd_mid = np.array(table.meta["mjd_mid"])

    # If times are available, process array columns
    if mjd_mid is not None:
        # Determine valid observations mask
        if "obs_valid" in table.colnames:
            valid_mask = table["obs_valid"]
            if valid_mask.ndim == 1 and len(valid_mask) > 0 and isinstance(valid_mask[0], (list, np.ndarray)):
                valid_mask = np.array(list(valid_mask))
        else:
            raise ValueError("obs_valid column not found in results")

        mjd_first_arr = np.full(len(table), np.nan)
        mjd_last_arr = np.full(len(table), np.nan)
        mjd_valid_arr = np.empty(len(table), dtype=object)
        for i in range(len(table)):
            valid_idx = np.where(valid_mask[i])[0]
            mjd_valid_arr[i] = mjd_mid[valid_idx]
            if len(valid_idx) > 0:
                mjd_first_arr[i] = mjd_mid[valid_idx[0]]
                mjd_last_arr[i] = mjd_mid[valid_idx[-1]]
            if len(valid_idx) != table["obs_count"][i]:
                raise ValueError("obs_count in table does not match computed obs_count")
            
        out_dict["mjd_mid_first"] = mjd_first_arr
        out_dict["mjd_mid_last"] = mjd_last_arr
        out_dict["obs_mjd"] = mjd_valid_arr

        obs_cnt_arr = np.sum(valid_mask, axis=1)
        # Check that obs_cnt matches the one in the table
        if "obs_count" in table.colnames:
            if not np.array_equal(obs_cnt_arr, table["obs_count"]):
                logger.warning("obs_count in table does not match computed obs_count")
        out_dict["obs_count"] = obs_cnt_arr
        if len(mjd_mid) > 0:
            out_dict["obs_valid_ratio"] = obs_cnt_arr / len(mjd_mid)
        else:
            out_dict["obs_valid_ratio"] = np.zeros(len(table))

        def _compute_array_endpoint_rate(ra_col, dec_col):
            # Evaluate endpoint rates for (ra_col, dec_col) across valid indices
            rate_ra = np.full(len(table), np.nan)
            rate_dec = np.full(len(table), np.nan)

            ra_arr = table[ra_col]
            dec_arr = table[dec_col]

            for i in range(len(table)):
                valid_idx = np.where(valid_mask[i])[0]
                if len(valid_idx) < 2:
                    continue

                idx_first = valid_idx[0]
                idx_last = valid_idx[-1]

                dt_days = mjd_mid[idx_last] - mjd_mid[idx_first]
                if dt_days <= 0:
                    continue

                r1 = ra_arr[i][idx_first] if ra_arr.ndim == 1 else ra_arr[i, idx_first]
                r2 = ra_arr[i][idx_last] if ra_arr.ndim == 1 else ra_arr[i, idx_last]
                d1 = dec_arr[i][idx_first] if dec_arr.ndim == 1 else dec_arr[i, idx_first]
                d2 = dec_arr[i][idx_last] if dec_arr.ndim == 1 else dec_arr[i, idx_last]

                mean_dec_rad = np.radians(0.5 * (d1 + d2))
                d_ra = (r2 - r1) * np.cos(mean_dec_rad)
                d_dec = d2 - d1

                rate_ra[i] = (d_ra / dt_days) * 3600.0
                rate_dec[i] = (d_dec / dt_days) * 3600.0

            return rate_ra, rate_dec

        if "global_ra" in table.colnames and "global_dec" in table.colnames:
            out_dict["global_ra_rate_arcsec_per_day"], out_dict["global_dec_rate_arcsec_per_day"] = (
                _compute_array_endpoint_rate("global_ra", "global_dec")
            )

        if "img_ra" in table.colnames and "img_dec" in table.colnames:
            out_dict["img_ra_rate_arcsec_per_day"], out_dict["img_dec_rate_arcsec_per_day"] = (
                _compute_array_endpoint_rate("img_ra", "img_dec")
            )

    return Table(out_dict)


def batch_compute_trajectory_rates(
    input_filename,
    output_filename,
    chunk_size=100000,
    wcs=None,
    default_pixel_scale=5.5555555555556e-05,
    requested_columns=None,
):
    """Batch process a KBMOD result file and output a rates table.

    Reads the input file in chunks, computes RA/Dec rates for each trajectory,
    and writes the resulting rates table to an output parquet file.

    Parameters
    ----------
    input_filename : `str`
        Path to the input file (e.g., `.search.parquet`).
        Must be a parquet file to support chunking.
    output_filename : `str`
        Path to the output parquet file where the rates table will be written.
    chunk_size : `int`
        Number of rows to process at a time.
    wcs : `astropy.wcs.WCS`, optional
        Global WCS for pixel conversion.
    default_pixel_scale : `float`
        Default pixel scale if WCS is absent.
    requested_columns : `list`, optional
        List of additional columns to pass through to the output table.
    """
    import pyarrow as pa

    logger.info(f"Computing trajectory rates from {input_filename} in chunks of {chunk_size}")

    first_chunk = True
    writer = None

    for results_chunk in Results.read_table_chunks(input_filename, chunk_size=chunk_size):
        chunk_wcs = wcs if wcs is not None else results_chunk.wcs
        rates_table = compute_trajectory_rates(
            results_chunk,
            wcs=chunk_wcs,
            default_pixel_scale=default_pixel_scale,
            requested_columns=requested_columns,
        )

        # Convert to pyarrow
        batch_pa = pa.Table.from_pandas(rates_table.to_pandas())

        # add our filename to the table
        batch_pa = batch_pa.append_column("search_filename", pa.array([str(input_filename)] * len(batch_pa)))

        if first_chunk:
            writer = pq.ParquetWriter(output_filename, batch_pa.schema)
            first_chunk = False

        writer.write_table(batch_pa)

    if writer is not None:
        writer.close()
    else:
        logger.warning(f"No data found in {input_filename} to compute rates.")


def aggregate_rates_from_directories(
    directories,
    output_filename,
    chunk_size=100000,
    default_pixel_scale=5.5555555555556e-05,
    requested_columns=None,
):
    """Aggregate rates from multiple directories into a single output parquet file.

    Finds all `search.parquet` files in the given directories, attempts to lazy-load
    their corresponding sharded WorkUnit to extract the WCS, and computes the trajectory
    rates. All results, including a `search_filename` column, are combined into a single
    output table.

    Parameters
    ----------
    directories : `list` of `str`
        List of directory paths to scan.
    output_filename : `str`
        Path to the shared parquet file to output the combined rates table.
    chunk_size : `int`
        Number of rows to process at a time.
    default_pixel_scale : `float`
        Default pixel scale if WCS is absent.
    requested_columns : `list`, optional
        List of additional columns to pass through to the output table.
    """
    import pyarrow as pa
    from kbmod.work_unit import WorkUnit
    import tqdm

    logger.info(f"Aggregating trajectory rates from {directories} into {output_filename}")

    first_chunk = True
    writer = None

    if isinstance(directories, (str, Path)):
        directories = [directories]

    for directory in directories:
        directory_path = Path(directory)

        # Scan for search.parquet files (using standard glob matching)
        for result_path in tqdm.tqdm(
            directory_path.rglob("*.search.parquet"), desc="Processing result files"
        ):
            logger.info(f"Processing matched result file: {result_path}")

            # Assume workunit has the same stem and is a fits file in the same directory
            # For instance: obj.search.parquet -> obj.fits
            wu_filename = result_path.name.replace(".search.parquet", "")
            wu_path = result_path.parent / wu_filename

            wcs = None
            if wu_path.exists():
                try:
                    wu = WorkUnit.from_sharded_fits(wu_path.name, str(wu_path.parent), lazy=True)
                    wcs = wu.wcs
                    logger.debug(f"Successfully loaded WCS from sharded WorkUnit {wu_path}")
                except Exception as e:
                    logger.debug(f"Failed to load WCS from WorkUnit {wu_path}: {e}. Trying fallback.")

            if wcs is None:
                if ".collection." in result_path.name:
                    col_filename = result_path.name[: result_path.name.find(".collection.")] + ".collection"
                else:
                    col_filename = result_path.name.replace(".search.parquet", ".collection")
                col_path = result_path.parent / col_filename

                if col_path.exists():
                    try:
                        from kbmod.image_collection import ImageCollection

                        ic = ImageCollection.read(str(col_path))
                        wcs = ic.get_global_wcs(auto_fit=True)
                        logger.info(f"Loaded WCS fallback from via ImageCollection {col_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load WCS from ImageCollection {col_path}: {e}")
                else:
                    logger.warning(
                        f"Could not find expected WorkUnit {wu_path} or ImageCollection {col_path} for {result_path}"
                    )

            # Compute rates chunk by chunk
            try:
                for results_chunk in Results.read_table_chunks(str(result_path), chunk_size=chunk_size):
                    chunk_wcs = wcs if wcs is not None else results_chunk.wcs
                    rates_table = compute_trajectory_rates(
                        results_chunk,
                        wcs=chunk_wcs,
                        default_pixel_scale=default_pixel_scale,
                        requested_columns=requested_columns,
                    )

                    # Convert to pyarrow
                    batch_pa = pa.Table.from_pandas(rates_table.to_pandas())

                    # Add search_filename column for tracking
                    batch_pa = batch_pa.append_column(
                        "search_filename", pa.array([str(result_path)] * len(batch_pa))
                    )

                    if first_chunk:
                        writer = pq.ParquetWriter(output_filename, batch_pa.schema)
                        first_chunk = False

                    writer.write_table(batch_pa)
            except Exception as e:
                logger.error(f"Error processing chunks for {result_path}: {e}")

    if writer is not None:
        writer.close()
        logger.info(f"Finished aggregating rates to {output_filename}")
    else:
        logger.warning("No search.parquet files found or no data to write.")
