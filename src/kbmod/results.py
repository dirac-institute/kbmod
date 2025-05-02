"""Results is a column-based data structure for tracking results with additional global data
and helper functions for filtering and maintaining consistency between different attributes in each row.
"""

import copy
import csv
import logging
import numpy as np
import uuid

from astropy.table import Column, Table, vstack
from astropy.time import Time
from pathlib import Path

from kbmod.trajectory_utils import trajectories_to_dict
from kbmod.search import Trajectory
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
            self.table = data.copy()
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
        # Create dictionaries from the Trajectories.
        input_d = trajectories_to_dict(trajectories)

        # Check for any missing columns and fill in the default value.
        for col in cls.required_cols:
            if col[0] not in input_d:
                input_d[col[0]] = [col[2]] * len(trajectories)

        # Create the table and add the unfiltered (and filtered) results.
        results = Results(input_d, track_filtered=track_filtered)
        return results

    @classmethod
    def read_table(cls, filename, track_filtered=False):
        """Read the ResultList from a table file. The file format is automatically
        determined from the file name's suffix which must be one of ".ecsv",
        ".parquet", ".parq", or ".hdf5".

        Parameters
        ----------
        filename : `str`
            The name of the file to load.
        track_filtered : `bool`
            Indicates whether the object should track future filtered points.

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

        # Check if we have stored a global WCS.
        if "wcs" in data.meta:
            wcs = deserialize_wcs(data.meta["wcs"])
        else:
            wcs = None

        # Create the results object.
        results = Results(data, track_filtered=track_filtered, wcs=wcs)

        # Check if we have a list of observed times.
        if "mjd_utc_mid" in data.meta:
            results.set_mjd_utc_mid(np.array(data.meta["mjd_utc_mid"]))
        elif "mjd_mid" in data.meta:
            results.set_mjd_utc_mid(np.array(data.meta["mjd_mid"]))

        return results

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

    def mask_based_on_invalid_obs(self, input_mat, mask_value):
        """Mask the entries in a given input matrix based on the invalid observations
        in the results. If an observation in result i, time t is invalid, then the corresponding
        entry input_mat[i][t] will be masked. This helper function is used when computing
        statistics on arrays of information.

        The input should be N x T where N is the number of results and T is the number of time steps.

        Parameters
        ----------
        input_mat : `numpy.ndarray`
            An N x T input matrix.
        mask_value : any
            The value to subsitute into the input array.

        Returns
        -------
        result : `numpy.ndarray`
            An N x T output matrix where ``result[i][j]`` is ``input_mat[i][j]`` if
            result ``i``, timestep ``j`` is valid and ``mask_value`` otherwise.

        Raises
        ------
        Raises a ``ValueError`` if the array sizes do not match.
        """
        if len(input_mat) != len(self.table):
            raise ValueError(f"Incorrect input matrix dimensions.")
        masked_mat = np.copy(input_mat)

        # If we do have validity information, use it to do the mask.
        if "obs_valid" in self.table.colnames:
            if input_mat.shape[1] != self.table["obs_valid"].shape[1]:
                raise ValueError(f"Incorrect input matrix dimensions.")
            masked_mat[~self.table["obs_valid"]] = mask_value
        return masked_mat

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

        # Add global meta data that we can retrieve.
        if self.wcs is not None:
            logger.debug("Saving WCS to Results table meta data.")
            self.table.meta["wcs"] = serialize_wcs(self.wcs)
        if self.mjd_mid is not None:
            # Save different format time stamps.
            self.table.meta["mjd_mid"] = self.mjd_mid
            self.table.meta["mjd_utc_mid"] = self.mjd_mid
            self.table.meta["mjd_tai_mid"] = self.mjd_tai_mid

        if extra_meta is not None:
            for key, val in extra_meta.items():
                logger.debug(f"Saving {key} to Results table meta data.")
                self.table.meta[key] = val

        # Write out the table.
        self.table.write(filename, overwrite=overwrite)

    def write_column(self, colname, filename):
        """Save a single column's data as a numpy data file.

        Parameters
        ----------
        colname : `str`
           The name of the column to save.
        filename : `str`
            The file name for the ouput file.

        Raises
        ------
        Raises a KeyError if the column is not in the data.
        """
        logger.info(f"Writing {colname} column data to {filename}")
        if colname not in self.table.colnames:
            raise KeyError(f"Column {colname} missing from data.")

        # Save the column.
        data = np.asarray(self.table[colname])
        np.save(filename, data, allow_pickle=False)

    def load_column(self, filename, colname):
        """Read in a file containing a single column as numpy data
        and join it into the table. The column must be the same length
        as the current table.

        Parameters
        ----------
        filename : `str`
            The file name to read.
        colname : `str`
           The name of the column in which to save the data.

        Raises
        ------
        Raises a FileNotFoundError if the file does not exist.
        Raises a ValueError if column is of the wrong length.
        """
        logger.info(f"Loading column data from {filename} as {colname}")
        if not Path(filename).is_file():
            raise FileNotFoundError(f"{filename} not found for load.")
        data = np.load(filename, allow_pickle=False)

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
