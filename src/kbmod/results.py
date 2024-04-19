"""Results is a column-based data structure for tracking results with additional global data
and helper functions for filtering and maintaining consistency between different attributes in each row.
"""

import numpy as np
import os.path as ospath
from pathlib import Path

from astropy.table import Table, vstack

from kbmod.file_utils import FileUtils
from kbmod.trajectory_utils import make_trajectory, trajectory_from_np_object
from kbmod.search import Trajectory


class Results:
    """This class stores a collection of related data from all of the kbmod results.

    At a minimum it contains columns for the trajectory information:
    (x, y, vx, vy, likelihood, flux, obs_count)
    but additional columns can be added as needed.

    Attributes
    ----------
    table : `astropy.table.Table`
        The stored results data.
    track_filtered : `bool`
        Whether to track (save) the filtered trajectories. This will use
        more memory and is recommended only for analysis.
    filtered : `dict`
        A dictionary mapping a string of filtered name to a table
        of the results removed by that filter.
    """

    _required_cols = ["x", "y", "vx", "vy", "likelihood", "flux", "obs_count"]

    def __init__(self, trajectories, track_filtered=False):
        """Create a ResultTable class.

        Parameters
        ----------
        trajectories : `list[Trajectory]`
            A list of trajectories to include in these results.
        track_filtered : `bool`
            Whether to track (save) the filtered trajectories. This will use
            more memory and is recommended only for analysis.
        """
        # Set up information to track which row is filtered at which round.
        self.track_filtered = track_filtered
        self.filtered = {}

        # Create dictionaries for the required columns.
        input_d = {}
        invalid_d = {}
        for col in self._required_cols:
            input_d[col] = []
            invalid_d[col] = []

        # Add the valid trajectories to the table. If we are tracking filtered
        # data, add invalid trajectories to the invalid_d dictionary.
        for trj in trajectories:
            if trj.valid:
                input_d["x"].append(trj.x)
                input_d["y"].append(trj.y)
                input_d["vx"].append(trj.vx)
                input_d["vy"].append(trj.vy)
                input_d["likelihood"].append(trj.lh)
                input_d["flux"].append(trj.flux)
                input_d["obs_count"].append(trj.obs_count)
            elif track_filtered:
                invalid_d["x"].append(trj.x)
                invalid_d["y"].append(trj.y)
                invalid_d["vx"].append(trj.vx)
                invalid_d["vy"].append(trj.vy)
                invalid_d["likelihood"].append(trj.lh)
                invalid_d["flux"].append(trj.flux)
                invalid_d["obs_count"].append(trj.obs_count)

        self.table = Table(input_d)
        if track_filtered:
            self.filtered["invalid_trajectory"] = Table(invalid_d)

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
    def colnames(self):
        return self.table.colnames

    @classmethod
    def from_table(cls, data, track_filtered=False):
        """Extract data from an astropy Table with the minimum trajectory information.

        Parameters
        ----------
        data : `astropy.table.Table`
            The input data.
        track_filtered : `bool`
            Indicates whether to track future filtered points.

        Raises
        ------
        Raises a KeyError if any required columns are missing.
        """
        # Check that the minimum information is present.
        for col in cls._required_cols:
            if col not in data.colnames:
                raise KeyError(f"Column {col} missing from input data.")

        # Create an empty Results object and append the data table.
        results = Results([], track_filtered=track_filtered)
        results.table = data

        return results

    @classmethod
    def from_dict(cls, input_dict, track_filtered=False):
        """Extract data from a dictionary with the minimum trajectory information.

        Parameters
        ----------
        input_dict : `dict`
            The input data.
        track_filtered : `bool`
            Indicates whether to track future filtered points.

        Raises
        ------
        Raises a KeyError if any required columns are missing.
        """
        return cls.from_table(Table(input_dict))

    @classmethod
    def read_table(cls, filename, track_filtered=False):
        """Read the ResultList from a table file.

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
        if not Path(filename).is_file():
            raise FileNotFoundError(f"File {filename} not found.")
        data = Table.read(filename)
        return Results.from_table(data, track_filtered=track_filtered)

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
        if set(self.colnames) != set(results2.colnames):
            raise ValueError("Column mismatch when merging results")

        self.table = vstack([self.table, results2.table])

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
            make_trajectory(
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

    def compute_likelihood_curves(self, filter_indices=True, mask_value=0.0):
        """Create a matrix of likelihood curves where each row has a likelihood
        curve for a single trajectory.

        Parameters
        ----------
        filter_indices : `bool`
            Filter any indices marked as invalid in the 'index_valid' column.
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
        if filter_indices and "index_valid" in self.table.colnames:
            valid = valid & self.table["index_valid"]

        lh_matrix = np.full(psi.shape, mask_value)
        lh_matrix[valid] = np.divide(psi[valid], np.sqrt(phi[valid]))
        return lh_matrix

    def _update_likelihood(self):
        """Update the likelihood related trajectory information from the
        psi and phi information. Requires the existence of the columns
        'psi_curve' and 'phi_curve' which can be set with add_psi_phi_data().
        Uses the (optional) 'valid_indices' if it exists.

        This should be called any time that the psi_curve, phi_curve, or
        index_valid columns are modified.

        Raises
        ------
        Raises an IndexError if the necessary columns are missing.
        """
        if "psi_curve" not in self.table.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")
        if "phi_curve" not in self.table.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")

        num_rows = len(self.table)
        num_times = len(self.table["phi_curve"][0])
        if "index_valid" in self.table.colnames:
            phi_sum = (self.table["phi_curve"] * self.table["index_valid"]).sum(axis=1)
            psi_sum = (self.table["psi_curve"] * self.table["index_valid"]).sum(axis=1)
            num_obs = self.table["index_valid"].sum(axis=1)
        else:
            phi_sum = self.table["phi_curve"].sum(axis=1)
            psi_sum = self.table["psi_curve"].sum(axis=1)
            num_obs = np.full((num_rows, 1), num_times)

        non_zero = phi_sum != 0
        self.table["likelihood"] = np.zeros((num_rows))
        self.table["likelihood"][non_zero] = psi_sum[non_zero] / np.sqrt(phi_sum[non_zero])
        self.table["flux"] = np.zeros((num_rows))
        self.table["flux"][non_zero] = psi_sum[non_zero] / phi_sum[non_zero]
        self.table["obs_count"] = num_obs

    def add_psi_phi_data(self, psi_array, phi_array, index_valid=None):
        """Append columns for the psi and phi data and use this to update the
        relevant trajectory information.

        Parameters
        ----------
        psi_array : `numpy.ndarray`
            An array of psi_curves with one for each row.
        phi_array : `numpy.ndarray`
            An array of psi_curves with one for each row.
        index_valid : `numpy.ndarray`, optional
            An optional array of index_valid arrays with one for each row.

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
            raise ValueError("Wrong number of psi curves provided.")
        if len(phi_array) != len(self.table):
            raise ValueError("Wrong number of phi curves provided.")
        self.table["psi_curve"] = psi_array
        self.table["phi_curve"] = phi_array

        if index_valid is not None:
            # Make the data to match.
            if len(index_valid) != len(self.table):
                raise ValueError("Wrong number of index_valid lists provided.")
            self.table["index_valid"] = index_valid

        # Update the track likelihoods given this new information.
        self._update_likelihood()

        return self

    def update_index_valid(self, index_valid):
        """Updates or appends the 'index_valid' column.

        Parameters
        ----------
        index_valid : `numpy.ndarray`
            An array with one row per results and one column per timestamp
            with Booleans indicating whether the corresponding observation
            is valid.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.

        Raises
        ------
        Raises a ValueError if the input array is not the same size as the table
        or a given pair of rows in the arrays are not the same length.
        """
        if len(index_valid) != len(self.table):
            raise ValueError("Wrong number of index_valid lists provided.")
        self.table["index_valid"] = index_valid

        # Update the track likelihoods given this new information.
        self._update_likelihood()
        return self

    def filter_mask(self, mask, label=None):
        """Filter the rows in the ResultTable to only include those indices
        that are marked True in the mask.

        Parameters
        ----------
        mask : `list` or `numpy.ndarray`
            A list the same length as the table with True/False indicating
            which row to keep.
        label : `str`
            The label of the filtering stage to use. Only used if
            we keep filtered trajectories.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.
        """
        if self.track_filtered:
            if label is None:
                label = ""

            if label in self.filtered:
                self.filtered[label] = vstack([self.filtered[label], self.table[~mask]])
            else:
                self.filtered[label] = self.table[~mask]

        # Do the actual filtering.
        self.table = self.table[mask]

        # Return a reference to the current object to allow chaining.
        return self

    def filter_by_index(self, rows_to_keep, label=None):
        """Filter the rows in the ResultTable to only include those indices
        in the list indices_to_keep.

        Parameters
        ----------
        rows_to_keep : `list[int]`
            The indices of the rows to keep.
        label : `str`
            The label of the filtering stage to use. Only used if
            we keep filtered trajectories.

        Returns
        -------
        self : `Results`
            Returns a reference to itself to allow chaining.
        """
        row_set = set(rows_to_keep)
        mask = np.array([i in row_set for i in range(len(self.table))])
        self.filter_mask(mask, label)
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

        Note
        ----
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
            self.table[add_column] = [""] * len(self.table)

        # Make a list of tables to merge.
        table_list = [self.table]
        for key in to_revert:
            filtered_table = self.filtered[key]
            if add_column is not None and len(filtered_table) > 0:
                filtered_table[add_column] = [key] * len(filtered_table)
            table_list.append(filtered_table)
            del self.filtered[key]
        self.table = vstack(table_list)

        return self

    def write_table(self, filename, overwrite=True, cols_to_drop=[]):
        """Write the unfiltered results to a single (ecsv) file.

        Parameter
        ---------
        filename : `str`
            The name of the result file.
        overwrite : `bool`
            Overwrite the file if it already exists. [default: True]
        cols_to_drop : `list`
            A list of columns to drop (to save space). [default: []]
        """
        if len(cols_to_drop) > 0:
            # Make a copy so we can modify the table
            write_table = self.table.copy()

            for col in cols_to_drop:
                if col in write_table.colnames:
                    write_table.remove_column(col)

            # Write out the table.
            write_table.write(filename, overwrite=overwrite)
        else:
            self.table.write(filename, overwrite=overwrite)

    def write_trajectory_file(self, filename, overwrite=True):
        """Save the trajectories to a numpy file.

        Parameters
        ----------
        filename : `str`
            The file name for the ouput file.
        overwrite : `bool`
            Whether to overwrite an existing file.

        Raises
        ------
        Raises a FileExistsError is the file already exists and
        ``overwrite`` is set to ``False``.
        """
        if not overwrite and Path(filename).is_file():
            raise FileExistsError(f"{filename} already exists")
        FileUtils.save_results_file(filename, self.make_trajectory_list())

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
        Raises a FileNotFoundError is the file does not exist.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"{filename} not found for load.")

        trj_list = FileUtils.load_results_file_as_trajectories(filename)
        return cls(trj_list, track_filtered)
