"""ResultTable is a column-based data structure for tracking results with additional global data
and helper functions for filtering and maintaining consistency between different attributes in each row.
"""

import numpy as np
from pathlib import Path

from astropy.table import Table, vstack

from kbmod.trajectory_utils import make_trajectory, update_trajectory_from_psi_phi
from kbmod.search import Trajectory


class ResultTable:
    """This class stores a collection of related data from all of the kbmod results.

    At a minimum it contains columns for the trajectory information:
    (x, y, vx, vy, likelihood, flux, obs_count)
    but additional columns can be added as needed.
    """

    def __init__(self, trj_list, track_filtered=False):
        """Create a ResultTable class.

        Parameters
        ----------
        trj_list : `list[Trajectory]`
            A list of trajectories to include in these results.
        track_filtered : bool
            Whether to track (save) the filtered trajectories. This will use
            more memory and is recommended only for analysis.
        """
        valid_inds = [i for i in range(len(trj_list)) if trj_list[i].valid]
        input_dict = {
            "x": [trj_list[i].x for i in valid_inds],
            "y": [trj_list[i].y for i in valid_inds],
            "vx": [trj_list[i].vx for i in valid_inds],
            "vy": [trj_list[i].vy for i in valid_inds],
            "likelihood": [trj_list[i].lh for i in valid_inds],
            "flux": [trj_list[i].flux for i in valid_inds],
            "obs_count": [trj_list[i].obs_count for i in valid_inds],
            "trajectory": [trj_list[i] for i in valid_inds],
        }
        self.results = Table(input_dict)

        # Set up information to track which row is filtered at which round.
        self.track_filtered = track_filtered
        self.filtered = {}

    def __len__(self):
        """Return the number of results in the list."""
        return len(self.results)

    @property
    def colnames(self):
        return self.results.colnames

    @classmethod
    def from_table(cls, data, track_filtered=False):
        """Extract the ResultList from an astropy Table with the minimum
        trajectory information. Fills in missing columns (such as the trajectory
        object) if they are not present.

        Parameters
        ----------
        data : `astropy.table.Table`
            The input data.
        track_filtered : `bool`
            Indicates whether the ResultList should track future filtered points.

        Raises
        ------
        KeyError if any required columns are missing.
        """
        # Check that the minimum information is present.
        required_cols = ["x", "y", "vx", "vy", "likelihood", "flux", "obs_count"]
        for col in required_cols:
            if col not in data.colnames:
                raise KeyError(f"Column {col} missing from input data.")

        # Create an empty ResultTable and append the data table.
        table = ResultTable([], track_filtered=track_filtered)
        table.results = data

        # If the data did not have a column for Trajectory object, add it with
        # an expensive linear scan.
        if "trajectory" not in data.colnames:
            trjs = [
                make_trajectory(
                    x=row["x"],
                    y=row["y"],
                    vx=row["vx"],
                    vy=row["vy"],
                    flux=row["flux"],
                    lh=row["likelihood"],
                    obs_count=row["obs_count"],
                )
                for row in data
            ]
        table.results["trajectory"] = trjs

        return table

    @classmethod
    def read_table(self, filename, track_filtered=False):
        """Read the ResultList from a table file.

        Parameters
        ----------
        filename : `str`
            The name of the file to load.
        track_filtered : `bool`
            Indicates whether the ResultList should track future filtered points.

        Raises
        ------
        FileNotFoundError if the file is not found.
        KeyError if any of the columns are missing.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError
        data = Table.read(filename)
        return ResultTable.from_table(data, track_filtered=track_filtered)

    def extend(self, table2):
        """Append the results in a second ResultTable to the current one.

        Parameters
        ----------
        table2 : `ResultTable`
            The data structure containing additional `ResultTable` elements to add.
        """
        self.results = vstack([self.results, table2.results])

        # When merging the filtered results extend lists with the
        # same key and create new lists for new keys.
        for key in table2.filtered.keys():
            if key in self.filtered:
                self.filtered[key] = vstack([self.filtered[key], table2.filtered[key]])
            else:
                self.filtered[key] = table2.filtered[key]

    def _update_likelihood(self):
        """Update the likelihood related trajectory information from the
        psi and phi information. Requires the existence of the columns
        'psi_curve' and 'phi_curve' which can be set with add_psi_phi_data().
        Uses the (optional) 'valid_indices' if it exists.

        Raises
        ------
        Raises an IndexError if the necessary columns are missing.
        """
        if "psi_curve" not in self.results.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")
        if "phi_curve" not in self.results.colnames:
            raise IndexError("Missing column 'phi_curve'. Use add_psi_phi_data()")
        use_valid_indices = "index_valid" in self.results.colnames
        inds = None

        # Go through each row to update.
        for row in self.results:
            if use_valid_indices:
                inds = row["index_valid"]
            trj = update_trajectory_from_psi_phi(
                row["trajectory"], row["psi_curve"], row["phi_curve"], index_valid=inds, in_place=True
            )

            # Update the exploded columns.
            row["likelihood"] = trj.lh
            row["flux"] = trj.flux
            row["obs_count"] = trj.obs_count

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

        Raises
        ------
        Raises a ValueError if the input arrays are not the same size as the table
        or a given pair of rows in the arrays are not the same length.
        """
        if len(psi_array) != len(self.results):
            raise ValueError("Wrong number of psi curves provided.")
        if len(phi_array) != len(self.results):
            raise ValueError("Wrong number of phi curves provided.")
        self.results["psi_curve"] = psi_array
        self.results["phi_curve"] = phi_array

        if index_valid is not None:
            # Make the data to match.
            if len(index_valid) != len(self.results):
                raise ValueError("Wrong number of index_valid lists provided.")
            self.results["index_valid"] = index_valid

        # Update the track likelihoods given this new information.
        self._update_likelihood()

    def filter_mask(self, mask, label=None):
        """Filter the rows in the ResultTable to only include those indices
        that are marked True in the mask.

        Parameters
        ----------
        mask : list
            A list the same length as the table with True/False indicating
            which row to keep.
        label : string
            The label of the filtering stage to use. Only used if
            we keep filtered trajectories.

        Returns
        -------
        self : ResultTable
            Returns a reference to itself to allow chaining.
        """
        if self.track_filtered:
            if label is None:
                label = ""

            if label in self.filtered:
                self.filtered[label] = vstack([self.filtered[label], self.results[~mask]])
            else:
                self.filtered[label] = self.results[~mask]

        # Do the actual filtering.
        self.results = self.results[mask]

        # Return a reference to the current object to allow chaining.
        return self

    def filter_by_index(self, indices_to_keep, label=None):
        """Filter the rows in the ResultTable to only include those indices
        in the list indices_to_keep.

        Parameters
        ----------
        indices_to_keep : `list[int]`
            The indices of the rows to keep.
        label : `str`
            The label of the filtering stage to use. Only used if
            we keep filtered trajectories.

        Returns
        -------
        self : `ResultTable`
            Returns a reference to itself to allow chaining.
        """
        indices_set = set(indices_to_keep)
        mask = np.array([i in indices_set for i in range(len(self.results))])
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
        results : `astropy.table.Table` or None
            A table with the filtered rows or None if there are no entries.
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
        self : `ResultTable`
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
        if add_column is not None and add_column not in self.results.colnames:
            self.results[add_column] = [""] * len(self.results)

        # Make a list of tables to merge.
        table_list = [self.results]
        for key in to_revert:
            filtered_table = self.filtered[key]
            if add_column is not None:
                filtered_table[add_column] = [key] * len(filtered_table)
            table_list.append(filtered_table)
            del self.filtered[key]
        self.results = vstack(table_list)

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
        # Make a copy so we can modify the table (drop the Trajectory objects)
        write_table = self.results.copy()

        all_cols_to_drop = ["trajectory"] + cols_to_drop
        for col in all_cols_to_drop:
            if col in write_table.colnames:
                write_table.remove_column(col)

        # Write out the table.
        write_table.write(filename, overwrite=overwrite)
