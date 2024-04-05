"""ResultTable is a column-based data structure for tracking results with additional global data
and helper functions for filtering and maintaining consistency between different attributes in each row.
"""

import numpy as np

from astropy.table import Table, vstack

from kbmod.trajectory_utils import make_trajectory
from kbmod.search import Trajectory


class ResultTable:
    """This class stores a collection of related data from all of the kbmod results."""

    def __init__(self, trj_list, all_times, track_filtered=False):
        """Create a ResultTable class.

        Parameters
        ----------
        trj_list : `list[Trajectory]`
            A list of trajectories to include in these results.
        all_times : `list[float]`
            A list of all time stamps.
        track_filtered : bool
            Whether to track (save) the filtered trajectories. This will use
            more memory and is recommended only for analysis.
        """
        self._all_times = all_times

        valid_inds = [i for i in range(len(trj_list)) if trj_list[i].valid]
        input_dict = {
            "x": [trj_list[i].x for i in valid_inds],
            "y": [trj_list[i].y for i in valid_inds],
            "vx": [trj_list[i].vx for i in valid_inds],
            "vy": [trj_list[i].vy for i in valid_inds],
            "likelihood": [trj_list[i].lh for i in valid_inds],
            "flux": [trj_list[i].flux for i in valid_inds],
            "obs_count": [trj_list[i].obs_count for i in valid_inds],
            "valid_indices": [np.arange(0, len(all_times)) for i in valid_inds],
        }
        self.results = Table(input_dict)

        # Set up information to track which row is filtered at which round.
        self.track_filtered = track_filtered
        self.filtered = {}

    # All times should be externally read-only once set.
    @property
    def all_times(self):
        return self._all_times

    def __len__(self):
        """Return the number of results in the list."""
        return len(self.results)

    def extend(self, table2):
        """Append the results in a second ResultTable to the current one.

        Parameters
        ----------
        table2 : `ResultTable`
            The data structure containing additional `ResultTable` elements to add.
        """
        if not np.array_equal(self._all_times, table2._all_times):
            raise ValueError("Incompatible ResultTables. Different time arrays.")
        self.results = vstack([self.results, table2.results])

        # When merging the filtered results extend lists with the
        # same key and create new lists for new keys.
        for key in table2.filtered.keys():
            if key in self.filtered:
                self.filtered[key] = vstack([self.filtered[key], table2.filtered[key]])
            else:
                self.filtered[key] = table2.filtered[key]

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

    def revert_filter(self, label=None):
        """Revert the filtering by re-adding filtered ResultRows.

        Note
        ----
        Filtered rows are appended to the end of the list. Does not return
        the results to the original ordering.

        Parameters
        ----------
        label : str
            The filtering stage to use. If no label is provided,
            revert all filtered rows.

        Returns
        -------
        self : ResultList
            Returns a reference to itself to allow chaining.

        Raises
        ------
        ValueError if filtering is not enabled.
        KeyError if label is unknown.
        """
        if not self.track_filtered:
            raise ValueError("ResultList filter tracking not enabled.")

        if label is not None:
            # Check if anything was filtered at this stage.
            if label in self.filtered:
                self.results = vstack([self.results, self.filtered[label]])
                del self.filtered[label]
            else:
                raise KeyError(f"Unknown filtered label {label}")
        else:
            result_list = [self.results]
            for key in self.filtered:
                result_list.append(self.filtered[key])
            self.results = vstack(result_list)

            # Reset the entire dictionary.
            self.filtered = {}

        return self
