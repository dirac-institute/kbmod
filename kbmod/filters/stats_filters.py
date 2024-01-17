import numpy as np

from kbmod.filters.base_filter import RowFilter
from kbmod.result_list import ResultRow


class LHFilter(RowFilter):
    """A filter for result's likelihood."""

    def __init__(self, min_lh, max_lh, *args, **kwargs):
        """Create a ResultsLHFilter.

        Parameters
        ----------
        min_lh : ``float``
            Minimal allowed likelihood. Use None for no min filter.
            Using None is equivalent to using -math.inf.
        max_lh : ``float``
            Maximal allowed likelihood. Use None for no max filter.
            Using None is equivalent to using math.inf.

        Examples
        --------
        # r is a ResultRow with likelihood of 10.
        >>> LHFilter(5, 15).keep_row(r)
        True
        >>> LHFilter(20, 30).keep_row(r)
        False
        >>> LHFilter(None, 30).keep_row(r)
        True
        """
        super().__init__(*args, **kwargs)

        self.min_lh = min_lh
        self.max_lh = max_lh

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"LH_Filter_{self.min_lh}_to_{self.max_lh}"

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        the likelihood.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        lh = row.final_likelihood
        if self.min_lh is not None and lh < self.min_lh:
            return False
        if self.max_lh is not None and lh > self.max_lh:
            return False
        return True


class NumObsFilter(RowFilter):
    """A filter for result's number of valid observations."""

    def __init__(self, min_obs, *args, **kwargs):
        """Create a ResultsNumObsFilter.

        Parameters
        ----------
        min_obs : ``int``
            The minimum number of valid observations needed.

        Examples
        --------
        # r is a ResultRow with 5 valid indices.
        >>> NumObsFilter(10).keep_row(r)
        False
        >>> NumObsFilter(2).keep_row(r)
        True
        """
        super().__init__(*args, **kwargs)

        self.min_obs = min_obs

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"MinObsFilter_{self.min_obs}"

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        the number of valid observations.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        return len(row.valid_indices) >= self.min_obs


class CombinedStatsFilter(RowFilter):
    """A filter for result's likelihood and number of observations."""

    def __init__(self, min_obs=0, min_lh=-np.inf, max_lh=np.inf, *args, **kwargs):
        """Create a ResultsLHFilter.

        Parameters
        ----------
        min_obs : ``int``
            The minimum number of observations.
        min_lh : ``float``
            Minimal allowed likelihood.
        max_lh : ``float``
            Maximal allowed likelihood.
        """
        super().__init__(*args, **kwargs)

        self.min_obs = min_obs
        self.min_lh = min_lh
        self.max_lh = max_lh

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"CombinedStats_{self.min_obs}_{self.min_lh}_to_{self.max_lh}"

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        the likelihood.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        if row.final_likelihood < self.min_lh or row.final_likelihood > self.max_lh:
            return False
        if len(row.valid_indices) < self.min_obs:
            return False
        return True


class DurationFilter(RowFilter):
    """A filter for the amount of time covered by the trajectory"""

    def __init__(self, all_times, min_duration, *args, **kwargs):
        """Create a ResultsLHFilter.

        Parameters
        ----------
        all_times : ``list``
            The time stamps in increasing order.
        min_duration : ``float``
            The minimum duration in days for a valid result.
        """
        super().__init__(*args, **kwargs)

        self.all_times = all_times
        self.min_duration = min_duration

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"Duration_{self.min_duration}"

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        the likelihood.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        min_index = np.min(row.valid_indices)
        max_index = np.max(row.valid_indices)
        if self.all_times[max_index] - self.all_times[min_index] < self.min_duration:
            return False
        return True
