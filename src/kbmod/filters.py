from kbmod.result_list import *


class Filter:
    """The base class for derived filters on the ResultList."""

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        # As a default return the class name.
        return self.__class__

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row.

        The derived class should overwrite this placeholder.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        return True


class LHFilter(Filter):
    """A filter for result's likelihood."""

    def __init__(self, **kwargs):
        """Create a ResultsLHFilter.

        Takes the following optional parameters:
        min_lh : float - The minimum likelihood.
        max_lh : float - The maximum likelihood.
        """
        self.min_lh = kwargs.get("min_lh", None)
        self.max_lh = kwargs.get("max_lh", None)

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


class NumObsFilter(Filter):
    """A filter for result's number of valid observations."""

    def __init__(self, min_obs):
        """Create a ResultsNumObsFilter.

        Parameters
        ----------
        min_obs : int
            The minimum number of valid observations.
        """
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
