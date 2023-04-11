import abc

from kbmod.result_list import *


class RowFilter(abc.ABC):
    """The base class for derived filters on the ResultList
    that operate on the results one row at a time."""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        pass

    @abc.abstractmethod
    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        the row's information and the parameters of the filter.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        pass


class BatchFilter(abc.ABC):
    """The base class for derived filters on the ResultList
    that operate on the results in a single batch.

    Batching should be used when the user needs greater control
    over how the filter is run, such as using aggregate statistics
    from all candidates or running batch computations on GPUs.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        pass

    @abc.abstractmethod
    def keep_indices(self, results: ResultList):
        """Determine which of the ResultList's indices to keep.

        Parameters
        ----------
        results: ResultList
            The set of results to filter.

        Returns
        -------
        list
           A list of indices (int) indicating which rows to keep.
        """
        pass
