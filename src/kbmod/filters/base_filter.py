import abc

from kbmod.result_list import *


class Filter(abc.ABC):
    """The base class for derived filters on the ResultList."""

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
