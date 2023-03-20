import abc

from kbmod.result_list import *


class Filter(abc.ABC):
    """The base class for derived filters on the ResultList."""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_filter_name(self):
        pass

    @abc.abstractmethod
    def keep_row(self, row: ResultRow):
        pass
