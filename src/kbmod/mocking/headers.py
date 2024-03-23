import abc
import types
import warnings
import functools

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, ImageHDU, BinTableHDU, TableHDU, Header

from .utils import header_archive_to_table


__all__ = [
    "HeaderFactory",
    "StaticHeader",
    "MutableHeader",
    "ArchivedHeader",
]


class HeaderFactory(abc.ABC):
    @abc.abstractmethod
    def mock(self, hdu=None, **kwargs):
        raise NotImplementedError()


class StaticHeader(HeaderFactory):
    def __init__(self, metadata=None):
        self.metadata = metadata

    def mock(self, hdu=None, **kwargs):
        # I have no idea why cards are an empty list by default
        cards = [] if self.metadata is None else self.metadata
        return Header(cards=cards)


class MutableHeader(HeaderFactory):
    def __init__(self, metadata=None, mutables=None, callbacks=None):
        for k in mutables:
            if k not in metadata:
                raise ValueError(
                    f"Registered mutable key {k} does not exists " "in given metadata: f{metadata}."
                )

        self.metadata = metadata
        self.mutables = mutables
        self.callbacks = callbacks

    def mock(self, hdu=None, **kwargs):
        for i, mutable in enumerate(self.mutables):
            self.metadata[mutable] = self.callbacks[i](self.metadata[mutable])

        return Header(self.metadata)


class ArchivedHeader(HeaderFactory):
    # will almost never be anything else. Rather, it would be a miracle if it
    # were something else, since FITS standard shouldn't allow it. Further
    # casting by some packages will always be casting implemented in terms of
    # parsing these types.
    lexical_type_map = {
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
    }
    """Map between type names and types themselves."""

    def __init__(self, archive_name, fname, compression="bz2", format="ascii.ecsv"):
        self.table = header_archive_to_table(archive_name, fname, compression, format)

        # Create HDU groups for easier iteration
        self.table = self.table.group_by(["filename", "hdu"])
        self.n_hdus = len(self.table)

        # Internal counter for the current fits index,
        # so that we may auto-increment it and avoid returning
        # the same data all the time.
        self._current = 0

    def lexical_cast(self, value, format):
        """Cast str literal of a type to the type itself. Supports just
        the builtin Python types.
        """
        if format in self.lexical_type_map:
            return self.lexical_type_map[format](value)
        return value

    def mock(self, hdu=None):
        header = Header()
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        for k, v, f in self.table.groups[self._current]["keyword", "value", "format"]:
            header[k] = self.lexical_cast(v, f)
        warnings.resetwarnings()
        self._current += 1
        return header
