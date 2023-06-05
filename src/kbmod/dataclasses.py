import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


__all__ = ["StandardizedHeader",]




@dataclass
class StandardizedHeader:
    """A dataclass that associates standardized metadata with one or more
    standardized WCS.
    """
    metadata: Metadata = None
    wcs: Sequence[Wcs] = field(default_factory=list)

    @classmethod
    def fromDict(cls, data):
        """Construct an StandardizedHeader from a dictionary.

        The dictionary can contain either a flattened set of values like:

            {obs_lon: ... <metadata keys>, wcs_radius: ... <wcs_keys>}

        or have separated metadata and wcs values like:

            {metadata: {...}, wcs: {...}}

        in which case the wcs can be an iterable, i.e.

            {metadata: {...}, wcs: [{...}, {...}, ... ]}

        Parameters
        ----------
        data : `dict`
            Dictionary containing at least the required standardized keys for
            `Metadata` and `Wcs`.
        """
        meta, wcs = None, []
        if "metadata" in data and "wcs" in data:
            meta = Metadata(**data["metadata"])

            # sometimes multiExt Fits have only 1 valid image extension
            # otherwise we expect a list.
            if isinstance(data["wcs"], dict):
                wcs.append(Wcs(metadata=meta, **data["wcs"]))
            else:
                for ext in data["wcs"]:
                    wcs.append(Wcs(metadata=meta, **ext))
        else:
            meta = Metadata.fromDictSubset(data)
            wcs = Wcs.fromDictSubset(data)
            wcs.metadata = meta

        if type(wcs) != list:
            wcs = [wcs, ]

        return cls(metadata=meta, wcs=wcs)

    def __eq__(self, other):
        return self.isClose(other)

    @property
    def isMultiExt(self):
        """True when the header is a multi extension header."""
        return len(self.wcs) > 1

    def updateMetadata(self, standardizedMetadata):
        """Update metadata values from a dictionary or another Metadata object.

        Parameters
        ----------
        standardizedMetadata : `dict` or `Metadata`
            All, or subset of all, metadata keys which values will be updated.
        """
        metadata = dataToComponent(standardizedMetadata, Metadata)
        if metadata is not None:
            self.metadata = metadata
        else:
            raise ValueError(f"Could not create metadata from the given data: {standardizedMetadata}")

    def appendWcs(self, standardizedWcs):
        """Append a WCS component.

        Parameters
        ----------
        standardizedWcs : `dict` or `Wcs`
            Data which to append to the current collection of associated wcs's.
        """
        wcs = dataToComponent(standardizedWcs, Wcs)
        if wcs is not None:
            self.wcs.append(wcs)
        else:
            raise ValueError(f"Could not create a WCS from the given data: {standardizedWcs}")

    def extendWcs(self, standardizedWcs):
        """Extend current collection of associated wcs by appending elements
        from an iterable.

        Parameters
        ----------
        standardizedWcs : `iterable``
            Data which to append to the current collection of associated wcs's.
        """
        for wcs in standardizedWcs:
            self.appendWcs(wcs)

    def isClose(self, other, **kwargs):
        """Tests approximate equality between two standardized headers by
        testing appoximate equality of respective metadatas and wcs's.

        Parameters
        ----------
        other : `StandardizeHeader`
            Another `Metadata` instance to test approximate equality with.
        **kwargs : `dict`
            Keyword arguments passed onto `numpy.allclose`

        Returns
        -------
        approxEqual : `bool`
            True when approximately equal, False otherwise.
        """
        if len(self.wcs) != len(other.wcs):
            return False

        areClose = self.metadata.isClose(other.metadata, **kwargs)
        for thisWcs, otherWcs in zip(self.wcs, other.wcs):
            areClose = areClose and thisWcs.isClose(otherWcs, **kwargs)

        return areClose

    def toDict(self):
        """Returns a dictionary of standardized metadata and wcs values."""
        if self.isMultiExt:
            wcsDicts = {"wcs": [wcs.toDict() for wcs in self.wcs]}
        else:
            wcsDicts = {"wcs": self.wcs[0].toDict()}
        metadataDict = {"metadata": self.metadata.toDict()}
        metadataDict.update(wcsDicts)
        return metadataDict
