import warnings

from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.io.fits import Header

from .utils import header_archive_to_table


__all__ = [
    "HeaderFactory",
    "ArchivedHeader",
]


class HeaderFactory:
    base_primary = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "OBS-MJD": 58914.0,
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
    }

    base_ext = {"NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096, "BITPIX": 32}

    base_wcs = {
        "ctype": ["RA---TAN", "DEC--TAN"],
        "crval": [351, -5],
        "cunit": ["deg", "deg"],
        "radesys": "ICRS",
        "cd": [[-1.44e-07, 7.32e-05], [7.32e-05, 1.44e-05]],
    }

    def __validate_mutables(self):
        # !xor
        if bool(self.mutables) != bool(self.callbacks):
            raise ValueError(
                "When providing a list of mutable cards, you must provide associated callback methods."
            )

        if self.mutables is None:
            return

        if len(self.mutables) != len(self.callbacks):
            raise ValueError(
                "The number of mutable cards does not correspond to the number of given callbacks."
            )

        for k in self.mutables:
            if k not in self.header:
                raise ValueError(
                    f"Mutable key {k} does not exists "
                    "in the header. Please "
                    "provide the required metadata keys."
                )

    def __init__(self, metadata=None, mutables=None, callbacks=None):
        cards = [] if metadata is None else metadata
        self.header = Header(cards=cards)

        self.mutables = mutables
        self.callbacks = callbacks
        self.__validate_mutables()

    def mock(self, hdu=None, **kwargs):
        if self.mutables is not None:
            for i, mutable in enumerate(self.mutables):
                self.header[mutable] = self.callbacks[i](self.header[mutable])

        return self.header

    @classmethod
    def gen_wcs(cls, crval, metadata=None):
        metadata = cls.base_wcs if metadata is None else metadata
        wcs = WCS(naxis=2)
        for k, v in metadata.items():
            setattr(wcs.wcs, k, v)
        return wcs.to_header()

    @classmethod
    def gen_header(cls, base, metadata, extend, add_wcs):
        header = Header(base) if extend else Header()
        header.update(metadata)

        if add_wcs:
            naxis1 = header.get("NAXIS1", False)
            naxis2 = header.get("NAXIS2", False)
            if not all((naxis1, naxis2)):
                raise ValueError("Adding a WCS to the header requires NAXIS1 and NAXIS2 keys.")
            crpix = [naxis1 / 2.0, naxis2 / 2.0]
            header.update(cls.gen_wcs(crpix))

        return header

    @classmethod
    def from_base_primary(cls, metadata=None, mutables=None, callbacks=None, extend_base=True, add_wcs=False):
        hdr = cls.gen_header(cls.base_primary, metadata, extend_base, add_wcs)
        return cls(hdr, mutables, callbacks)

    @classmethod
    def from_base_ext(
        cls, metadata=None, mutables=None, callbacks=None, extend_base=True, add_wcs=True, dims=None
    ):
        hdr = cls.gen_header(cls.base_ext, metadata, extend_base, add_wcs)
        return cls(hdr, mutables, callbacks)


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
