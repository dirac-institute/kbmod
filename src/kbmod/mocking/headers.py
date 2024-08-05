import warnings

from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.io.fits import Header

from .utils import header_archive_to_table
from .config import Config


__all__ = [
    "HeaderFactory",
    "ArchivedHeader",
]


class HeaderFactory:
    primary_template = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "OBS-MJD": 58914.0,
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
        "OBS-ELEV": 2200,
        "OBSERVAT": "CTIO"
    }

    ext_template = {
        "NAXIS": 2,
        "NAXIS1": 2048,
        "NAXIS2": 4096,
        "CRPIX1": 1024,
        "CPRIX2": 2048,
        "BITPIX": 32
    }

    wcs_template = {
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

    def __init__(self, metadata, mutables=None, callbacks=None,
                 config=None, **kwargs):
        cards = [] if metadata is None else metadata
        self.header = Header(cards=cards)

        self.mutables = mutables
        self.callbacks = callbacks
        self.__validate_mutables()

        self.is_dynamic = self.mutables is not None
        self.counter = 0

    def mock(self, n=1):
        headers = []
        # This can't be vectorized because callbacks may share global state
        for i in range(n):
            if self.is_dynamic:
                header = self.header.copy()
                for i, mutable in enumerate(self.mutables):
                    header[mutable] = self.callbacks[i](header[mutable])
            else:
                header = self.header
            headers.append(header)
            self.counter += 1

        return headers

    @classmethod
    def gen_wcs(cls, metadata):
        wcs = WCS(naxis=2)
        for k, v in metadata.items():
            setattr(wcs.wcs, k, v)
        return wcs.to_header()

    @classmethod
    def gen_header(cls, base, overrides, wcs_base=None):
        header = Header(base)
        header.update(overrides)

        if wcs_base is not None:
            naxis1 = header.get("NAXIS1", False)
            naxis2 = header.get("NAXIS2", False)
            if not all((naxis1, naxis2)):
                raise ValueError("Adding a WCS to the header requires "
                                 "NAXIS1 and NAXIS2 keys.")
            header.update(cls.gen_wcs(wcs_base))

        return header

    @classmethod
    def from_primary_template(cls, overrides=None, mutables=None, callbacks=None):
        hdr = cls.gen_header(base=cls.primary_template, overrides=overrides)
        return cls(hdr, mutables, callbacks)

    @classmethod
    def from_ext_template(cls, overrides=None, mutables=None, callbacks=None, wcs=None, shape=None):
        ext_template = cls.ext_template.copy()

        if shape is not None:
            ext_template["NAXIS1"] = shape[0]
            ext_template["NAXIS2"] = shape[1]
            ext_template["CRPIX1"] = shape[0]//2
            ext_template["CRPIX2"] = shape[1]//2

        hdr = cls.gen_header(
            base=ext_template,
            overrides=overrides,
            wcs_base=cls.wcs_template if wcs is None else wcs
        )
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

    compression = "bz2"

    format = "ascii.ecsv"

    def __init__(self, archive_name, fname, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.table = header_archive_to_table(
            archive_name, fname, self.compression, self.format
        )

        # Create HDU groups for easier iteration
        self.table = self.table.group_by("filename")
        self.n_hdus = len(self.table)

    def lexical_cast(self, value, format):
        """Cast str literal of a type to the type itself. Supports just
        the builtin Python types.
        """
        if format in self.lexical_type_map:
            return self.lexical_type_map[format](value)
        return value

    def get_item(self, group_idx, hdr_idx):
        header = Header()
        # this is equivalent to one hdulist worth of headers
        group = self.table.groups[group_idx]
        # this is equivalent to one HDU's header
        subgroup = group.group_by("hdu")
        for k, v, f in subgroup.groups[hdr_idx]["keyword", "value", "format"]:
            header[k] = self.lexical_cast(v, f)
        warnings.resetwarnings()
        return header

    def get(self, group_idx):
        headers = []
        # this is a bit repetitive but it saves recreating
        # groups for one HDUL-equivalent many times
        group = self.table.groups[group_idx]
        subgroup = group.group_by("hdu")
        headers = []
        for subgroup in subgroup.groups:
            header = Header()
            for k, v, f in subgroup["keyword", "value", "format"]:
                header[k] = self.lexical_cast(v, f)
            headers.append(header)
        return headers

    def mock(self, n=1):
        res = []
        for _ in range(n):
            res.append(self.get(self.counter))
            self.counter += 1
        return res
