import abc
import tarfile
import warnings

import numpy as np
from astropy.table import Table, MaskedColumn
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, ImageHDU, BinTableHDU, TableHDU, Header

from .utils import header_archive_to_table


__all__ = [
    "SimpleHeaderFactory",
    "ArchivedHeadersFactory",
    "ZeroedData",
    "HDUFactory",
    "DECamImdiffs"
]


class HeaderFactory(abc.ABC):
    @abc.abstractmethod
    def mock(self, hdu=None, **kwargs):
        raise NotImplementedError()


class SimpleHeaderFactory(HeaderFactory):
    def __init__(self, metadata=None):
        self.metadata = metadata

    def mock(self, hdu=None, **kwargs):
        # I have no idea why cards are an empty list by default
        cards = [] if self.metadata is None else self.metadata
        return Header(cards=cards)


# HeadersFromSerializedHeadersFactory - how to name this?
class ArchivedHeadersFactory(HeaderFactory):
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


class DataFactory(abc.ABC):
    # https://archive.stsci.edu/fits/fits_standard/node39.html#s:man
    bitpix_type_map = {
        # or char
        8: int,
        # actually no idea what dtype, or C type for that matter,
        # is used to represent these two values. But default Headers
        16: np.float16,
        32: np.float32,
        64: np.float64,
        # classic IEEE float and double
        -32: np.float32,
        -64: np.float64
    }

    @abc.abstractmethod
    def mock(self, *args, **kwargs):
        pass


class ZeroedData(DataFactory):
    def __init__(self):
        super().__init__()

    def mock_image_data(self, hdu):
        cols = hdu.header.get("NAXIS1", False)
        rows = hdu.header.get("NAXIS2", False)

        cols = 5 if not cols else cols
        rows = 5 if not rows else rows

        data = np.zeros(
            (cols, rows),
            dtype=self.bitpix_type_map[hdu.header["BITPIX"]]
        )
        return data

    def mock_table_data(self, hdu):
        # interestingly, table HDUs create their own empty
        # tables from headers, but image HDUs do not, this
        # is why hdu.data exists and has a valid dtype
        nrows = hdu.header["TFIELDS"]
        return np.zeros((nrows,), dtype=hdu.data.dtype)

    def mock(self, hdu=None, **kwargs):
        if isinstance(hdu, (PrimaryHDU, CompImageHDU, ImageHDU)):
            return self.mock_image_data(hdu)
        elif isinstance(hdu, (TableHDU, BinTableHDU)):
            return self.mock_table_data(hdu)
        else:
            raise TypeError(f"Expected an HDU, got {type(hdu)} instead.")


class HDUFactory():
    def __init__(self, hdu_cls, header_factory=None, data_factory=None):
        self.hdu_cls = hdu_cls

        self.update_header = False if header_factory is None else True
        self.header_factory = header_factory

        self.update_data = False if data_factory is None else True
        self.data_factory = data_factory

    def mock(self, **kwargs):
        hdu = self.hdu_cls()

        if self.update_header:
            header = self.header_factory.mock(hdu=hdu, **kwargs)
            hdu.header.update(header)

        if self.update_data:
            data = self.data_factory.mock(hdu=hdu, **kwargs)
            hdu.data = data
            # not sure if we want this tbh
            #hdu.update_header()

        return hdu


class HDUListFactory():
    def __init__(self, layout):
        self.layout = layout

    def mock(self, **kwargs):
        hdul = HDUList()
        for layer in self.layout:
            hdul.append(layer.mock(**kwargs))
        return hdul


class SimpleFits(HDUListfactory):
    def __init__(self):
        primary = SimpleHeaderFactory({
            "EXTNAME": "PRIMARY",
        })

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1024.0, 2048.0]
        wcs.wcs.cd = np.array([
            [-1.44e-07, 7.32e-05],
            [7.32e-05, 1.44e-05]
        ])
        wcs.wcs.crval = [351, -5]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ['deg', 'deg']
        wcs.wcs.radesys = "ICRS"

        header = wcs.to_header()
        header["NAXIS"] = 2
        # eader["EXTNAME"]
        header["NAXIS1"] = 2048
        header["NAXIS2"] = 4096

        image_hdr = SimpleHeaderFactory(header)
        data = ZeroedData()

        layout = [
            HDUFactory(primary)
            HDUFactory(CompImageHDU, image_hdr, data)
            HDUFactory(CompImageHDU, image_hdr, data)
            HDUFactory(CompImageHDU, image_hdr, data)
        ]


class DECamImdiffs(HDUListFactory):
    def __init__(self):
        headers = ArchivedHeadersFactory("headers_archive.tar.bz2", "decam_imdiff_headers.ecsv")
        data = ZeroedData()
        image = HDUFactory(CompImageHDU, headers, data)

        # don't let Black have its way with these lines because it's a massacre
        # fmt: off
        # primary, science variance and mask
        layout = [
            HDUFactory(PrimaryHDU, headers),
            image,
            image,
            image,
        ]

        # PSF, SkyWCS, catalog meta, higher order corrections etc.
        layout.extend([HDUFactory(BinTableHDU, headers) ] * 12)
        # fmt: on

        super().__init__(layout)
