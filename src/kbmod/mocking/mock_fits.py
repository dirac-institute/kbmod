import abc
import warnings
import functools

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, ImageHDU, BinTableHDU, TableHDU, Header

from .utils import header_archive_to_table


__all__ = [
    "StaticHeader",
    "ArchivedHeader",
    "ZeroedData",
    "HDUFactory",
    "SimpleFits",
    "DECamImdiffs"
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
                raise ValueError(f"Registered mutable key {k} does not exists "
                                 "in given metadata: f{metadata}.")

        self.metadata = metadata
        self.mutables = mutables
        self.callbacks = callbacks

    def mock(self, hdu=None, **kwargs):
        for i, mutable in enumerate(self.mutables):
            self.metadata[mutable] = self.callbacks[i](self.metadata[mutable])

        return Header(self.metadata)


class IncrementalObstimeHeader(StaticHeader):
    def __init__(self, obstime_key, dt, metadata=None):
        self.metadata = metadata
        self.obstime_key = obstime_key
        self.dt = dt

    def mock(self, hdu=None, **kwargs):
        # I have no idea why cards are an empty list by default
        cards = [] if self.metadata is None else self.metadata

        # has to be a header or dict now
        cards[self.obstime_key] += self.dt

        return Header(cards=cards)


# HeadersFromSerializedHeadersFactory - how to name this?
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
    def __init__(self, layout, base_primary={}, base_ext={}, base_wcs={}):
        self.layout = layout
        self.base_primary = base_primary
        self.base_ext = base_ext
        self.base_wcs = base_wcs

    def gen_wcs(self, metadata=None):
        metadata = self.base_wcs if metadata is None else metadata
        wcs = WCS(naxis=2)
        for k, v in metadata.items():
            setattr(wcs.wcs, k, v)
        return wcs.to_header()

    def gen_header(self, base, metadata, extend, add_wcs):
        header = Header(base) if extend else Header()
        header.update(metadata)
        if add_wcs:
            header.update(self.gen_wcs())
        return header

    def gen_primary(self, metadata=None, extend_base=True, add_wcs=False):
        return self.gen_header(self.base_primary, metadata, extend_base, add_wcs)

    def gen_ext(self, metadata=None, extend_base=True, add_wcs=True):
        return self.gen_header(self.base_ext, metadata, extend_base, add_wcs)

    def mock(self, **kwargs):
        hdul = HDUList()
        for layer in self.layout:
            hdul.append(layer.mock(**kwargs))
        return hdul


# I am sure a decorator like this must exist somewhere in functools, but can't
# find it and I'm doing something wrong with functools.partial because that's
# strictly right-side binding?
def callback(func):
    def wrapper(*args, **kwargs):
        @functools.wraps(func)
        def f(*fargs, **fkwargs):
            kwargs.update(fkwargs)
            return func(*(args+fargs), **kwargs)
        return f
    return wrapper

class SimpleFits(HDUListFactory):
    base_primary = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "OBS-MJD": 58914.0,
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
        "STD": "SimpleFits"
    }

    base_ext = {
        "NAXIS": 2,
        "NAXIS1": 2048,
        "NAXIS2": 4096,
        "BITPIX": 32
    }

    base_wcs = {
        "crpix": [1024.0, 2048.0],
        "crval" : [351, -5],
        "ctype": ["RA---TAN", "DEC--TAN"],
        "cunit": ["deg", "deg"],
        "radesys": "ICRS",
        "cd": [
            [-1.44e-07, 7.32e-05],
            [7.32e-05, 1.44e-05]
        ]
    }

    @callback
    def increment_obstime(self, old, dt):
        return old+dt

    def __init__(self):
        primary_hdr = self.gen_primary()

        # multiple options on how to handle callbacks as class members
        #callbacks=[lambda old: old+0.01, ]
        #callbacks=[functools.partial(self.increment_obstime, dt=0.001), ]
        primary_header_factory = MutableHeader(
            metadata=primary_hdr,
            mutables=["OBS-MJD", ],
            callbacks=[self.increment_obstime(dt=0.001), ]
        )

        image_hdr = self.gen_ext({"EXTNAME": "IMAGE"}, add_wcs=True)
        variance_hdr = self.gen_ext({"EXTNAME": "VARIANCE"}, add_wcs=True)
        mask_hdr = self.gen_ext({"EXTNAME": "MASK", "BITPIX": 8}, add_wcs=True)

        data = ZeroedData()

        layout = [
            HDUFactory(PrimaryHDU, primary_header_factory),
            HDUFactory(CompImageHDU, StaticHeader(image_hdr), data),
            HDUFactory(CompImageHDU, StaticHeader(variance_hdr), data),
            HDUFactory(CompImageHDU, StaticHeader(mask_hdr), data)
        ]

        super().__init__(layout)


class DECamImdiffs(HDUListFactory):
    def __init__(self):
        headers = ArchivedHeader("headers_archive.tar.bz2", "decam_imdiff_headers.ecsv")
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

        # PSF, SkyWCS, catalog meta, chebyshev higher order corrections etc.
        # we don't use these so it's fine to leave them empty
        layout.extend([HDUFactory(BinTableHDU, headers) ] * 12)
        # fmt: on

        super().__init__(layout)
