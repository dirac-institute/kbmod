import abc

import numpy as np
from astropy.io.fits import (
    PrimaryHDU,
    CompImageHDU,
    ImageHDU,
    BinTableHDU,
    TableHDU
)


__all__ = [
    "DataFactory",
    "ZeroedData",
    "SimpleMask"
]


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
    def mock(self, hdu=None,  **kwargs):
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


class SimpleMask(DataFactory):
    def __init__(self, mask):
        super().__init__()

        # not sure if this is "best" "best" way, but
        # it does safe a lot of array copies if we don't
        # have to write to the array
        self.mask = mask
        self.mask.writeable = False

    @classmethod
    def from_params(cls, shape, padding=0):
        mask = np.zeros(shape)
        mask[:padding] = 1
        mask[shape[0]-padding:] = 1
        mask[:, :padding] = 1
        mask[: shape[1]-padding:] = 1
        return cls(mask)

    # bro I don't know antymore
    @classmethod
    def from_patches(cls, shape, patches):
        mask = np.zeros(shape)
        for patch, value in patches:
            if isinstance(patch, tuple):
                mask[*patch] = value
            elif isinstance(slice):
                mask[slice] = value
            else:
                raise ValueError("Expected a tuple (x, y), (slice, slice) or "
                                 f"slice, got {patch} instead.")

        return cls(mask)
