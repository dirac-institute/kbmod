import abc

import numpy as np
from astropy.io.fits import (
    PrimaryHDU,
    CompImageHDU,
    ImageHDU,
    BinTableHDU,
    TableHDU
)
from astropy.modeling import models
from astropy.convolution import discretize_model


__all__ = [
    "add_model_objects",
    "DataFactory",
    "ZeroedData",
    "SimpleImage",
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
    def __init__(self, base_image=None, return_copy=False):
        super().__init__()
        self.base_image = base_image
        self.base_image.flags.writeable = False
        self.return_copy = return_copy

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

    def mock_static(self, hdu=None):
        if self.return_copy:
            return self.base_image.copy()
        return self.base_image

    def mock(self, hdu=None, **kwargs):
        if self.base_image is not None:
            return self.mock_static(hdu)
        if isinstance(hdu, (PrimaryHDU, CompImageHDU, ImageHDU)):
            return self.mock_image_data(hdu)
        elif isinstance(hdu, (TableHDU, BinTableHDU)):
            return self.mock_table_data(hdu)
        else:
            raise TypeError(f"Expected an HDU, got {type(hdu)} instead.")


def add_model_objects(img, catalog, model, oversample=1):
    shape = img.shape
    yidx, xidx = np.indices(shape)

    # find catalog columns that exist for the model
    params_to_set = []
    for param in catalog.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    # Save the initial model parameters so we can set them back
    init_params = {param: getattr(model, param) for param in params_to_set}

    try:
        for i, source in enumerate(catalog):
            for param in params_to_set:
                setattr(model, param, source[param])
            model.render(img)
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return img


class SimpleImage(DataFactory):
    rng = np.random.default_rng(12345)
    read_noise_gen = rng.normal
    dark_current_gen = rng.poisson
    sky_count_gen = rng.poisson

    def __init__(self, image, return_copy=False):
        self.base_image = image
        self.base_image.flags.writeable = False
        self.return_copy = return_copy

    def mock(self, hdu=None, **kwargs):
        if self.return_copy:
            return self.base_image.copy()
        return self.base_image

    @classmethod
    def from_simplistic_sim(cls, shape, noise, catalog=None,
                            src_model=models.Gaussian2D(x_stddev=1, y_stddev=1),
                            oversample=1, **kwargs):
        # create poisson background
        img = cls.sky_count_gen(noise, size=shape).astype(np.float32)

        # add catalog objects
        if catalog is None:
            return cls(img, **kwargs)

        img = add_model_objects(img, catalog, src_model, oversample)
        return cls(img, **kwargs)


class SimpleVariance(DataFactory):
    def __init__(self, image, read_noise, gain, return_copy=False):
        self.base_image = image/gain + read_noise**2
        self.return_copy = return_copy

    def mock(self, hdu=None, **kwargs):
        if self.return_copy:
            return self.base_image.copy()
        else:
            return self.base_image


class SimpleMask(DataFactory):
    def __init__(self, mask, return_copy=False):
        super().__init__()

        # not sure if this is "best" "best" way, but
        # it does safe a lot of array copies if we don't
        # have to write to the mocked array (which we shouldn't?)
        self.mask = mask
        self.mask.writeable = False
        self.return_copy = return_copy

    @classmethod
    def from_params(cls, shape, padding=0, bad_columns=[]):
        mask = np.zeros(shape)

        # padding
        mask[:padding] = 1
        mask[shape[0]-padding:] = 1
        mask[:, :padding] = 1
        mask[: shape[1]-padding:] = 1


        # bad columns
        for col in bad_columns:
            mask[:, col] = 1

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

    def mock(self, hdu=None, **kwargs):
        if self.return_copy:
            return self.mask.copy()
        return self.mask


class SimulatedImage(DataFactory):
    rng = np.random.default_rng(12345)
    read_noise_gen = rng.normal
    dark_current_gen = rng.poisson
    sky_count_gen = rng.poisson

    def __add_bad_cols(self, image, cols, bias, n, seed, pattern_offset):
        if not cols:
            # most of the time I imagine we don't need bad cols
            return image

        if bad_columns == "random":
            rng = np.random.RandomState(seed=self.bad_cols_seed)
            bad_cols = rng.randint(0, shape[1], size=n)
        else:
            bad_cols = bad_columns

        self.col_pattern = rng.randint(0, int(pattern_offset * bias), size=shape[0])
        for col in columns:
            image[:, col] = bias + col_pattern

        return image

    def __add_hot_pixels(self, image, pixels, percent, offset):
        if not pixels:
            # most of the time I imagine we don't need hot pixels
            return image

        if pixels == "random":
            rng = np.random.RandomState(seed=self.hot_pixel_seed)
            n_pixels = image.shape[0] * image.shape[1]
            n = int(percent * n_pixels)
            x = rng.randint(0, shape[1], size=n)
            y = rng.randint(0, shape[0], size=n)
            hot_pixels = np.column_stack(x, y)
        else:
            hot_pixels = pixels

        for pix in hot_pixels:
            image[*pix] += image[*pix]*offset

        return image


    def __init__(
            self,
            shape,
            read_noise, gain,
            bias,
            exposure_time, dark_current,
            sky_level,
            bad_columns=False, bad_cols_seed=321, n_bad_cols=5, bad_col_pattern_offset=0.1,
            hot_pixels=False, hot_pix_seed=543, hot_percent=0.00001, hot_offset=1000,
    ):
        super().__init__()

        # starting empty image
        self.base_img = np.zeros(shape)

        # add read noise
        self.base_img += self.read_noise_gen(scale=read_noise/gain, size=shape)

        # add bias
        self.base_img += bias

        # add bad columns
        self.bad_cols_seed = bad_cols_seed
        self.__add_bad_cols(self.base_image, bad_columns, bias, n_bad_cols, bad_cols_seed)

        # add dark current
        current = dark_current * exposure_time/gain
        self.base_img = self.dark_current_gen(current, size=shape)

        # add hot pixels
        self.hot_pixel_seed = hot_pix_seed
        self.base_img = self.__add_hot_pixels(self.base_img, hot_pixels, hot_percent, hot_offset)

        # add sky counts
        self.base_img += self.sky_count_gen(sky_level*gain, shape)/gain

    def mock(self, hdu=None, **kwargs):
        # we do always have to return a new copy here, since sci images
        # are expected to be written on
        return self.base_img.copy()
