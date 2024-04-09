import numpy as np
from astropy.io.fits import (
    PrimaryHDU,
    CompImageHDU,
    ImageHDU,
    BinTableHDU,
    TableHDU
)
from astropy.modeling import models

from .config import Config, ConfigurationError


__all__ = [
    "add_model_objects",
    "DataFactory",
    "ZeroedData",
    "SimpleImage",
    "SimpleMask"
]


def add_model_objects(img, catalog, model):
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

            if all([
                    model.x_mean > 0,
                    model.x_mean < img.shape[1],
                    model.y_mean > 0,
                    model.y_mean < img.shape[0]
            ]):
                model.render(img)

    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return img




class DataFactoryConfig(Config):
    writeable = False
    """Sets the base array writeable flag."""

    copy_base = False
    """
    When `True`, a copy of the base data object is passed into the generate
    method, otherwise, the original (possibly mutable!) base data object is
    given.
    """

    copy_mocked = False
    """
    When `True`, the `DataFactory.mock` returns a copy of the final object,
    otherwise the original (possibly mutable!) object is returned.
    """

    return_copy = False


    isStatic = False
    """
    When `False` the `DataFactory.mock` will generate new data every time
    it is called. Otherwise it will memoize the result of the first time
    it's called and return that result directly or as a copy.
    """


class DataFactory:
    default_config = DataFactoryConfig

    def __init__(self, base=None, config=None, **kwargs):
        # not sure if this is "best" "best" way, but it does safe a lot of
        # array copies if we don't have to write to the mocked array
        # (which we shouldn't?). To be safe we set the writable flag to False
        # by default
        self.config = self.default_config()
        self.config.update(config, **kwargs)

        self.base = base
        if base is not None:
            self.base.flags.writeable = self.config.writeable
        self.counter = 0

    def mock(self, hdu=None, **kwargs):
        self.counter += 1
        if self.config.return_copy:
            return self.base.copy()
        return self.base



class SimpleVarianceConfig(DataFactoryConfig):
    read_noise = 0.0
    gain = 1.0
    calculate_base = True

class SimpleVariance(DataFactory):
    default_config = SimpleVarianceConfig

    def __init__(self, image=None, config=None, **kwargs):
        # skip setting the base here since the real base is
        # derived from given image we just set it manually
        super().__init__(base=None, config=config, **kwargs)

        if image is not None:
            self.base = image/self.config.gain + self.config.read_noise**2

    def mock(self, images=None):
        if images is None:
            return self.base
        return images/self.config.gain + self.config.read_noise**2


class SimpleMaskConfig(DataFactoryConfig):
    pass

class SimpleMask(DataFactory):
    default_config = SimpleMaskConfig
    def __init__(self, mask, config=None, **kwargs):
        super().__init__(base=mask, config=config, **kwargs)

    @classmethod
    def from_params(cls, shape, padding=0, bad_columns=[]):
        mask = np.zeros(shape)

        # padding
        mask[:padding] = 1
        mask[shape[0] - padding :] = 1
        mask[:, :padding] = 1
        mask[: shape[1] - padding :] = 1

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
                raise ValueError(f"Expected a tuple (x, y), (slice, slice) or slice, got {patch} instead.")

        return cls(mask)


class ZeroedDataConfig(DataFactoryConfig):
    shape = (5, 5)
    """Default image size."""

    # https://archive.stsci.edu/fits/fits_standard/node39.html#s:man
    bitpix_type_map = {
        # or char
        8: int,
        # actually no idea what dtype, or C type for that matter,
        # are used to represent these values. But default Headers return them
        16: np.float16,
        32: np.float32,
        64: np.float64,
        # classic IEEE float and double
        -32: np.float32,
        -64: np.float64,
    }
    """Map between FITS header BITPIX keyword value and NumPy return type."""

class ZeroedData(DataFactory):
    default_config = ZeroedDataConfig

    def __init__(self, base=None, config=None, **kwargs):
        super().__init__(base, config, **kwargs)

    def mock_image_data(self, hdu):
        cols = hdu.header.get("NAXIS1", False)
        rows = hdu.header.get("NAXIS2", False)
        shape = (cols, rows) if all((cols, rows)) else self.config.shape

        data = np.zeros(
            shape,
            dtype=self.config.bitpix_type_map[hdu.header["BITPIX"]]
        )
        return data

    def mock_table_data(self, hdu):
        # interestingly, table HDUs create their own empty
        # tables from headers, but image HDUs do not, this
        # is why hdu.data exists and has a valid dtype
        nrows = hdu.header["TFIELDS"]
        return np.zeros((nrows,), dtype=hdu.data.dtype)

    def mock(self, hdu=None, **kwargs):
        # two cases: a) static data shared by all instances created by the
        # factory set at init time or b) dynamic data generated from a header
        # specification at call-time.
        if self.base is not None:
            return super().mock(hdu, **kwargs)
        if isinstance(hdu, (PrimaryHDU, CompImageHDU, ImageHDU)):
            return self.mock_image_data(hdu)
        elif isinstance(hdu, (TableHDU, BinTableHDU)):
            return self.mock_table_data(hdu)
        else:
            raise TypeError(f"Expected an HDU, got {type(hdu)} instead.")



class SimpleImageConfig(DataFactoryConfig):
    shape = (100, 100)
    seed = None
    noise = 0
    noise_std = 1.0
    model = models.Gaussian2D


class SimpleImage(DataFactory):
    default_config = SimpleImageConfig

    def __init__(self, image=None, config=None, src_cat=None, **kwargs):
        super().__init__(image, config, **kwargs)

        if image is None:
            image = np.zeros(self.config.shape, dtype=np.float32)
        else:
            image = image
            self.config.shape = image.shape

        if src_cat is not None:
            add_model_objects(image, src_cat.table, self.config.model)

        self.base = image
        self._base_contains_data = image.sum() != 0

    @classmethod
    def add_noise(cls, n, images, config):
        rng = np.random.default_rng(seed=config.seed)
        shape = images.shape

        # noise has to be resampled for every image
        rng.standard_normal(size=shape, dtype=np.float32, out=images)

        # There's a lot of multiplications that happen, skip if possible
        if self.config.noise_std != 1.0:
            images *= config.noise_std
        images += config.noise

        return images

    def mock(self, n=1, obj_cats=None, **kwargs):
        shape = (n, *self.config.shape)
        images = np.zeros(shape, dtype=np.float32)

        if self.config.noise != 0:
            images = self.gen_noise(n, images, self.config)

        # but if base has no data (no sources, bad cols etc) skip
        if self._base_contains_data:
            images += self.base

        # same with moving objects
        if obj_cats is not None:
            for i, (img, cat) in enumerate(zip(images, obj_cats)):
                add_model_objects(
                    img,
                    cat,
                    self.config.model(x_stddev=1, y_stddev=1)
                )

        return images




class SimulatedImageConfig(DataFactoryConfig):
    # not sure this is a smart idea to put here
    rng = np.random.default_rng()

    # image properties
    shape = (100, 100)

    # detector properties
    read_noise_gen = rng.normal
    read_noise = 0
    gain = 1.0
    bias = 0.0

    add_bad_cols = False
    bad_cols_method = "random"
    bad_col_locs = []  # for manual setting of cols
    n_bad_cols = 5
    bad_cols_seed = 123
    bad_col_pattern_offset = 0.1

    dark_current_gen = rng.poisson
    dark_current = 0

    add_hot_pixels = False
    hot_pix_method = "random"
    hot_pix_locs = []
    hot_pix_seed = 321
    n_hot_pix = 10
    hot_pix_offset = 1000

    # Observation properties
    exposure_time = 120.0
    sky_count_gen = rng.poisson
    sky_level = 0

    # Object and Source properties
    model = models.Gaussian2D


class SimulatedImage(SimpleImage):
    default_config = SimulatedImageConfig

    @classmethod
    def add_bad_cols(cls, image, config):
        if not config.add_bad_cols:
            return image

        if config.bad_cols_method == "random":
            rng = np.random.RandomState(seed=self.bad_cols_seed)
            bad_cols = rng.randint(0, shape[1], size=config.n_bad_cols)
        elif config.bad_col_locs:
            bad_cols = config.bad_col_locs
        else:
            raise ConfigurationError("Bad columns method is not 'random', but `bad_col_locs` contains no bad column indices.")

        self.col_pattern = rng.randint(
            low=0,
            high=int(config.bad_col_pattern_offset * config.bias),
            size=shape[0]
        )

        for col in columns:
            image[:, col] = config.bias + col_pattern

        return image

    @classmethod
    def add_hot_pixels(cls, image, config):
        if not config.add_hot_pixels:
            return image

        if config.hot_pix_method == "random":
            rng = np.random.RandomState(seed=config.hot_pix_seed)
            x = rng.randint(0, shape[1], size=config.n_hot_pix)
            y = rng.randint(0, shape[0], size=config.n_hot_pix)
            hot_pixels = np.column_stack(x, y)
        elif config.hot_pix_locs:
            hot_pixels = pixels
        else:
            raise ConfigurationError("Hot pixels method is not 'random', but `hot_pix_locs` contains no (col, row) location indices of hot pixels.")

        for pix in hot_pixels:
            image[*pix] += image[*pix] * offset

        return image

    @classmethod
    def add_noise(cls, n, images, config):
        shape = images.shape

        # add read noise
        images += self.read_noise_gen(
            scale=config.read_noise / config.gain,
            size=shape
        )

        # add dark current
        current = config.dark_current * config.exposure_time / config.gain
        images += config.dark_current_gen(current, size=shape)

        # add sky counts
        images += self.sky_count_gen(
            lam=config.sky_level * config.gain,
            size=shape
        ) / config.gain

        return images

    @classmethod
    def gen_base_image(cls, config=None):
        config = cls.default_config(config)

        # empty image
        base = np.zeros(config.shape, dtype=np.float32)
        base += config.bias
        base = cls.add_bad_cols(base, config)
        base = cls.add_hot_pixels(base, config)

        return base

    def __init__(self, image=None, config=None, src_cat=None):
        conf = self.default_config(config)
        base = self.gen_base_image(conf)
        super().__init__(base, conf, src_cat)

#    def mock(self, n=1, obj_cats=None, **kwargs):
#        # we do always have to return a new copy here, since sci images
#        # are expected to be written on
#        return self.base_img.copy()
