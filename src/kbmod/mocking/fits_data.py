import numpy as np
from astropy.io.fits import PrimaryHDU, CompImageHDU, ImageHDU, BinTableHDU, TableHDU
from astropy.modeling import models

from .config import Config, ConfigurationError
from kbmod import Logging


__all__ = [
    "add_model_objects",
    "DataFactoryConfig",
    "DataFactory",
    "SimpleImageConfig",
    "SimpleImage",
    "SimpleMaskConfig",
    "SimpleMask",
    "SimulatedImageConfig",
    "SimulatedImage",
]


logger = Logging.getLogger(__name__)


def add_model_objects(img, catalog, model):
    """Adds a catalog of model objects to the image.

    Parameters
    ----------
    img : `np.array`
        Image.
    catalog : `astropy.table.QTable`
        Table of objects, a catalog
    model : `astropy.modelling.Model`
        Astropy's model of the surface brightness of an source. Must contain
        at least ``x_mean`` and ``y_mean``.

    Returns
    -------
    img: `np.array`
        Image including the rendenred models.
    """
    shape = img.shape
    yidx, xidx = np.indices(shape)

    # find catalog columns that exist for the model
    params_to_set = []
    for param in catalog.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    # Save the initial model parameters so we can set them back
    init_params = {param: getattr(model, param) for param in params_to_set}

    # model could throw a value error if drawn amplitude was too large, we must
    # restore the model back to its starting values to cover for a general
    # use-case because Astropy modelling is a bit awkward.
    try:
        for i, source in enumerate(catalog):
            for param in params_to_set:
                setattr(model, param, source[param])

            if all(
                [model.x_mean > 0, model.x_mean < img.shape[1], model.y_mean > 0, model.y_mean < img.shape[0]]
            ):
                model.render(img)
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return img


class DataFactoryConfig(Config):
    """Data factory configuration primarily controls mutability of the given
    and returned mocked datasets.
    """

    default_img_shape = (5, 5)

    default_img_bit_width = 32

    default_tbl_length = 5

    default_tbl_dtype = np.dtype([("a", int), ("b", int)])

    writeable = False
    """Sets the base array ``writeable`` flag. Default `False`."""

    return_copy = False
    """
    When `True`, the `DataFactory.mock` returns a copy of the final object,
    otherwise the original (possibly mutable!) object is returned. Default `False`.
    """

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


class DataFactory:
    """Generic data factory.

    Data can consists of tabular or array-like data of a single or compound
    types. Generally the process of creation of a mocked data is relatively
    simple:
    1) pre-generate whatever shared common base data is shared by all mocked
       data
    2) mock one or many instances of dynamic portions of the data and paste
       them onto the base static data.

    The base data can be writable or non-writable to prevent accidentally
    mutating its value. Some data factories consist of only the simple base
    data. In the case it is desired that the mocked data is further modified
    it is possible to return a copy of that base data, so that the base data is
    not mutated. By default it's assumed that all mocked data are allowed to
    share a single, non-mutable, base data.

    This saves memory and improves performance as, for example, there is no
    need to mock and allocate N (dimx, dimy) arrays for a shared static masking
    array shared by all associated images. Therefore the default implementation
    of mocking is to just continuously return the given base data.

    For dynamically created data, the data may depend on the type of the
    Header Data Unit (HDU), or dynamically changing header data. Data generated
    in these scenarios behaves as ``return_copy`` is always `True` regardless
    of the configuration value.

    Attributes
    ----------
    counter : `int`
        Data factory tracks an internal counter of generated objects that can
        be used as a ticker for generating new data.

    Parameters
    ----------
    base : `np.array`
        Static data shared by all mocked instances.
    config : `DataFactoryConfig`
        Configuration of the data factory.
    **kwargs :
        Additional keyword arguments are applied as configuration
        overrides.
    """

    default_config = DataFactoryConfig
    """Default configuration."""

    def __init__(self, base, config=None, **kwargs):
        self.config = self.default_config(config, **kwargs)

        self.base = base
        if base is None:
            self.shape = None
            self.dtype = None
        else:
            self.shape = base.shape
            self.dtype = base.dtype
            self.base.flags.writeable = self.config.writeable
        self.counter = 0

    @classmethod
    def gen_image(cls, metadata=None, config=None, **kwargs):
        conf = cls.default_config(config, method="subset", **kwargs)
        cols = metadata.get("NAXIS1", conf.default_img_shape[0])
        rows = metadata.get("NAXIS2", conf.default_img_shape[1])
        bitwidth = metadata.get("BITPIX", conf.default_img_bit_width)
        dtype = conf.bitpix_type_map[bitwidth]
        shape = (cols, rows)

        return np.zeros(shape, dtype)

    @classmethod
    def gen_table(cls, metadata=None, config=None, **kwargs):
        conf = cls.default_config(config, **kwargs, method="subset")

        # FITS format standards prescribe FORTRAN-77-like input format strings
        # for different types, but the base set has been extended significantly
        # since and Rubin uses completely non-standard keys with support for
        # their own internal abstractions like 'Angle' objects:
        # https://archive.stsci.edu/fits/fits_standard/node58.html
        # https://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
        # https://github.com/lsst/afw/blob/main/src/fits.cc#L207
        # So we really don't have much of a choice but to force a default
        # AstroPy HDU and then call the update. This might not preserve the
        # header or the data formats exactly and if metadata isn't given
        # could even assume a wrong class all together. The TableHDU is
        # almost never used however - so hopefully this keeps on working.
        table_cls = BinTableHDU
        data = None
        if metadata is not None:
            if metadata["XTENSION"] == "BINTABLE":
                table_cls = BinTableHDU
            elif metadata["XTENSION"] == "TABLE":
                table_cls = TableHDU

            hdu = table_cls()
            hdu.header.update(metadata)

            rows = metadata.get("NAXIS2", conf.default_tbl_length)
            shape = (rows,)
            data = np.zeros(shape, dtype=hdu.data.dtype)
        else:
            hdu = table_cls()
            shape = (conf.default_tbl_length,)
            data = np.zeros(shape, dtype=conf.default_tbl_dtype)

        return data

    @classmethod
    def from_hdu(cls, hdu, config=None, **kwargs):
        if isinstance(hdu, (PrimaryHDU, CompImageHDU, ImageHDU)):
            return cls(base=cls.gen_image(hdu), config=config, **kwargs)
        elif isinstance(hdu, (TableHDU, BinTableHDU)):
            return cls(base=cls.gen_table(hdu), config=config, **kwargs)
        else:
            raise TypeError(f"Expected an HDU, got {type(hdu)} instead.")

    @classmethod
    def from_header(cls, kind, header, config=None, **kwargs):
        if kind.lower() == "image":
            return cls(base=cls.gen_image(header), config=config, **kwargs)
        elif kind.lower() == "table":
            return cls(base=cls.gen_table(header), config=config, **kwargs)
        else:
            raise TypeError(f"Expected an 'image' or 'table', got {kind} instead.")

    @classmethod
    def zeros(cls, shape, dtype, config=None, **kwargs):
        return cls(np.zeros(shape, dtype), config, **kwargs)

    def mock(self, n=1, **kwargs):
        if self.base is None:
            raise ValueError(
                "Expected a DataFactory that has a base, but none was set. "
                "Use `zeros` or `from_hdu` to construct this object correctly."
            )

        if self.config.return_copy:
            base = np.repeat(self.base[np.newaxis,], (n,), axis=0)
        else:
            base = np.broadcast_to(self.base, (n, *self.shape))
        base.flags.writeable = self.config.writeable

        return base


class SimpleVarianceConfig(DataFactoryConfig):
    """Configure noise and gain of a simple variance factory."""

    read_noise = 0.0
    """Read noise"""

    gain = 1.0
    "Gain."


class SimpleVariance(DataFactory):
    """Simple variance factory.

    Variance is calculated as the

        variance = image/gain + read_noise^2

    thus variance has to be calculated for each individual mocked image.

    Parameters
    ----------
    image : `np.array`
        Science image from which the variance will be derived from.
    config : `DataFactoryConfig`
        Configuration of the data factory.
    **kwargs :
        Additional keyword arguments are applied as config
        overrides.
    """

    default_config = SimpleVarianceConfig

    def __init__(self, image=None, config=None, **kwargs):
        # skip setting the base here since the real base is
        # derived from given image we just set it manually
        super().__init__(base=None, config=config, **kwargs)

        if image is not None:
            self.base = image / self.config.gain + self.config.read_noise**2

    def mock(self, images=None):
        if images is None:
            return self.base
        return images / self.config.gain + self.config.read_noise**2


class SimpleMaskConfig(DataFactoryConfig):
    """Simple mask configuration."""

    dtype = np.float32

    threshold = 1e-05

    shape = (5, 5)
    padding = 0
    bad_columns = []
    patches = []


class SimpleMask(DataFactory):
    """Simple mask factory.

    Masks are assumed to be shared, static data. To create an instance of this
    factory use one of the provided class methods. Created mask will correspond
    to a bitmask already appropriate for use with KBMOD.

    Parameters
    ----------
    mask : `np.array`
        Bitmask array.
    """

    default_config = SimpleMaskConfig

    def __init__(self, mask, config=None, **kwargs):
        super().__init__(base=mask, config=config, **kwargs)

    @classmethod
    def from_image(cls, image, config=None, **kwargs):
        config = cls.default_config(config=config, **kwargs, method="subset")
        mask = image.copy()
        mask[image > config.threshold] = 1
        return cls(mask)

    @classmethod
    def from_params(cls, config=None, **kwargs):
        """Create a mask by adding a padding around the edges of the array with
        the given dimensions and mask out bad columns.

        Parameters
        ----------
        shape : `tuple`
            Tuple of (width, height)/(cols, rows) dimensions of the mask.
        padding : `int`
            Number of pixels near the edge that will be masked.
        bad_columns : `list[int]`
            Indices of bad columns to mask
        patches : `list[tuple]`
            Patches to mask. This is a list of tuples. Each tuple consists of
            a patch and a value. The patch can be any combination of array
            coordinates such as ``(int, int)`` for individual pixels,
           ``(slice, int)`` or ``(int, slice)`` for columns and rows
           respectively  or ``(slice, slice)`` for areas.

        Returns
        -------
        mask_factory : `SimpleMask`
            Mask factory.

        Examples
        --------
        >>> SimpleMask.from_params(
                shape=(10, 10),
                padding=1,
                bad_columns=[2, 3],
                patches=[
                    ((5, 5), 2),
                    ((slice(6, 8), slice(6, 8)), 3)
                ]
            )
        array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
               [1., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
               [1., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
               [1., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
               [1., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
               [1., 0., 1., 1., 0., 1., 0., 0., 0., 1.],
               [1., 0., 1., 1., 0., 0., 1., 1., 0., 1.],
               [1., 0., 1., 1., 0., 0., 1., 1., 0., 1.],
               [1., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        """
        config = cls.default_config(config=config, **kwargs, method="subset")
        mask = np.zeros(config.shape, dtype=config.dtype)

        shape, padding = config.shape, config.padding

        # padding
        mask[:padding] = 1
        mask[shape[0] - padding :] = 1
        mask[:, :padding] = 1
        mask[: shape[1] - padding :] = 1

        # bad columns
        for col in config.bad_columns:
            mask[:, col] = 1

        for patch, value in config.patches:
            if isinstance(patch, tuple):
                mask[patch] = 1
            elif isinstance(slice):
                mask[slice] = 1
            else:
                raise ValueError(f"Expected a tuple (x, y), (slice, slice) or slice, got {patch} instead.")

        return cls(mask, config=config)


class SimpleImageConfig(DataFactoryConfig):
    """Simple image configuration."""

    return_copy = True

    shape = (100, 100)
    """Dimensions of the generated images."""

    add_noise = False
    """Add noise to the base image."""

    seed = None
    """Seed of the random number generator used to generate noise."""

    noise = 10
    """Mean of the standard Gaussian distribution of the noise."""

    noise_std = 1.0
    """Standard deviation of the Gaussian distribution of the noise."""

    model = models.Gaussian2D
    """Source and object model used to render them on the image."""


class SimpleImage(DataFactory):
    """Simple image data factory.

    Simple image consists of an blank empty base, onto which noise, sources
    and objects can be added. All returned images act as a copy of the base
    image.

    Noise realization is drawn from a Gaussian distribution with the given
    standard deviation and centered on the given mean.

    Parameters
    ----------
    image : `np.array`
        Science image that will be used as a base onto which to render details.
    config : `SimpleImageConfig`
        Configuration.
    src_cat : `CatalogFactory`
        Static source catalog.
    obj_cat : `CatalogFactory`
        Moving object catalog factory.
    **kwargs :
        Additional keyword arguments are applied as config
        overrides.
    """

    default_config = SimpleImageConfig

    def __init__(self, image=None, src_cat=None, obj_cat=None, config=None, **kwargs):
        self.config = self.default_config(config=config, **kwargs)
        super().__init__(image, self.config, **kwargs)

        if image is None:
            image = np.zeros(self.config.shape, dtype=np.float32)
        else:
            image = image
            self.config.shape = image.shape

        # Astropy throws a strange ValueError instead of reporting a non-writeable
        # array, This must be a bug TODO: report. It's not safe to edit a
        # non-writeable array and then revert writeability so make a copy.
        self.src_cat = src_cat
        if self.src_cat is not None:
            image = image if image.flags.writeable else image.copy()
            add_model_objects(image, src_cat.table, self.config.model(x_stddev=1, y_stddev=1))
            image.flags.writeable = self.config.writeable

        self.base = image
        self._base_contains_data = image.sum() != 0

    @classmethod
    def add_noise(cls, images, config):
        """Adds gaussian noise to the images.

        Parameters
        ----------
        images : `np.array`
           A ``(n_images, image_width, image_height)`` shaped array of images.
        config : `SimpleImageConfig`
           Configuration.
        """
        rng = np.random.default_rng(seed=config.seed)
        shape = images.shape

        # noise has to be resampled for every image
        rng.standard_normal(size=shape, dtype=images.dtype, out=images)

        # There's a lot of multiplications that happen, skip if possible
        if config.noise_std != 1.0:
            images *= config.noise_std
        images += config.noise

        return images

    def mock(self, n=1, obj_cats=None, **kwargs):
        """Creates a set of images.

        Parameters
        ----------
        n : `int`, optional
            Number of images to create, default: 1.
        obj_cats : `list[Catalog]`
            A list of catalogs as long as the number of requested images of
            moving objects that will be inserted into the image.
        """
        shape = (n, *self.config.shape)
        images = np.zeros(shape, dtype=np.float32)

        if self.config.add_noise:
            images = self.add_noise(images=images, config=self.config)

        # if base has no data (no sources, bad cols etc) skip
        if self._base_contains_data:
            images += self.base

        # same with moving objects, each image has to have a new realization of
        # a catalog in which moving objects have different coordinate. This way
        # any trajectory can be mocked. When we have only 1 mocked image though
        # zip will attempt to iterate over the next available dimension, and
        # that's rows of the image and the table - we don't want that.
        if obj_cats is not None:
            pairs = ([(images[0], obj_cats[0])]  if n == 1 else zip(images, obj_cats))
            for i, (img, cat) in enumerate(pairs):
                add_model_objects(img, cat, self.config.model(x_stddev=1, y_stddev=1))

        return images


class SimulatedImageConfig(DataFactoryConfig):
    """Simulated image configuration.

    Simulated image add several sources of noise into the image, bad columns,
    hot pixels, including the possibility of adding static and moving sources.
    On a higher level it is possible to modify this behavior by setting:
    - add_noise
    - add_bad_cols
    - add_hot_pix
    parameters to `False`. Optionally, for a more fine-grained control set the
    distribution parameters, f.e. mean and standard deviation, such that they
    do not produce measurable values in the image.

    The quantities are expressed in physical units and the defaults were
    selected to sort of make sense. Sky levels are not modeled very
    realistically, being expressed as an additional level of noise, in counts,
    additive to the detector noise.

    Generally expect the mean value of pixel counts to be:

        bias + mean(dark_current)*exposure + mean(sky_level)

    The deviation of the pixel counts around the expected value can be
    estimated by:

        sqrt( std(read_noise)^2 + sqrt(sky_level)^2 )
    """

    # not sure this is a smart idea to put here
    rng = np.random.default_rng()

    seed = None
    """Random number generator seed shared by all number generators."""

    # image properties
    shape = (100, 100)
    """Dimensions of the created images."""

    # detector properties
    add_noise = True
    """Add noise (includes read noise, dark current and sky) to the image."""

    read_noise_gen = rng.normal
    """Read noise follows a Gaussian distribution."""

    read_noise = 5
    """Standard deviation of read noise distribution, in electrons."""

    gain = 1.0
    """Gain in electrons/count."""

    bias = 0.0
    """Bias in counts."""

    add_bad_columns = False
    """Add bad columns to the image."""

    bad_cols_method = "random"
    """Method by which bad columns are picked. If not 'random', 'bad col_locs'
    must be provided."""

    bad_col_locs = []
    """Location, column indices, of bad columns."""

    n_bad_cols = 5
    """When bad columns method is random, sets the number of bad columns."""

    bad_cols_seed = seed
    """Seed for the bad columns random number generator."""

    bad_col_offset = 100
    """Bad column signal offset (in counts) with regards to the baseline noise."""

    bad_col_pattern_offset = 10
    """Random-looking noise variation along the length of the bad columns is
    offset from the mean bad column counts by this amount."""

    dark_current_gen = rng.poisson
    """Dark current follows a Poisson distribution."""

    dark_current = 0.1
    """Dark current mean in electrons/pixel/sec. Typically ~0.1-0.2."""

    add_hot_pixels = False
    """Simulate hot pixels."""

    hot_pix_method = "random"
    """When `random` the hop pixels are selected randomly, otherwise their
    indices must be provided."""

    hot_pix_locs = []
    """A `list[tuple]` of hot pixel indices."""

    hot_pix_seed = seed
    """Seed for hot pixel random number generator."""

    n_hot_pix = 10
    """Number of hot pixels to insert into the image."""

    hot_pix_offset = 10000
    """Offset of hot pixel counts from the baseline. Usally a very large number."""

    # Observation properties
    exposure_time = 120.0
    """Exposure time of the simulated image, affects noise properties."""

    sky_count_gen = rng.poisson
    """Sky background random number generator."""

    sky_level = 20
    """Sky level, in counts."""

    # Object and Source properties
    model = models.Gaussian2D


class SimulatedImage(SimpleImage):
    """Simulated image includes multiple sources of noise, bad columns, hot
    pixels, static sources and moving objects.

    Parameters
    ----------
    image : `np.array`
        Science image that will be used as a base onto which to render details.
    config : `SimpleImageConfig`
        Configuration.
    src_cat : `CatalogFactory`
        Static source catalog.
    obj_cat : `CatalogFactory`
        Moving object catalog factory.
    **kwargs :
        Additional keyword arguments are applied as config
        overrides.
    """

    default_config = SimulatedImageConfig

    @classmethod
    def add_bad_cols(cls, image, config):
        """Adds bad columns to the image based on the configuration.

        Columns can be sampled randomly, or a list of bad column indices can
        be provided.

        Parameters
        ----------
        image : `np.array`
            Image.
        config : `SimpleImageConfig`
            Configuration.

        Returns
        -------
        image : `np.array`
            Image.
        """
        if not config.add_bad_columns:
            return image

        shape = image.shape
        rng = np.random.RandomState(seed=config.bad_cols_seed)
        if config.bad_cols_method == "random":
            bad_cols = rng.randint(0, shape[1], size=config.n_bad_cols)
        elif config.bad_col_locs:
            bad_cols = config.bad_col_locs
        else:
            raise ConfigurationError(
                "Bad columns method is not 'random', but `bad_col_locs` " "contains no column indices."
            )

        col_pattern = rng.randint(low=0, high=int(config.bad_col_pattern_offset), size=shape[0])

        for col in bad_cols:
            image[:, col] += col_pattern + config.bad_col_offset

        return image

    @classmethod
    def add_hot_pixels(cls, image, config):
        """Adds hot pixels to the image based on the configuration.

        Indices of hot pixels can be sampled randomly, or a list of hot pixel
        indices can be provided.

        Parameters
        ----------
        image : `np.array`
            Image.
        config : `SimpleImageConfig`
            Configuration.

        Returns
        -------
        image : `np.array`
            Image.
        """
        if not config.add_hot_pixels:
            return image

        shape = image.shape
        if config.hot_pix_method == "random":
            rng = np.random.RandomState(seed=config.hot_pix_seed)
            x = rng.randint(0, shape[1], size=config.n_hot_pix)
            y = rng.randint(0, shape[0], size=config.n_hot_pix)
            hot_pixels = np.column_stack([x, y])
        elif config.hot_pix_locs:
            hot_pixels = pixels
        else:
            raise ConfigurationError(
                "Hot pixels method is not 'random', but `hot_pix_locs` contains "
                "no (col, row) location indices of hot pixels."
            )

        for pix in hot_pixels:
            image[*pix] += config.hot_pix_offset

        return image

    @classmethod
    def add_noise(cls, images, config):
        """Adds read noise (gaussian), dark noise (poissonian) and sky
        background (poissonian) noise to the image.

        Parameters
        ----------
        image : `np.array`
            Image.
        config : `SimpleImageConfig`
            Configuration.

        Returns
        -------
        image : `np.array`
            Image.
        """
        shape = images.shape

        # add read noise
        images += config.read_noise_gen(scale=config.read_noise / config.gain, size=shape)

        # add dark current
        current = config.dark_current * config.exposure_time / config.gain
        images += config.dark_current_gen(current, size=shape)

        # add sky counts
        images += config.sky_count_gen(lam=config.sky_level * config.gain, size=shape) / config.gain

        return images

    @classmethod
    def gen_base_image(cls, config=None, src_cat=None):
        """Generate base image from configuration.

        Parameters
        ----------
        config : `SimpleImageConfig`
            Configuration.
        src_cat : `CatalogFactory`
            Static source catalog.

        Returns
        -------
        image : `np.array`
            Image.
        """
        config = cls.default_config(config)

        # empty image
        base = np.zeros(config.shape, dtype=np.float32)
        base += config.bias
        base = cls.add_hot_pixels(base, config)
        base = cls.add_bad_cols(base, config)
        if src_cat is not None:
            add_model_objects(base, src_cat.table, config.model(x_stddev=1, y_stddev=1))

        return base

    def __init__(self, image=None, config=None, src_cat=None, obj_cat=None, **kwargs):
        conf = self.default_config(config=config, **kwargs)
        # static objects are added in SimpleImage init
        super().__init__(image=self.gen_base_image(conf), config=conf, src_cat=src_cat, obj_cat=obj_cat)
