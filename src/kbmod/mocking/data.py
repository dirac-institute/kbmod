import numpy as np
from astropy.io.fits import PrimaryHDU, CompImageHDU, ImageHDU, BinTableHDU, TableHDU
from astropy.modeling import models

from .config import Config, ConfigurationError
from kbmod import Logging


__all__ = [
    "add_model_objects",
    "DataFactory",
    "SimpleImage",
    "SimpleMask",
    "SimpleVariance",
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
        Astropy's model of the surface brightness of an source.

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
            model.render(img)
    except ValueError as e:
        # ignore rendering models larger than the image
        message = "The `bounding_box` is larger than the input out in one or more dimensions."
        if message in str(e):
            pass
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return img


class DataFactoryConfig(Config):
    """Data factory configuration primarily controls mutability of the given
    and returned mocked datasets.
    """

    default_img_shape = (5, 5)
    """Default image size, used if mocking ImageHDU or CompImageHDUs."""

    default_img_bit_width = 32
    """Default image data type is float32; the value of BITPIX flag in headers.
    See bitpix_type_map for other codes.
    """

    default_tbl_length = 5
    """Default table length, used if mocking BinTableHDU or TableHDU HDUs."""

    default_tbl_dtype = np.dtype([("a", int), ("b", int)])
    """Default table dtype, used when mocking table-HDUs that do not contain
    a description of table layout.
    """

    writeable = False
    """Sets the base array ``writeable`` flag. Default `False`."""

    return_copy = False
    """When `True`, the `DataFactory.mock` returns a copy of the final object,
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
    """Generic data factory that can mock table and image HDUs from default
    settings or given header definitions.

    Given a template, this factory repeats it for each mock.
    A reference to the base template is returned whenever possible for
    performance reasons. To prevent accidental mutation of the shared
    array, the default behavior is that the returned data is not writable.

    A base template value of `None` is accepted as valid to satisfy FITS
    factory use-case of generating HDUList stubs containing only headers.

    Primary purpose of this factory is to derive the template data given a
    table, HDU or a Header object. When the base has no data, but just a
    description of one, such as Headers, the default is to return "zeros"
    for that datatype. This can be a zero length string, literal integer
    zero, a float zero etc...

    Attributes
    ----------
    base : `np.array`, `np.recarray` or `None`
        Base data template.
    shape : `tuple`
        Shape of base array when it exists.
    dtype : `type`
        Numpy type of the base array, when it exists.
    counter : `int`
        Data factory tracks an internal counter of generated objects that can
        be used as a ticker for generating new data.

    Parameters
    ----------
    base : `np.array`
        Static data shared by all mocked instances.
    kwargs :
        Additional keyword arguments are applied as configuration
        overrides.

    Examples
    --------
    >>> from astropy.io.fits import Header, CompImageHDU, BinTableHDU
    >>> import kbmod.mocking as kbmock
    >>> import numpy as np
    >>> base = np.zeros((2, 2))
    >>> hdu = CompImageHDU(base)
    >>> kbmock.DataFactory.from_hdu(hdu).mock()
    array([[[0., 0.],
            [0., 0.]]])
    >>> kbmock.DataFactory.from_header("image", hdu.header).mock()
    array([[[0., 0.],
            [0., 0.]]])
    >>> base = np.array([("test1", 10), ("test2", 11)], dtype=[("col1", "U5"), ("col2", int)])
    >>> hdu = BinTableHDU(base)
    >>> kbmock.DataFactory.from_hdu(hdu).mock()
    array([[(b'test1', 10), (b'test2', 11)]],
          dtype=(numpy.record, [('col1', 'S5'), ('col2', '<i8')]))
    >>> kbmock.DataFactory.from_header("table", hdu.header).mock()
    array([[(b'', 0), (b'', 0)]],
          dtype=(numpy.record, [('col1', 'S5'), ('col2', '>i8')]))
    """

    default_config = DataFactoryConfig
    """Default configuration."""

    def __init__(self, base, **kwargs):
        self.config = self.default_config(**kwargs)

        self.base = base
        if self.base is not None:
            self.shape = base.shape
            self.dtype = base.dtype
            self.base.flags.writeable = self.config["writeable"]
        self.counter = 0

    @classmethod
    def gen_image(cls, header=None, **kwargs):
        """Generate an image from a complete or partial header and config.

        If a header is given, it trumps the default config values. When the
        header is not complete, config values are used. Config overrides are
        applied before the data description is evaluated.

        Parameters
        ----------
        header : `None`, `Header` or dict-like, optional
            Header, or dict-like object, containing the image-data descriptors.
        kwargs :
            Any additional keyword arguments are applied as config overrides.

        Returns
        -------
        image : `np.array`
            Image
        """
        conf = cls.default_config(**kwargs)
        metadata = {} if header is None else header
        cols = metadata.get("NAXIS1", conf["default_img_shape"][0])
        rows = metadata.get("NAXIS2", conf["default_img_shape"][1])
        bitwidth = metadata.get("BITPIX", conf["default_img_bit_width"])
        dtype = conf.bitpix_type_map[bitwidth]
        shape = (cols, rows)
        return np.zeros(shape, dtype)

    @classmethod
    def gen_table(cls, metadata=None, **kwargs):
        """Generate an table from a complete or partial header and config.

        If a header is given, it trumps the default config values. When the
        header is not complete, config values are used. Config overrides are
        applied before the data description is evaluated.

        Parameters
        ----------
        header : `None`, `Header` or dict-like, optional
            Header, or dict-like object, containing the image-data descriptors.
        kwargs :
            Any additional keyword arguments are applied as config overrides.

        Returns
        -------
        table : `np.array`
            Table, a structured array.

        Notes
        -----
        FITS format standards prescribe FORTRAN-77-like input format strings
        for different data types, but the base set has been extended and/or
        altered significantly by various pipelines to support their objects
        internal to their pipelines. Constructing objects, or values, described
        by non-standard strings will result in a failure. For a list of supported
        column-types see:
        https://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
        """
        conf = cls.default_config(**kwargs)

        # https://github.com/lsst/afw/blob/main/src/fits.cc#L207
        # So we really don't have much of a choice but to force a default
        # AstroPy HDU and then call the update. This might not preserve the
        # header or the data formats exactly and if metadata isn't given
        # could even assume a wrong class all together. The TableHDU is
        # almost never used by us however - so hopefully this keeps on working.
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
    def from_hdu(cls, hdu, **kwargs):
        """Create the factory from an HDU with or without data and with or
        without a complete Header.

        If the given HDU has data, it is preferred over creating a zero-array
        based on the header. If the header is not complete, config defaults are
        used. Config overrides are applied beforehand.

        Parameters
        ----------
        hdu : `HDU`
            One of AstroPy's Header Data Unit classes.
        kwargs :
            Config overrides.

        Returns
        -------
        data : `np.array`
            Data array, an ndarray or a recarray depending on the HDU.
        """
        if isinstance(hdu, (PrimaryHDU, CompImageHDU, ImageHDU)):
            base = hdu.data if hdu.data is not None else cls.gen_image(hdu.header)
            return cls(base=base, **kwargs)
        elif isinstance(hdu, (TableHDU, BinTableHDU)):
            base = hdu.data if hdu.data is not None else cls.gen_table(hdu.header)
            return cls(base=base, **kwargs)
        else:
            raise TypeError(f"Expected an HDU, got {type(hdu)} instead.")

    @classmethod
    def from_header(cls, header, kind=None, **kwargs):
        """Create the factory from an complete or partial Header.

        Provide the ``kind`` of data the header represents in situations where
        the Header does not have an well defined ``XTENSION`` card.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            Header
        kind : `str` or `None`, optional
            Kind of data the header is representing.
        kwargs :
            Config overrides.

        Returns
        -------
        data : `np.array`
            Data array, an ndarray or a recarray depending on the Header and kind.
        """
        hkind = header.get("XTENSION", False)
        if hkind and "table" in hkind.lower():
            kind = "table"
        elif hkind and "image" in hkind.lower():
            kind = "image"
        elif kind is None:
            raise ValueError("Must provide a header with XTENSION or ``kind``")
        else:
            # kind was defined as keyword arg, so all is right
            pass

        if kind.lower() == "image":
            return cls(base=cls.gen_image(header), **kwargs)
        elif kind.lower() == "table":
            return cls(base=cls.gen_table(header), **kwargs)
        else:
            raise TypeError(f"Expected an 'image' or 'table', got {kind} instead.")

    def mock(self, n=1):
        """Mock one or multiple data arrays.

        Parameters
        ----------
        n : `int`
            Number of data to mock.
        """
        if self.base is None:
            raise ValueError(
                "Expected a DataFactory that has a base, but none was set. "
                "Use `zeros` or `from_hdu` to construct this object correctly."
            )

        if self.config["return_copy"]:
            base = np.repeat(self.base[np.newaxis,], (n,), axis=0)
        else:
            base = np.broadcast_to(self.base, (n, *self.shape))
            base.flags.writeable = self.config["writeable"]

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

    Examples
    --------
    >>> import kbmod.mocking as kbmock
    >>> si = kbmock.SimpleImage(shape=(3, 3), add_noise=True, seed=100)
    >>> sv = kbmock.SimpleVariance(gain=10)
    >>> imgs = si.mock()
    >>> imgs
    array([[[ 8.694266,  9.225379, 10.046582],
            [ 8.768851, 10.201585,  8.870326],
            [10.702058,  9.910087,  9.283925]]], dtype=float32)
    >>> sv.mock(imgs)
    array([[[0.8694266 , 0.9225379 , 1.0046582 ],
            [0.8768851 , 1.0201585 , 0.8870326 ],
            [1.0702058 , 0.99100864, 0.9283925 ]]], dtype=float32)
    """

    default_config = SimpleVarianceConfig

    def __init__(self, image=None, **kwargs):
        super().__init__(base=image, **kwargs)
        if image is not None:
            self.base = image / self.config["gain"] + self.config["read_noise"] ** 2

    def mock(self, images=None):
        """Mock one or multiple data arrays.

        Parameters
        ----------
        images : `list[np.array]`, optional
            List, or otherwise a collection, of images from which the variances
            will be generated. When not provided, and base template was
            defined, returns the base template.
        """
        if images is None:
            return self.base
        return images / self.config["gain"] + self.config["read_noise"] ** 2


class SimpleMaskConfig(DataFactoryConfig):
    """Simple mask configuration."""

    dtype = np.float32
    """Data type"""

    threshold = 1e-05
    """Default pixel value threshold above which every pixel in the template
    will be masked.
    """

    shape = (5, 5)
    """Default image shape."""

    padding = 0
    """Number of pixels near the edge that are masked."""

    bad_columns = []
    """List of columns marked as bad."""

    patches = []
    """Default patches to mask. This is a list of tuples. Each tuple consists of
    a patch and a value. The patch can be any combination of array coordinates
    such as ``(int, int)`` for individual pixels, ``(slice, int)`` or
    ``(int, slice)`` for columns and rows respectively  or ``(slice, slice)``
    for areas. See `SimpleMask.from_params` for an example.
    """


class SimpleMask(DataFactory):
    """Simple mask factory.

    Masks are assumed to be shared, static data. To create an instance of this
    factory use one of the provided class methods. Created mask will correspond
    to a bitmask already appropriate for use with KBMOD.

    Parameters
    ----------
    mask : `np.array`
        Bitmask array.
    kwargs :
        Config overrides.

    Examples
    --------
    >>> import kbmod.mocking as kbmock
    >>> si = kbmock.SimpleImage(shape=(3, 3), add_noise=True, seed=100)
    >>> imgs = si.mock()
    >>> imgs
    array([[[ 8.694266,  9.225379, 10.046582],
            [ 8.768851, 10.201585,  8.870326],
            [10.702058,  9.910087,  9.283925]]], dtype=float32)
    >>> sm = kbmock.SimpleMask.from_image(imgs, threshold=9)
    >>> sm.base
    array([[[0., 1., 1.],
            [0., 1., 0.],
            [1., 1., 1.]]], dtype=float32)
    """

    default_config = SimpleMaskConfig
    """Default configuration."""

    def __init__(self, mask, **kwargs):
        super().__init__(base=mask, **kwargs)

    @classmethod
    def from_image(cls, image, **kwargs):
        """Create a factory instance out of an image, masking all pixels above
        a threshold.

        Parameters
        ----------
        image : `np.array`
            Template image from which a mask is created.
        kwargs :
            Config overrides.
        """
        config = cls.default_config(**kwargs)
        mask = image.copy()
        mask[image > config["threshold"]] = 1
        mask[image <= config["threshold"]] = 0
        return cls(mask)

    @classmethod
    def from_params(cls, **kwargs):
        """Create a factory instance from config parameters.

        Parameters
        ----------
        kwargs :
            Config overrides.

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
        config = cls.default_config(**kwargs)
        mask = np.zeros(config["shape"], dtype=config["dtype"])

        shape, padding = config["shape"], config["padding"]

        # padding
        mask[:padding] = 1
        mask[shape[0] - padding :] = 1
        mask[:, :padding] = 1
        mask[: shape[1] - padding :] = 1

        # bad columns
        for col in config["bad_columns"]:
            mask[:, col] = 1

        for patch, value in config["patches"]:
            if isinstance(patch, tuple):
                mask[patch] = 1
            elif isinstance(slice):
                mask[slice] = 1
            else:
                raise ValueError(f"Expected a tuple (x, y), (slice, slice) or slice, got {patch} instead.")

        return cls(mask, **config)


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

    model = models.Gaussian2D(x_stddev=1, y_stddev=1)
    """Source and object model used to render them on the image."""

    dtype = np.float32
    """Numpy data type."""


class SimpleImage(DataFactory):
    """Simple image data factory.

    Simple image consists of an blank empty base, onto which noise, sources
    and objects can be added. All returned images act as a copy of the base
    image.

    Noise realization is drawn from a Gaussian distribution with the given
    standard deviation and mean.

    Parameters
    ----------
    image : `np.array`
        Science image that will be used as a base onto which to render details.
    src_cat : `CatalogFactory`
        Static source catalog.
    obj_cat : `CatalogFactory`
        Moving object catalog factory.
    kwargs :
        Additional keyword arguments are applied as config.
        overrides.

    Examples
    --------
    >>> import kbmod.mocking as kbmock
    >>> si = kbmock.SimpleImage()
    >>> si.mock()
    array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]], dtype=float32)
    >>> si = kbmock.SimpleImage(shape=(3, 3), add_noise=True, seed=100)
    >>> si.mock()
    array([[[ 8.694266,  9.225379, 10.046582],
            [ 8.768851, 10.201585,  8.870326],
            [10.702058,  9.910087,  9.283925]]], dtype=float32)
    """

    default_config = SimpleImageConfig
    """Default configuration."""

    def __init__(self, image=None, src_cat=None, obj_cat=None, **kwargs):
        super().__init__(image, **kwargs)

        if image is None:
            image = np.zeros(self.config["shape"], dtype=self.config["dtype"])
        else:
            image = image
            self.config["shape"] = image.shape

        # Astropy throws a strange ValueError instead of reporting a non-writeable
        # array, This must be a bug TODO: report. It's not safe to edit a
        # non-writeable array and then revert writeability so make a copy.
        self.src_cat = src_cat
        if self.src_cat is not None:
            image = image if image.flags.writeable else image.copy()
            add_model_objects(image, src_cat.table, self.config["model"])
            image.flags.writeable = self.config["writeable"]

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

        Returns
        -------
        images : `np.array`
           A ``(n_images, image_width, image_height)`` shaped array of images.
        """
        rng = np.random.default_rng(seed=config["seed"])
        shape = images.shape

        # noise has to be resampled for every image
        rng.standard_normal(size=shape, dtype=images.dtype, out=images)

        # There's a lot of multiplications that happen, skip if possible
        if config["noise_std"] != 1.0:
            images *= config["noise_std"]
        images += config["noise"]

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

        Returns
        -------
        images : `np.array`
           A ``(n_images, image_width, image_height)`` shaped array of images.
        """
        shape = (n, *self.config["shape"])
        images = np.zeros(shape, dtype=self.config["dtype"])

        if self.config["add_noise"]:
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
            pairs = [(images[0], obj_cats[0])] if n == 1 else zip(images, obj_cats)
            for i, (img, cat) in enumerate(pairs):
                add_model_objects(img, cat, self.config["model"])

        return images


class SimulatedImageConfig(DataFactoryConfig):
    """Simulated image configuration.

    Simulated image attempts to add noise to the image in a statistically
    meaningful sense, but it does not reproduce the noise qualities in the same
    way an optical simulation would. Noise sources added are:
    - bad columns
    - hot pixels
    - read noise
    - dark current
    - sky level

    The quantities are expressed in physical units and the defaults were
    selected to sort of make sense.

    Control over which source of noise are included in the image can be done by
    setting the
    - add_noise
    - add_bad_cols
    - add_hot_pix
    flags to `False`. For a more fine-grained control set the distribution
    parameters, f.e. mean and standard deviation, such that they do not produce
    measurable values in the image.

    Expect the mean value of pixel counts to be:

        bias + mean(dark_current)*exposure + mean(sky_level)

    The deviation of the pixel counts should be expected to be:

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
    model = models.Gaussian2D(x_stddev=1, y_stddev=1)
    """Source and object model used to render them on the image."""

    dtype = np.float32
    """Numpy data type."""


class SimulatedImage(SimpleImage):
    """Simulated image attempt to include a more realistic noise profile.

    Noise sources added are:
    - bad columns
    - hot pixels
    - read noise
    - dark current
    - sky level

    Static or moving objects may be added to the simulated image.

    Parameters
    ----------
    image : `np.array`
        Base template image on which details will be rendered.
    src_cat : `CatalogFactory`
        Static source catalog.
    obj_cat : `CatalogFactory`
        Moving object catalog factory.
    **kwargs :
        Additional keyword arguments are applied as config
        overrides.
    """

    default_config = SimulatedImageConfig
    """Default config."""

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
        if not config["add_bad_columns"]:
            return image

        shape = image.shape
        rng = np.random.RandomState(seed=config["bad_cols_seed"])
        if config["bad_cols_method"] == "random":
            bad_cols = rng.randint(0, shape[1], size=config["n_bad_cols"])
        elif config["bad_col_locs"]:
            bad_cols = config["bad_col_locs"]
        else:
            raise ConfigurationError(
                "Bad columns method is not 'random', but `bad_col_locs` contains no column indices."
            )

        col_pattern = rng.randint(low=0, high=int(config["bad_col_pattern_offset"]), size=shape[0])

        for col in bad_cols:
            image[:, col] += col_pattern + config["bad_col_offset"]

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
        if not config["add_hot_pixels"]:
            return image

        shape = image.shape
        if config["hot_pix_method"] == "random":
            rng = np.random.RandomState(seed=config["hot_pix_seed"])
            x = rng.randint(0, shape[1], size=config["n_hot_pix"])
            y = rng.randint(0, shape[0], size=config["n_hot_pix"])
            hot_pixels = np.column_stack([x, y])
        elif config["hot_pix_locs"]:
            hot_pixels = config["hot_pix_locs"]
        else:
            raise ConfigurationError(
                "Hot pixels method is not 'random', but `hot_pix_locs` contains "
                "no (col, row) location indices of hot pixels."
            )

        for pix in hot_pixels:
            image[pix] += config["hot_pix_offset"]

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
        images += config["read_noise_gen"](scale=config["read_noise"] / config["gain"], size=shape)

        # add dark current
        current = config["dark_current"] * config["exposure_time"] / config["gain"]
        images += config["dark_current_gen"](current, size=shape)

        # add sky counts
        images += (
            config["sky_count_gen"](lam=config["sky_level"] * config["gain"], size=shape) / config["gain"]
        )

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
        base = np.zeros(config["shape"], dtype=config["dtype"])
        base += config["bias"]
        base = cls.add_hot_pixels(base, config)
        base = cls.add_bad_cols(base, config)
        if src_cat is not None:
            add_model_objects(base, src_cat.table, config["model"])

        return base

    def __init__(self, image=None, src_cat=None, obj_cat=None, **kwargs):
        conf = self.default_config(**kwargs)
        super().__init__(image=self.gen_base_image(conf), src_cat=src_cat, obj_cat=obj_cat, **conf)
