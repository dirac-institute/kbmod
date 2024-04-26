import abc
import types
import functools

import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy.modeling import models
from astropy.io.fits import (
    HDUList,
    PrimaryHDU,
    CompImageHDU,
    BinTableHDU,
    Header
)

from .config import Config
from .headers import HeaderFactory, ArchivedHeader, HeaderFactoryConfig
from .catalogs import gen_catalog, SimpleCatalog, SourceCatalog, ObjectCatalog
from .fits_data import (
    DataFactoryConfig,
    DataFactory,
    SimpleImageConfig,
    SimpleImage,
    SimulatedImageConfig,
    SimulatedImage,
    SimpleVarianceConfig,
    SimpleVariance,
    SimpleMaskConfig,
    SimpleMask,
    add_model_objects
)


__all__ = [
    "callback",
    "EmptyFitsConfig",
    "EmptyFits",
    "SimpleFitsConfig",
    "SimpleFits",
    "DECamImdiffConfig",
    "DECamImdiff",
]


class HDUListFactoryConfig(Config):
    validate_header = False
    """Call ``Header.update`` instead of assigning header values. This enforces
    FITS standards and may strip non-standard header keywords."""

    update_header = False
    """After mocking and assigning data call ``Header.update_header``. This
    enforces header to be consistent to the data type and shape. May alter or
    remove keywords from the mocked header."""


class HDUListFactory(abc.ABC):
    default_config = HDUListFactoryConfig

    def __init__(self, config=None, **kwargs):
        self.config = self.default_config(config=config, **kwargs)
        self.current = 0

    def hdu_cast(self, hdu_cls, hdr, data=None, config=None, **kwargs):
        hdu = hdu_cls()

        if self.config.validate_header:
            hdu.header.update(hdr)
        else:
            hdu.header = hdr

        if data is not None:
            hdu.data = data
            if self.config.update_header:
                hdu.update_header()

        return hdu

    @abc.abstractmethod
    def mock(self, n=1):
        raise NotImplementedError()


# I am sure a decorator like this must exist somewhere in functools, but can't
# find it and I'm doing something wrong with functools.partial because that's
# strictly right-side binding?
# multiple options on how to handle callbacks as class members
# callbacks=[lambda old: old+0.01, ]
# callbacks=[functools.partial(self.increment_obstime, dt=0.001), ]
# but they look a bit ugly...
def callback(func):
    if isinstance(func, types.FunctionType):
        # bound methods
        def wrapper(*args, **kwargs):
            @functools.wraps(func)
            def f(*fargs, **fkwargs):
                kwargs.update(fkwargs)
                return func(*(args + fargs), **kwargs)
            return f
    else:
        # functions, static methods
        def wrapper(*args, **kwargs):
            @functools.wraps(func)
            def f(*fargs, **fkwargs):
                kwargs.update(fkwargs)
                return func(*fargs, **kwargs)
            return f
    return wrapper


class EmptyFitsConfig(HDUListFactoryConfig):
    editable_images = False
    separate_masks = False
    writeable_mask = False
    dt = 0.001

    shape = (100, 100)
    """Dimensions of the generated images."""


class EmptyFits(HDUListFactory):
    default_config = EmptyFitsConfig

    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, metadata=None, config=None, **kwargs):
        super().__init__(config=config, method="extend", **kwargs)

        # 1. Update all the default configs. Use class configs that simplify
        #    and unite the many smaller config settings of each underlying factory.
        hdr_conf = HeaderFactoryConfig(config=self.config, method="subset", shape=self.config.shape)

        # 2. Set up Header and Data factories that go into creating HDUs
        #    2.1) First headers, since that metadata specified data formats
        self.primary_hdr = HeaderFactory.from_primary_template(
            overrides=metadata,
            mutables=["OBS-MJD"],
            callbacks=[self.increment_obstime(dt=self.config.dt)],
            config=hdr_conf
        )

        self.img_hdr = HeaderFactory.from_ext_template({"EXTNAME": "IMAGE"}, config=hdr_conf)
        self.var_hdr = HeaderFactory.from_ext_template({"EXTNAME": "VARIANCE"}, config=hdr_conf)
        self.mask_hdr = HeaderFactory.from_ext_template({"EXTNAME": "MASK"}, config=hdr_conf)

        #   2.2) Then data factories, attempt to save performance and memory
        #        where possible by really only allocating 1 array whenever the data
        #        is read-only and content-static between created HDUs.
        writeable, return_copy = False, False
        if self.config.editable_images:
            writeable, return_copy = True, True

        self.data = DataFactory.from_header(
            kind="image",
            header=self.img_hdr.header,
            writeable=writeable,
            return_copy=return_copy
        )
        self.shared_data = DataFactory.from_header("image", self.mask_hdr.header,
                                                   writeable=self.config.writeable_mask)

    def mock(self, n=1):
        # 3) Finally when mocking, vectorize as many operations as possible.
        #    The amount of data scales fast with number of images generated so
        #    that even modest iterative generation of data costs significantly.
        #    (F.e. for each HDU - 3 images are allocated, if we use DECam, for
        #    each HDU, 62 to 70 images can be generated). Many times generated values
        #    are not trivial zeros, but randomly drawn, modified, and then
        #    calculated.
        var_hdr = self.var_hdr.mock()
        mask_hdr = self.mask_hdr.mock()

        imgs = self.data.mock(n)
        variances = self.data.mock(n)
        masks = self.shared_data.mock(n)

        hduls = []
        for i in range(n):
            hduls.append(HDUList(hdus=[
                self.hdu_cast(PrimaryHDU, self.primary_hdr.mock()),
                self.hdu_cast(CompImageHDU, self.img_hdr.mock(), imgs[i]),
                self.hdu_cast(CompImageHDU, var_hdr, variances[i]),
                self.hdu_cast(CompImageHDU, mask_hdr, masks[i])
            ]))

        self.current += n
        return hduls


class SimpleFitsConfig(HDUListFactoryConfig):
    editable_images = False
    separate_masks = False
    writeable_mask = False
    noise_generation = "simplistic"
    dt = 0.001

    shape = (100, 100)
    """Dimensions of the generated images."""


class SimpleFits(HDUListFactory):
    default_config = SimpleFitsConfig

    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, metadata=None, source_catalog=None, object_catalog=None,
                 config=None,  **kwargs):
        super().__init__(config=config, method="extend", **kwargs)

        self.src_cat = source_catalog
        self.obj_cat = object_catalog

        # 1. Update all the default configs using more user-friendly kwargs
        hdr_conf = HeaderFactoryConfig(config=self.config, shape=self.config.shape, method="subset")

        if self.config.noise_generation == "realistic":
            img_cfg = SimulatedImageConfig(config=self.config, method="subset")
        else:
            img_cfg = SimpleImageConfig(config=self.config, method="subset")
        var_cfg = SimpleVarianceConfig(config=self.config, method="subset")
        mask_cfg = SimpleMaskConfig(config=self.config, method="subset")

        # 2. Set up Header and Data factories that go into creating HDUs
        #    2.1) First headers, since that metadata specified data formats
        self.primary_hdr = HeaderFactory.from_primary_template(
            overrides=metadata,
            mutables=["OBS-MJD"],
            callbacks=[self.increment_obstime(dt=self.config.dt)],
            config=hdr_conf
        )
        self.img_hdr = HeaderFactory.from_ext_template({"EXTNAME": "IMAGE"}, config=hdr_conf)
        self.var_hdr = HeaderFactory.from_ext_template({"EXTNAME": "VARIANCE"}, config=hdr_conf)
        self.mask_hdr = HeaderFactory.from_ext_template({"EXTNAME": "MASK"}, config=hdr_conf)

        #   2.2) Then data factories
        if self.config.noise_generation == "realistic":
            self.img_data = SimulatedImage(src_cat=self.src_cat, config=img_cfg)
        else:
            self.img_data = SimpleImage(src_cat=self.src_cat, config=img_cfg)
        self.var_data = SimpleVariance(self.img_data.base, config=var_cfg)
        self.mask_data = SimpleMask.from_image(self.img_data.base, config=mask_cfg)

    @classmethod
    def from_defaults(cls, config=None, sources=None, objects=None, **kwargs):
        config = cls.default_config(config, **kwargs, method="extend")
        cat_lims = {"x_mean": [0, config.shape[1]], "y_mean": [0, config.shape[0]]}
        src_cat, obj_cat = None, None
        if sources is not None:
            if isinstance(sources, SimpleCatalog):
                src_cat = sources
            elif isinstance(sources, Table):
                src_cat = SourceCatalog(table=sources)
            elif isinstance(sources, Config) or sources == True:
                sources.param_ranges.update(cat_lims)
                src_cat = SourceCatalog(**sources)
            elif isinstance(sources, dict):
                sources.update(cat_lims)
                src_cat = SourceCatalog(**sources)
            elif isinstance(sources, int):
                src_cat = SourceCatalog(n=sources, **cat_lims)
            elif sources is True:
                obj_cat = SourceCatalog(**cat_lims)
            else:
                raise ValueError("Sources are expected to be a catalog, table, config or config overrides.")
        if objects is not None:
            if isinstance(objects, SimpleCatalog):
                obj_cat = objects
            elif isinstance(objects, Table):
                obj_cat = ObjectCatalog(table=objects)
            elif isinstance(objects, Config):
                objects.param_ranges.update(cat_lims)
                obj_cat = ObjectCatalog(**objects)
            elif isinstance(objects, dict):
                objects.update(cat_lims)
                obj_cat = ObjectCatalog(**objects)
            elif isinstance(objects, int):
                obj_cat = ObjectCatalog(n=objects, **cat_lims)
            elif objects is True:
                obj_cat = ObjectCatalog(**cat_lims)
            else:
                raise ValueError("Objects are expected to be a catalog, table, config or config overrides.")
        return cls(source_catalog=src_cat, object_catalog=obj_cat, config=config, **kwargs)

    def mock(self, n=1):
        obj_cats = None
        if self.obj_cat is not None:
            obj_cats = self.obj_cat.mock(n, dt=self.config.dt)

        var_hdr = self.var_hdr.mock()
        mask_hdr = self.mask_hdr.mock()

        images = self.img_data.mock(n, obj_cats=obj_cats)
        variances = self.var_data.mock(images=images)
        masks = self.mask_data.mock(n)

        hduls = []
        for i in range(n):
            hduls.append(HDUList(hdus=[
                self.hdu_cast(PrimaryHDU, self.primary_hdr.mock()),
                self.hdu_cast(CompImageHDU, self.img_hdr.mock(), images[i]),
                self.hdu_cast(CompImageHDU, var_hdr, variances[i]),
                self.hdu_cast(CompImageHDU, mask_hdr, masks[i])
            ]))

        self.current += n
        return hduls


class DECamImdiffConfig2(HDUListFactoryConfig):
    archive_name = "headers_archive.tar.bz2"
    file_name = "decam_imdiff_headers.ecsv"
    n_hdrs_per_hdu = 16


class DECamImdiff2(HDUListFactory):
    default_config = DECamImdiffConfig2

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.hdr_factory = ArchivedHeader(self.config.archive_name, self.config.file_name)

    def mock(self, n=1):
        hdrs = self.hdr_factory.mock(n)

        hduls = []
        for hdul_idx in range(n):
            hduls.append(HDUList(hdus=[
                self.hdu_cast(PrimaryHDU,   hdrs[hdul_idx][0]),
                self.hdu_cast(CompImageHDU, hdrs[hdul_idx][1]),
                self.hdu_cast(CompImageHDU, hdrs[hdul_idx][2]),
                self.hdu_cast(CompImageHDU, hdrs[hdul_idx][3]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][4]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][5]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][6]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][7]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][8]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][9]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][10]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][11]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][12]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][13]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][14]),
                self.hdu_cast(BinTableHDU, hdrs[hdul_idx][15]),
            ]))

        self.current += 1
        return hduls




class DECamImdiffConfig(HDUListFactoryConfig):
    archive_name = "headers_archive.tar.bz2"
    file_name = "decam_imdiff_headers.ecsv"

    with_data=False

    editable_images = False
    separate_masks = False
    writeable_mask = False
    noise_generation = "simplistic"
    dt = 0.001

    shape = (100, 100)
    """Dimensions of the generated images."""


class NoneFactory:
    "Kinda makes some code later prettier. Kinda"
    def mock(self, n):
        return [None, ]*n


class DECamImdiff(HDUListFactory):
    default_config = DECamImdiffConfig

    def __init__(self, source_catalog=None, object_catalog=None, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        # 1. Get the header factory - this is different than before. In a
        # header preserving factory, it's the headers that dictate the shape
        # and format of data. Since these are also optional for this class, we
        # just create empty placeholders for data and only fill it with
        # factories if we need to.
        self.hdr_factory = ArchivedHeader(self.config.archive_name,
                                          self.config.file_name)

        self.hdu_types = [PrimaryHDU, CompImageHDU, CompImageHDU, CompImageHDU]
        self.hdu_types.extend([BinTableHDU, ]*12)
        self.data = [NoneFactory(), ]*16

        self.src_cat = source_catalog
        self.obj_cat = object_catalog

        if self.config.with_data:
            self.__init_data_factories()

    def __init_data_factories(self):
        # 2. To fill in data placehodlers, we get an example set of headers in
        # a file. Since they dictate data, we must derive whatever configs we
        # need to and use them as overrides to trump defaults and user-given
        # values. This is the point at which we rely on user telling this
        # factory what the important header keys are since these could be
        # non-standard. And since these change for each HDU, it is rather
        # difficult to generalize without resorting to iterative and fully
        # dynamical data generation - which will be slow. The only mercy is
        # that we only have to do it for the HDUs we care about and use factory
        # methods for the rest.
        headers = self.hdr_factory.get(0)

        img_shape = (headers[1]["NAXIS1"], headers[1]["NAXIS2"])
        img_overrides = {
            "shape": img_shape,
            "dtype": DataFactoryConfig.bitpix_type_map[headers[1]["BITPIX"]],
        }

        if self.config.noise_generation == "realistic":
            img_cfg = SimulatedImageConfig(config=self.config, **img_overrides,  method="subset")
        else:
            img_cfg = SimpleImageConfig(config=self.config,  **img_overrides, method="subset")
        var_cfg = SimpleVarianceConfig(config=self.config, **img_overrides, method="subset")
        mask_cfg = SimpleMaskConfig(config=self.config, **img_overrides, method="subset")

        #   2.1) Now we can instantiate data factories with correct configs
        #        and fill in the data placeholder
        if self.config.noise_generation == "realistic":
            self.img_data = SimulatedImage(src_cat=self.src_cat, config=img_cfg)
        else:
            self.img_data = SimpleImage(src_cat=self.src_cat, config=img_cfg)
            self.var_data = SimpleVariance(self.img_data.base, config=var_cfg)
            self.mask_data = SimpleMask.from_image(self.img_data.base, config=mask_cfg)

        self.hdu_types = [PrimaryHDU, CompImageHDU, CompImageHDU, CompImageHDU]
        self.hdu_types.extend([BinTableHDU, ]*12)

        self.data = [
            NoneFactory(),
            self.img_data,
            self.var_data,
            self.mask_data
        ]
        self.data.extend([
            DataFactory.from_header("table", h) for h in headers[4:]
        ])

    def mock(self, n=1):
        obj_cats = None
        if self.obj_cat is not None:
            obj_cats = self.obj_cat.mock(n, dt=self.config.dt)

        hdrs = self.hdr_factory.mock(n)

        if self.config.with_data:
            images = self.img_data.mock(n, obj_cats=obj_cats)
            variances = self.var_data.mock(images=images)
            data = [self.data[0].mock(n), images, variances]
            for factory in self.data[3:]:
                data.append(factory.mock(n=n))
        else:
            data = [f.mock(n=n) for f in self.data]

        hduls = []
        for hdul_idx in range(n):
            hdus = []
            for hdu_idx, hdu_cls in enumerate(self.hdu_types):
                hdus.append(self.hdu_cast(
                    hdu_cls, hdrs[hdul_idx][hdu_idx], data[hdu_idx][hdul_idx]
                ))
            hduls.append(HDUList(hdus=hdus))
        self.current += n
        return hduls
