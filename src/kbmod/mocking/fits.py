import types
import functools

import numpy as np
from astropy.wcs import WCS
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, BinTableHDU, Header

from .headers import HeaderFactory, ArchivedHeader
from .catalogs import gen_catalog, SimpleSourceCatalog, SimpleObjectCatalog
from .fits_data import ZeroedData, SimpleImage, SimpleVariance, SimpleMask, add_model_objects


__all__ = [
    "HDUFactory",
    "HDUListFactory",
    "callback",
    "EmptyFits",
    "SimpleFits",
    "DECamImdiffs",
]


class HDUFactory:
    def __init__(self, hdu_cls, header_factory, data_factory=None):
        self.hdu_cls = hdu_cls
        self.header_factory = header_factory
        self.data_factory = data_factory
        self.update_data = False if data_factory is None else True

    def mock(self, **kwargs):
        hdu = self.hdu_cls()

        header = self.header_factory.mock(hdu=hdu, **kwargs)
        hdu.header.update(header)

        if self.update_data:
            data = self.data_factory.mock(hdu=hdu, **kwargs)
            hdu.data = data
            # not sure if we want this tbh
            # hdu.update_header()

        return hdu


class HDUListFactory:
    def __init__(self, layout, base_primary={}, base_ext={}, base_wcs={}):
        self.layout = layout
        self.base_primary = base_primary
        self.base_ext = base_ext
        self.base_wcs = base_wcs

    def mock(self, **kwargs):
        hdul = HDUList()
        for layer in self.layout:
            hdul.append(layer.mock(**kwargs))
        return hdul


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


class EmptyFits(HDUListFactory):
    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, primary_hdr=None, dt=0.001):
        # header and data factories that go into creating HDUs
        primary = HeaderFactory.from_base_primary(
            metadata=primary_hdr,
            mutables=["OBS-MJD"],
            callbacks=[self.increment_obstime(dt=dt)],
        )
        image = HeaderFactory.from_base_ext({"EXTNAME": "IMAGE"})
        variance = HeaderFactory.from_base_ext({"EXTNAME": "VARIANCE"})
        mask = HeaderFactory.from_base_ext({"EXTNAME": "MASK"})
        data = ZeroedData()

        # a map of HDU class and their respective header and data generators
        layout = [
            HDUFactory(PrimaryHDU, primary),
            HDUFactory(CompImageHDU, image, data),
            HDUFactory(CompImageHDU, variance, data),
            HDUFactory(CompImageHDU, mask, data),
        ]

        super().__init__(layout)


class SimpleFits(HDUListFactory):
    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, shape=(2048, 4096), noise=100, dt=0.001, source_catalog=None, object_catalog=None):
        # internal counter of n images created so far
        self._idx = 0
        self.dt = dt
        self.src_cat = source_catalog
        self.obj_cat = object_catalog
        dims = {"NAXIS1": shape[0], "NAXIS2": shape[1]}

        # Then we can generate the header factories primary header contains no
        # data, but does contain update-able metadata fields (f.e. timestamps),
        # others contain static headers, but dynamic, mutually connected data
        primary = HeaderFactory.from_base_primary(
            mutables=["OBS-MJD"],
            callbacks=[self.increment_obstime(dt=0.001)],
        )
        image = HeaderFactory.from_base_ext({"EXTNAME": "IMAGE", **dims})
        variance = HeaderFactory.from_base_ext({"EXTNAME": "VARIANCE", **dims})
        mask = HeaderFactory.from_base_ext({"EXTNAME": "MASK", **dims})

        img_data = SimpleImage.simulate(
            shape=shape,
            noise=noise,
            catalog=self.src_cat,
            return_copy=False if self.obj_cat is None else True,
        )
        var_data = SimpleVariance(img_data.base_data, read_noise=noise, gain=1.0)
        mask_data = ZeroedData(np.zeros(shape))

        # Now we can build the HDU map and the HDUList layout
        layout = [
            HDUFactory(PrimaryHDU, primary),
            HDUFactory(CompImageHDU, image, img_data),
            HDUFactory(CompImageHDU, variance, var_data),
            HDUFactory(CompImageHDU, mask, mask_data),
        ]

        super().__init__(layout)

    @classmethod
    def from_defaults(cls, add_static_sources=False, add_moving_objects=True):
        source_catalog, object_catalog = None, None
        if add_static_sources:
            source_catalog = SimpleSourceCatalog.from_params()
        if add_moving_objects:
            object_catalog = SimpleObjectCatalog.from_params()

        return cls(source_catalog=source_catalog, object_catalog=object_catalog)

    def mock(self, **kwargs):
        if self.obj_cat is None:
            new_cat = None
        else:
            new_cat = self.obj_cat.gen_realization(dt=self.dt)

        return super().mock(catalog=new_cat)


class DECamImdiffs(HDUListFactory):
    def __init__(self):
        headers = ArchivedHeader("headers_archive.tar.bz2", "decam_imdiff_headers.ecsv")
        data = ZeroedData()
        image = HDUFactory(CompImageHDU, headers, data)

        # DECam imdiff data products consist of 16 headers, first 4 are:
        # primary, science variance and mask and the rest are PSF, SkyWCS,
        # catalog metadata, chebyshev higher order corrections etc. We don't
        # use these, so leave them with no data generators. The first 4 we fill
        # with all-zeros.
        # don't let Black have its way with these lines because it's a massacre
        # fmt: off
        layout = [
            HDUFactory(PrimaryHDU, headers),
            image,
            image,
            image,
        ]
        layout.extend([HDUFactory(BinTableHDU, headers) ] * 12)
        # fmt: on

        super().__init__(layout)
