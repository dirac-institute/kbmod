import abc
import types
import functools

import numpy as np
from astropy.wcs import WCS
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, BinTableHDU, Header
from astropy.modeling import models

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
        # hdu.header.update(header) costs more but maybe better?
        hdu.header = header

        if self.update_data:
            data = self.data_factory.mock(hdu=hdu, **kwargs)
            hdu.data = data
            # not sure if we want this tbh
            # hdu.update_header()

        return hdu


class HDUListFactory(abc.ABC):
    def __init__(self, layout, base_primary={}, base_ext={}, base_wcs={}):
        self.layout = layout
        self.base_primary = base_primary
        self.base_ext = base_ext
        self.base_wcs = base_wcs

    def generate(self, **kwargs):
        hdul = HDUList()
        for layer in self.layout:
            hdul.append(layer.mock(**kwargs))
        return hdul

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


class EmptyFits(HDUListFactory):
    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, primary_hdr=None, dt=0.001):
        # header and data factories that go into creating HDUs
        self.primary = HeaderFactory.from_base_primary(
            metadata=primary_hdr,
            mutables=["OBS-MJD"],
            callbacks=[self.increment_obstime(dt=dt)],
        )
        self.image = HeaderFactory.from_base_ext({"EXTNAME": "IMAGE"})
        self.variance = HeaderFactory.from_base_ext({"EXTNAME": "VARIANCE"})
        self.mask = HeaderFactory.from_base_ext({"EXTNAME": "MASK"})
        data = ZeroedData()

        # a map of HDU class and their respective header and data generators
        layout = [
            HDUFactory(PrimaryHDU, self.primary),
            HDUFactory(CompImageHDU, self.image, data),
            HDUFactory(CompImageHDU, self.variance, data),
            HDUFactory(CompImageHDU, self.mask, data),
        ]

        super().__init__(layout)

    def mock(self, n=1):
        if n == 1:
            return self.step()

        shape = (n*3, self.image.header["NAXIS1"], self.image.header["NAXIS1"])
        images = np.zeros(shape, dtype=np.float32)
        imghdr, varhdr, maskhdr = self.image.mock(), self.variance.mock(), self.mask.mock()
        hduls = []
        for i in range(n):
            hduls.append(HDUList(
                hdus=[
                    PrimaryHDU(header=self.primary.mock()),
                    CompImageHDU(header=imghdr, data=images[i]),
                    CompImageHDU(header=varhdr, data=images[n+i]),
                    CompImageHDU(header=maskhdr, data=images[2*n+i])
                ]
            ))

        return hduls



class SimpleFits(HDUListFactory):
    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, shape, noise=0, noise_std=1, dt=0.001, source_catalog=None, object_catalog=None):
        # internal counter of n images created so far
        self._idx = 0
        self.dt = dt
        self.src_cat = source_catalog
        self.obj_cat = object_catalog
        self.shape = shape
        self.noise = noise
        self.noise_std = noise_std
        dims = {"NAXIS1": shape[0], "NAXIS2": shape[1]}

        # Then we can generate the header factories primary header contains no
        # data, but does contain update-able metadata fields (f.e. timestamps),
        # others contain static headers, but dynamic, mutually connected data
        self.primary = HeaderFactory.from_base_primary(
            mutables=["OBS-MJD"],
            callbacks=[self.increment_obstime(dt=0.001)],
        )
        self.image = HeaderFactory.from_base_ext({"EXTNAME": "IMAGE", **dims})
        self.variance = HeaderFactory.from_base_ext({"EXTNAME": "VARIANCE", **dims})
        self.mask = HeaderFactory.from_base_ext({"EXTNAME": "MASK", **dims})

        self.img_data = SimpleImage.simulate(
            shape=shape,
            noise=noise,
            catalog=self.src_cat,
            return_copy=False if self.obj_cat is None else True,
        )
        self.var_data = SimpleVariance(self.img_data.base_data, read_noise=noise, gain=1.0)
        self.mask_data = ZeroedData(np.zeros(shape))

        # Now we can build the HDU map and the HDUList layout
        layout = [
            HDUFactory(PrimaryHDU, self.primary),
            HDUFactory(CompImageHDU, self.image, self.img_data),
            HDUFactory(CompImageHDU, self.variance, self.var_data),
            HDUFactory(CompImageHDU, self.mask, self.mask_data),
        ]

        super().__init__(layout)

    @classmethod
    def from_defaults(cls, shape=(2048, 4096), add_static_sources=False, add_moving_objects=True, **kwargs):
        source_catalog, object_catalog = None, None
        cat_lims = {"x_mean": [0, shape[1]], "y_mean": [0, shape[0]]}
        if add_static_sources:
            source_catalog = SimpleSourceCatalog.from_params(param_ranges=cat_lims)
        if add_moving_objects:
            object_catalog = SimpleObjectCatalog.from_params(param_ranges=cat_lims)

        return cls(shape=shape, source_catalog=source_catalog, object_catalog=object_catalog, **kwargs)

    def mock(self, n=1, catalog=None):
        if n == 1:
            # a valiant effort, but too slow...
            return self.generate()

        shape = (n*2, *self.shape)
        images = np.zeros(shape, dtype=np.float32)
        mask_data = self.mask_data.mock()

        if self.noise != 0:
            rng = np.random.default_rng()
            rng.standard_normal(size=shape, dtype=np.float32, out=images)
            images *= self.noise_std
            images += self.noise

        # update the base image (static objects, bad columns etc.)
        if self.src_cat is not None:
            images += self.img_data.base_data

        #  update the variance images if needed only
        if self.var_data.gain != 1.0:
            images[n:] /= self.var_data.gain
        if self.var_data.read_noise != 0.0:
            images[n:] += self.var_data.read_noise**2

        # shared headers
        imghdr, varhdr, maskhdr = self.image.mock(), self.variance.mock(), self.mask.mock()

        hduls = []
        for i in range(n):
            if self.obj_cat is not None:
                new_cat = self.obj_cat.gen_realization(dt=self.dt)
                new_img = add_model_objects(np.zeros(images[0].shape), new_cat, models.Gaussian2D(x_stddev=1, y_stddev=1))
                images[i] += new_img
                images[n+1] += new_img

            hduls.append(HDUList(
                hdus=[
                    PrimaryHDU(header=self.primary.mock()),
                    CompImageHDU(header=imghdr, data=images[i]),
                    CompImageHDU(header=varhdr, data=images[n+i]),
                    CompImageHDU(header=maskhdr, data=mask_data),
                ]
            ))

        self._idx += n
        return hduls


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

    def mock(self, n=1):
        if n==1:
            return self.step()
        hduls = [self.step() for i in range(n)]
        return hduls
