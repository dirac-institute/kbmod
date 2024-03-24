import types
import functools

import numpy as np
from astropy.wcs import WCS
from astropy.modeling import models
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, BinTableHDU, Header

from .headers import StaticHeader, MutableHeader, ArchivedHeader
from .catalogs import gen_catalog, SimpleSourceCatalog, SimpleObjectCatalog
from .fits_data import (
    ZeroedData,
    SimpleImage,
    SimpleVariance,
    SimpleMask,
    add_model_objects
)


__all__ = ["HDUFactory", "HDUListFactory", "callback", "SimpleFits", "DECamImdiffs"]


class HDUFactory:
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
            # hdu.update_header()

        return hdu


class HDUListFactory:
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

    def gen_primary_hdr(self, metadata=None, extend_base=True, add_wcs=False):
        return self.gen_header(self.base_primary, metadata, extend_base, add_wcs)

    def gen_ext_hdr(self, metadata=None, extend_base=True, add_wcs=True):
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
    base_primary = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "OBS-MJD": 58914.0,
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
        "STD": "SimpleFits",
    }

    base_ext = {"NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096, "BITPIX": 32}

    base_wcs = {
        "crpix": [1024.0, 2048.0],
        "crval": [351, -5],
        "ctype": ["RA---TAN", "DEC--TAN"],
        "cunit": ["deg", "deg"],
        "radesys": "ICRS",
        "cd": [[-1.44e-07, 7.32e-05], [7.32e-05, 1.44e-05]],
    }

    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self):
        primary_hdr = self.gen_primary()

        # multiple options on how to handle callbacks as class members
        # callbacks=[lambda old: old+0.01, ]
        # callbacks=[functools.partial(self.increment_obstime, dt=0.001), ]
        primary_header_factory = MutableHeader(
            metadata=primary_hdr,
            mutables=[
                "OBS-MJD",
            ],
            callbacks=[
                self.increment_obstime(dt=0.001),
            ],
        )

        image_hdr = self.gen_ext({"EXTNAME": "IMAGE"}, add_wcs=True)
        variance_hdr = self.gen_ext({"EXTNAME": "VARIANCE"}, add_wcs=True)
        mask_hdr = self.gen_ext({"EXTNAME": "MASK", "BITPIX": 8}, add_wcs=True)

        data = ZeroedData()

        layout = [
            HDUFactory(PrimaryHDU, primary_header_factory),
            HDUFactory(CompImageHDU, StaticHeader(image_hdr), data),
            HDUFactory(CompImageHDU, StaticHeader(variance_hdr), data),
            HDUFactory(CompImageHDU, StaticHeader(mask_hdr), data),
        ]

        super().__init__(layout)


class SimpleFits(HDUListFactory):
    base_primary = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "OBS-MJD": 58914.0,
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
        "STD": "SimpleFits",
    }

    base_ext = {"NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096, "BITPIX": 32}

    base_wcs = {
        "crpix": [1024.0, 2048.0],
        "crval": [351, -5],
        "ctype": ["RA---TAN", "DEC--TAN"],
        "cunit": ["deg", "deg"],
        "radesys": "ICRS",
        "cd": [[-1.44e-07, 7.32e-05], [7.32e-05, 1.44e-05]],
    }

    # multiple options on how to handle callbacks as class members
    # callbacks=[lambda old: old+0.01, ]
    # callbacks=[functools.partial(self.increment_obstime, dt=0.001), ]
    @callback
    @staticmethod
    def increment_obstime(old, dt):
        return old + dt

    def __init__(self, shape=(2048, 4096), noise=100, dt=0.001,
                 source_catalog=None, object_catalog=None, **kwargs):
        # let's just make sure the base headers align with the basic params
        self.base_ext["NAXIS1"] = shape[0]
        self.base_ext["NAXIS2"] = shape[1]
        self.base_wcs["crpix"] = (int(shape[0]/2), int(shape[1]/2))

        # internal counter of n images created so far
        self._idx = 0
        self.dt = dt
        self.src_cat = source_catalog
        self.obj_cat = object_catalog

        # Then we can generate The header and data factories that build our
        # HDUList.  Primary header contains no data, but does contain
        # update-able metadata fields (f.e. timestamps)
        primary_hdr = self.gen_primary_hdr()
        primary_header_factory = MutableHeader(
            metadata=primary_hdr,
            mutables=[
                "OBS-MJD",
            ],
            callbacks=[
                self.increment_obstime(dt=dt),
            ],
        )

        # The remaining HDUs contain static headers, but data that changes
        # (f.e. updating positions of linearly moving objects in each image)
        img_hdr = self.gen_ext_hdr({"EXTNAME": "IMAGE"}, add_wcs=True)
        img_data = SimpleImage.from_simplistic_sim(
            shape=shape,
            noise=noise,
            catalog=self.src_cat,
            return_copy=False if self.obj_cat is None else True
        )

        var_hdr = self.gen_ext_hdr({"EXTNAME": "VARIANCE"}, add_wcs=True)
        var_data = SimpleVariance(
            img_data.base_data,
            read_noise=noise,
            gain=1.0
        )

        mask_hdr = self.gen_ext_hdr({"EXTNAME": "MASK", "BITPIX": 8}, add_wcs=True)
        mask_data = ZeroedData(np.zeros(shape))

        # Now we can build an HDU layout and let the HDUList factory
        # generate self-updateable HDULists.
        layout = [
            HDUFactory(PrimaryHDU, primary_header_factory),
            HDUFactory(CompImageHDU, StaticHeader(img_hdr), img_data),
            HDUFactory(CompImageHDU, StaticHeader(var_hdr), var_data),
            HDUFactory(CompImageHDU, StaticHeader(mask_hdr), mask_data),
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
