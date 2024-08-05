from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, BinTableHDU

from .callbacks import IncrementObstime
from .headers import HeaderFactory, ArchivedHeader
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
)


__all__ = [
    "EmptyFits",
    "SimpleFits",
    "DECamImdiff",
]


class NoneFactory:
    "Kinda makes some code later prettier. Kinda"

    def mock(self, n):
        return [
            None,
        ] * n


def hdu_cast(hdu_cls, hdr, data=None, validate_header=False, update_header=False):
    hdu = hdu_cls()

    if validate_header:
        hdu.header.update(hdr)
    else:
        hdu.header = hdr

    if data is not None:
        hdu.data = data
        if update_header:
            hdu.update_header()

    return hdu


def hdu_cast_array(hdu_cls, hdr, data, validate_header=False, update_header=False):
    hdus = []
    for hdr, dat in zip(hdr, data):
        hdus.append(hdu_cast(hdu_cls, hdr, dat))
    return hdus


class EmptyFits:
    def __init__(
        self,
        header=None,
        shape=(100, 100),
        start_mjd=60310,
        step_mjd=0.001,
        editable_images=False,
        editable_masks=False,
    ):
        self.prim_hdr = HeaderFactory.from_primary_template(
            overrides=header, mutables=["OBS-MJD"], callbacks=[IncrementObstime(start=start_mjd, dt=step_mjd)]
        )

        self.img_hdr = HeaderFactory.from_ext_template({"EXTNAME": "IMAGE"}, shape=shape)
        self.var_hdr = HeaderFactory.from_ext_template({"EXTNAME": "VARIANCE"}, shape=shape)
        self.mask_hdr = HeaderFactory.from_ext_template({"EXTNAME": "MASK"}, shape=shape)

        #   2.2) Then data factories, attempt to save performance and memory
        #        where possible by really only allocating 1 array whenever the
        #        data is read-only and content-static between created HDUs.
        self.img_data = DataFactory.from_header(
            kind="image", header=self.img_hdr.header, writeable=editable_images, return_copy=editable_images
        )
        self.mask_data = DataFactory.from_header(
            kind="image", header=self.mask_hdr.header, return_copy=editable_masks, writeable=editable_masks
        )

        self.current = 0

    def mock(self, n=1):
        img_hdr = self.img_hdr.mock()[0]
        var_hdr = self.var_hdr.mock()[0]
        mask_hdr = self.mask_hdr.mock()[0]
        images = self.img_data.mock(n=n)
        variances = self.img_data.mock(n=n)
        masks = self.mask_data.mock(n=n)

        hduls = []
        for i in range(n):
            hduls.append(
                HDUList(
                    hdus=[
                        PrimaryHDU(header=self.prim_hdr.mock()[0]),
                        CompImageHDU(header=img_hdr, data=images[i]),
                        CompImageHDU(header=var_hdr, data=variances[i]),
                        CompImageHDU(header=mask_hdr, data=masks[i]),
                    ]
                )
            )

        self.current += n
        return hduls


class SimpleFits:
    def __init__(
        self,
        header=None,
        shape=(100, 100),
        start_mjd=60310,
        step_mjd=0.001,
        with_noise=False,
        noise="simplistic",
        src_cat=None,
        obj_cat=None,
    ):
        # 2. Set up Header and Data factories that go into creating HDUs
        #    2.1) First headers, since that metadata specified data formats
        self.prim_hdr = HeaderFactory.from_primary_template(
            overrides=header, mutables=["OBS-MJD"], callbacks=[IncrementObstime(start=start_mjd, dt=step_mjd)]
        )

        self.img_hdr = HeaderFactory.from_ext_template({"EXTNAME": "IMAGE"}, shape=shape)
        self.var_hdr = HeaderFactory.from_ext_template({"EXTNAME": "VARIANCE"}, shape=shape)
        self.mask_hdr = HeaderFactory.from_ext_template({"EXTNAME": "MASK"}, shape=shape)

        #   2.2) Then data factories
        if noise == "realistic":
            self.img_data = SimulatedImage(shape=shape, src_cat=src_cat, add_noise=with_noise)
        else:
            self.img_data = SimpleImage(shape=shape, src_cat=src_cat, add_noise=with_noise)
        self.var_data = SimpleVariance(self.img_data.base)
        self.mask_data = SimpleMask.from_image(self.img_data.base)

        self.start_mjd = start_mjd
        self.step_mjd = step_mjd
        self.obj_cat = obj_cat
        self.current = 0

    def mock(self, n=1):
        prim_hdrs = self.prim_hdr.mock(n=n)
        img_hdrs = self.img_hdr.mock(n=n)
        var_hdrs = self.var_hdr.mock(n=n)
        mask_hdrs = self.mask_hdr.mock(n=n)

        obj_cats = None
        if self.obj_cat is not None:
            kwargs = {"dt": self.step_mjd, "t": [hdr["OBS-MJD"] for hdr in prim_hdrs]}
            obj_cats = self.obj_cat.mock(n=n, **kwargs)

        images = self.img_data.mock(n, obj_cats=obj_cats)
        variances = self.var_data.mock(images=images)
        masks = self.mask_data.mock(n)

        hduls = []
        for ph, ih, vh, mh, imd, vd, md in zip(
            prim_hdrs, img_hdrs, var_hdrs, mask_hdrs, images, variances, masks
        ):
            hduls.append(
                HDUList(
                    hdus=[
                        PrimaryHDU(header=ph),
                        CompImageHDU(header=ih, data=imd),
                        CompImageHDU(header=vh, data=vd),
                        CompImageHDU(header=mh, data=md),
                    ]
                )
            )

        self.current += n
        return hduls


class DECamImdiff:
    @classmethod
    def from_defaults(
        cls,
        with_data=False,
        noise="simplistic",
        src_cat=None,
        obj_cat=None,
        editable_images=False,
        separate_masks=False,
        writeable_masks=False,
        editable_masks=False,
    ):
        if obj_cat.config.type == "progressive":
            raise ValueError(
                "Only folding or static object catalogs can be used with"
                "default archived DECam headers since header timestamps are not "
                "required to be equally spaced."
            )

        hdr_factory = ArchivedHeader("headers_archive.tar.bz2", "decam_imdiff_headers.ecsv")

        hdu_types = [PrimaryHDU, CompImageHDU, CompImageHDU, CompImageHDU]
        hdu_types.extend(
            [
                BinTableHDU,
            ]
            * 12
        )
        data = [
            NoneFactory(),
        ] * 16

        if with_data:
            headers = hdr_factory.get(0)

            shape = (headers[1]["NAXIS1"], headers[1]["NAXIS2"])
            dtype = DataFactoryConfig.bitpix_type_map[headers[1]["BITPIX"]]

            #   2.1) Now we can instantiate data factories with correct configs
            #        and fill in the data placeholder
            if noise == "realistic":
                img_data = SimulatedImage(src_cat=src_cat, shape=shape, dtype=dtype)
            else:
                img_data = SimpleImage(src_cat=src_cat, shape=shape, dtype=dtype)
                var_data = SimpleVariance(img_data.base)
                mask_data = SimpleMask.from_image(img_data.base)

                data = [NoneFactory(), img_data, var_data, mask_data]
                data.extend([DataFactory.from_header("table", h) for h in headers[4:]])

        return cls(hdr_factory, data_factories=data, obj_cat=obj_cat)

    def __init__(self, header_factory, data_factories=None, obj_cat=None):
        self.hdr_factory = header_factory
        self.data_factories = data_factories
        self.hdu_layout = [PrimaryHDU, CompImageHDU, CompImageHDU, CompImageHDU]
        self.hdu_layout.extend(
            [
                BinTableHDU,
            ]
            * 12
        )

    def mock(self, n=1):
        obj_cats = None
        if self.obj_cat is not None:
            obj_cats = self.obj_cat.mock(n, dt=self.config.dt)

        hdrs = self.hdr_factory.mock(n)

        if self.data_factories is not None:
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
                hdus.append(self.hdu_cast(hdu_cls, hdrs[hdul_idx][hdu_idx], data[hdu_idx][hdul_idx]))
            hduls.append(HDUList(hdus=hdus))
        self.current += n
        return hduls
