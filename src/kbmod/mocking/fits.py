from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, BinTableHDU
from astropy.wcs import WCS

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


class EmptyFits:
    def __init__(
        self,
        header=None,
        shape=(100, 100),
        start_t="2024-01-01T00:00:00.00",
        step_t=0.001,
        editable_images=False,
        editable_masks=False,
    ):
        self.prim_hdr = HeaderFactory.from_primary_template(
            overrides=header,
            mutables=["DATE-OBS"],
            callbacks=[IncrementObstime(start=start_t, dt=step_t)]
        )

        self.img_hdr = HeaderFactory.from_ext_template({"EXTNAME": "IMAGE"}, shape=shape)
        self.var_hdr = HeaderFactory.from_ext_template({"EXTNAME": "VARIANCE"}, shape=shape)
        self.mask_hdr = HeaderFactory.from_ext_template({"EXTNAME": "MASK"}, shape=shape)

        self.img_data = DataFactory.from_header(
            kind="image", header=self.img_hdr.header, writeable=editable_images, return_copy=editable_images
        )
        self.mask_data = DataFactory.from_header(
            kind="image", header=self.mask_hdr.header, return_copy=editable_masks, writeable=editable_masks
        )

        self.current = 0

    def mock(self, n=1):
        prim_hdrs = self.prim_hdr.mock(n=n)
        img_hdrs = self.img_hdr.mock(n=n)
        var_hdrs = self.var_hdr.mock(n=n)
        mask_hdrs = self.mask_hdr.mock(n=n)

        images = self.img_data.mock(n=n)
        variances = self.img_data.mock(n=n)
        masks = self.mask_data.mock(n=n)

        hduls = []
        for ph, ih, vh, mh, imd, vd, md in zip(
            prim_hdrs, img_hdrs, var_hdrs, mask_hdrs, images, variances, masks
        ):
            hduls.append(
                HDUList(hdus=[
                    PrimaryHDU(header=ph),
                    CompImageHDU(header=ih, data=imd),
                    CompImageHDU(header=vh, data=vd),
                    CompImageHDU(header=mh, data=md)
                ])
            )

        self.current += n
        return hduls


class SimpleFits:
    def __init__(
            self,
            shared_header_metadata=None,
            shape=(100, 100),
            start_t="2024-01-01T00:00:00.00",
            step_t=0.001,
            with_noise=False,
            noise="simplistic",
            src_cat=None,
            obj_cat=None,
            wcs_factory=None,
    ):
        # 2. Set up Header and Data factories that go into creating HDUs
        #    2.1) First headers, since that metadata specified data formats
        self.prim_hdr = HeaderFactory.from_primary_template(
            overrides=shared_header_metadata,
            mutables=["DATE-OBS"], callbacks=[IncrementObstime(start=start_t, dt=step_t)]
        )

        wcs = None
        if wcs_factory is not None:
            wcs = wcs_factory

        if shared_header_metadata is None:
            shared_header_metadata = {"EXTNAME": "IMAGE"}

        self.img_hdr = HeaderFactory.from_ext_template(
            overrides=shared_header_metadata.copy(),
            shape=shape,
            wcs=wcs
        )
        shared_header_metadata["EXTNAME"] = "VARIANCE"
        self.var_hdr = HeaderFactory.from_ext_template(
            overrides=shared_header_metadata.copy(),
            shape=shape,
            wcs=wcs
        )
        shared_header_metadata["EXTNAME"] = "MASK"
        self.mask_hdr = HeaderFactory.from_ext_template(
            overrides=shared_header_metadata.copy(),
            shape=shape,
            wcs=wcs
        )

        #   2.2) Then data factories
        if noise == "realistic":
            self.img_data = SimulatedImage(shape=shape, src_cat=src_cat, add_noise=with_noise)
        else:
            self.img_data = SimpleImage(shape=shape, src_cat=src_cat, add_noise=with_noise)
        self.var_data = SimpleVariance(self.img_data.base)
        self.mask_data = SimpleMask.from_image(self.img_data.base)

        self.start_t = start_t
        self.step_t = step_t
        self.obj_cat = obj_cat
        self.current = 0

    def mock(self, n=1):
        prim_hdrs = self.prim_hdr.mock(n=n)
        img_hdrs = self.img_hdr.mock(n=n)
        var_hdrs = self.var_hdr.mock(n=n)
        mask_hdrs = self.mask_hdr.mock(n=n)

        obj_cats = None
        if self.obj_cat is not None:
            obj_cats = self.obj_cat.mock(
                n=n,
                dt=self.step_t,
                t=[hdr["DATE-OBS"] for hdr in prim_hdrs],
                wcs=[WCS(hdr) for hdr in img_hdrs]
            )

        images = self.img_data.mock(n, obj_cats=obj_cats)
        variances = self.var_data.mock(images=images)
        masks = self.mask_data.mock(n)

        hduls = []
        for ph, ih, vh, mh, imd, vd, md in zip(
            prim_hdrs, img_hdrs, var_hdrs, mask_hdrs, images, variances, masks
        ):
            hduls.append(
                HDUList(hdus=[
                    PrimaryHDU(header=ph),
                    CompImageHDU(header=ih, data=imd),
                    CompImageHDU(header=vh, data=vd),
                    CompImageHDU(header=mh, data=md)
                ])
            )

        self.current += n
        return hduls


class DECamImdiff:
    def __init__(self, with_data=False, with_noise=False, noise="simplistic",
                 src_cat=None, obj_cat=None):
        if obj_cat is not None and obj_cat.mode == "progressive":
            raise ValueError(
                "Only folding or static object catalogs can be used with"
                "default archived DECam headers since header timestamps are not "
                "required to be equally spaced."
            )

        self.hdr_factory = ArchivedHeader("headers_archive.tar.bz2", "decam_imdiff_headers.ecsv")

        self.data_factories = [NoneFactory()] * 16
        if with_data:
            headers = self.hdr_factory.get(0)

            shape = (headers[1]["NAXIS1"], headers[1]["NAXIS2"])
            dtype = DataFactoryConfig.bitpix_type_map[headers[1]["BITPIX"]]

            # Read noise and gain are typical values. DECam has 2 amps per CCD,
            # each powering ~half of the plane. Their values and areas are
            # recorded in the header, but that would mean I would have to
            # produce an image which has different zero-offsets for the two
            # halves which is too much detail for this use-case. Typical values
            # are taken from the DECam Data Handbook Version 2.05 March 2014
            # Table 2.2
            if noise == "realistic":
                self.img_data = SimulatedImage(src_cat=src_cat, shape=shape, dtype=dtype)
            else:
                self.img_data = SimpleImage(src_cat=src_cat, shape=shape, dtype=dtype)
                self.var_data = SimpleVariance(self.img_data.base, read_noise=7.0, gain=4.0)
                self.mask_data = SimpleMask.from_image(self.img_data.base)

                self.data_factories[1] = self.img_data
                self.data_factories[2] = self.mask_data
                self.data_factories[3] = self.mask_data
                self.data_factories[4:] = [DataFactory.from_header("table", h) for h in headers[4:]]

        self.with_data = with_data
        self.src_cat = src_cat
        self.obj_cat = obj_cat
        self.hdu_layout = [PrimaryHDU, CompImageHDU, CompImageHDU, CompImageHDU]
        self.hdu_layout.extend([BinTableHDU] * 12)
        self.current = 0

    def mock(self, n=1):
        headers = self.hdr_factory.mock(n=n)

        obj_cats = None
        if self.obj_cat is not None:
            kwargs = {"t": [hdrs[0][0]["DATE-AVG"] for hdr in hdrs]}
            obj_cats = self.obj_cat.mock(n=n, **kwargs)

        if self.with_data:
            images = self.img_data.mock(n=n, obj_cats=obj_cats)
            masks = self.mask_data.mock(n=n)
            variances = self.var_data.mock(images=images)
            data = [
                NoneFactory().mock(n=n),
                images,
                masks,
                variances
            ]
            data.extend([factory.mock(n=n) for factory in self.data_factories[4:]])
        else:
            data = [factory.mock(n=n) for factory in self.data_factories]

        hduls = []
        for i, hdrs in enumerate(headers):
            hdus = []
            for j, (layer, hdr) in enumerate(zip(self.hdu_layout, hdrs)):
                hdus.append(layer(header=hdr, data=data[j][i]))
            hduls.append(HDUList(hdus=hdus))

        self.current += n
        return hduls
