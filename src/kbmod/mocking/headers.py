import random
import warnings
import itertools

import numpy as np

from astropy.wcs import WCS
from astropy.io.fits import Header
from astropy.io.fits.verify import VerifyWarning

from .utils import header_archive_to_table
from .config import Config


__all__ = [
#    "make_wcs",
    "WCSFactory",
    "HeaderFactory",
    "ArchivedHeader",
]

class WCSFactory:
    def __init__(self, mode="static",
                 pointing=(351., -5), rotation=0, pixscale=0.2,
                 dither_pos=False, dither_rot=False, dither_amplitudes=(0.01, 0.01, 0.0),
                 cycle=None):
        self.pointing = pointing
        self.rotation = rotation
        self.pixscale = pixscale

        self.dither_pos = dither_pos
        self.dither_rot = dither_rot
        self.dither_amplitudes = dither_amplitudes
        self.cycle = cycle

        self.template = self.gen_wcs(self.pointing, self.rotation, self.pixscale)

        self.mode = mode
        if dither_pos or dither_rot or cycle is not None:
            self.mode = "dynamic"
        self.current = 0

    @classmethod
    def gen_wcs(cls, center_coords, rotation, pixscale, shape=None):
        """
        Create a simple celestial `~astropy.wcs.WCS` object in ICRS
        coordinate system.

        Parameters
        ----------
        shape : tuple[int]
            Two-tuple, dimensions of the WCS footprint
        center_coords : tuple[int]
            Two-tuple of on-sky coordinates of the center of the WCS in
            decimal degrees, in ICRS.
        rotation : float, optional
            Rotation in degrees, from ICRS equator. In decimal degrees.
        pixscale : float
            Pixel scale in arcsec/pixel.

        Returns
        -------
        wcs : `astropy.wcs.WCS`
        The world coordinate system.

        Examples
        --------
        >>> from kbmod.mocking import make_wcs
        >>> shape = (100, 100)
        >>> wcs = make_wcs(shape)
        >>> wcs = make_wcs(shape, (115, 5), 45, 0.1)
        """
        wcs = WCS(naxis=2)
        rho = rotation*0.0174533 # deg to rad
        scale = pixscale  / 3600.0  # arcsec/pixel to deg/pix

        if shape is not None:
            wcs.pixel_shape = shape
            wcs.wcs.crpix = [shape[1] // 2, shape[0] // 2]
        else:
            wcs.wcs.crpix = [0, 0]
        wcs.wcs.crval = center_coords
        wcs.wcs.cunit = ['deg', 'deg']
        wcs.wcs.pc = np.array([
            [-scale * np.cos(rho), scale * np.sin(rho)],
            [scale * np.sin(rho), scale * np.cos(rho)]
        ])
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        return wcs

    def update_from_header(self, header):
        t = self.template.to_header()
        t.update(header)
        self.template = WCS(t)

    def mock(self, header):
        wcs = self.template

        if self.cycle is not None:
            wcs = self.cycle[self.current % len(self.cycle)]

        if self.dither_pos:
            dra = random.uniform(-self.dither_amplitudes[0], self.dither_amplitudes[0])
            ddec = random.uniform(-self.dither_amplitudes[1], self.dither_amplitudes[1])
            wcs.wcs.crval += [dra, ddec]
        if self.dither_rot:
            ddec = random.uniform(-self.dither_amplitudes[2], self.dither_amplitudes[2])
            rho = self.dither_amplitudes[2]*0.0174533 # deg to rad
            rot_matrix =  np.array([
                [-np.cos(rho), np.sin(rho)],
                [np.sin(rho), np.cos(rho)]
            ])
            new_pc = wcs.wcs.pc @ rot_matrix
            wcs.wcs.pc = new_pc

        self.current += 1
        header.update(wcs.to_header())
        return header


class HeaderFactory:
    primary_template = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "DATE-OBS": "2021-03-19T00:27:21.140552",
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
        "OBS-ELEV": 2200,
        "OBSERVAT": "CTIO",
    }

    ext_template = {"NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096, "CRPIX1": 1024, "CRPIX2": 2048, "BITPIX": 32}

    def __validate_mutables(self):
        # !xor
        if bool(self.mutables) != bool(self.callbacks):
            raise ValueError(
                "When providing a list of mutable cards, you must provide associated callback methods."
            )

        if self.mutables is None:
            return

        if len(self.mutables) != len(self.callbacks):
            raise ValueError(
                "The number of mutable cards does not correspond to the number of given callbacks."
            )

        for k in self.mutables:
            if k not in self.header:
                raise ValueError(
                    f"Mutable key {k} does not exists "
                    "in the header. Please "
                    "provide the required metadata keys."
                )

    def __init__(self, metadata, mutables=None, callbacks=None, has_wcs=False, wcs_factory=None):
        cards = [] if metadata is None else metadata
        self.header = Header(cards=cards)
        self.mutables = mutables
        self.callbacks = callbacks
        self.__validate_mutables()

        self.is_dynamic = mutables is not None

        self.has_wcs = has_wcs
        if has_wcs:
            self.wcs_factory = WCSFactory() if wcs_factory is None else wcs_factory
            self.wcs_factory.update_from_header(self.header)
            self.is_dynamic = self.is_dynamic or self.wcs_factory.mode != "static"

        self.counter = 0

    def mock(self, n=1):
        headers = []
        # This can't be vectorized because callbacks may share global state
        for i in range(n):
            if not self.is_dynamic:
                header = self.header
            else:
                header = self.header.copy()
                if self.mutables is not None:
                    for i, mutable in enumerate(self.mutables):
                        header[mutable] = self.callbacks[i](header[mutable])
            if self.has_wcs:
                header = self.wcs_factory.mock(header)
            headers.append(header)
            self.counter += 1
        return headers

    @classmethod
    def gen_header(cls, base, overrides, wcs=None):
        header = Header(base)
        header.update(overrides)

        if wcs is not None:
            # Sync WCS with header + overwrites
            wcs_header = wcs.to_header()
            wcs_header.update(header)
            # then merge back to mocked header template
            header.update(wcs_header)

        return header

    @classmethod
    def from_primary_template(cls, overrides=None, mutables=None, callbacks=None):
        hdr = cls.gen_header(base=cls.primary_template, overrides=overrides)
        return cls(hdr, mutables, callbacks)

    @classmethod
    def from_ext_template(cls, overrides=None, mutables=None, callbacks=None, shape=None,
                          wcs=None):
        ext_template = cls.ext_template.copy()

        if shape is not None:
            ext_template["NAXIS1"] = shape[0]
            ext_template["NAXIS2"] = shape[1]
            ext_template["CRPIX1"] = shape[0] // 2
            ext_template["CRPIX2"] = shape[1] // 2

        hdr = cls.gen_header(base=ext_template, overrides=overrides)
        return cls(hdr, mutables, callbacks, has_wcs=True, wcs_factory=wcs)


class ArchivedHeader(HeaderFactory):
    # will almost never be anything else. Rather, it would be a miracle if it
    # were something else, since FITS standard shouldn't allow it. Further
    # casting by some packages will always be casting implemented in terms of
    # parsing these types.
    lexical_type_map = {
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
    }
    """Map between type names and types themselves."""

    compression = "bz2"

    format = "ascii.ecsv"

    def __init__(self, archive_name, fname):
        super().__init__({})
        self.table = header_archive_to_table(archive_name, fname, self.compression, self.format)

        # Create HDU groups for easier iteration
        self.table = self.table.group_by("filename")
        self.n_hdus = len(self.table)

    def lexical_cast(self, value, format):
        """Cast str literal of a type to the type itself. Supports just
        the builtin Python types.
        """
        if format in self.lexical_type_map:
            return self.lexical_type_map[format](value)
        return value

    def get_item(self, group_idx, hdr_idx):
        header = Header()
        # this is equivalent to one hdulist worth of headers
        group = self.table.groups[group_idx]
        # this is equivalent to one HDU's header
        subgroup = group.group_by("hdu")
        for k, v, f in subgroup.groups[hdr_idx]["keyword", "value", "format"]:
            header[k] = self.lexical_cast(v, f)
        warnings.resetwarnings()
        return header

    def get(self, group_idx):
        headers = []
        # this is a bit repetitive but it saves recreating
        # groups for one HDUL-equivalent many times
        group = self.table.groups[group_idx]
        subgroup = group.group_by("hdu")
        headers = []
        for subgroup in subgroup.groups:
            header = Header()
            for k, v, f in subgroup["keyword", "value", "format"]:
                # ignore warnings about non-standard keywords
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=VerifyWarning)
                    header[k] = self.lexical_cast(v, f)
            headers.append(header)
        return headers

    def mock(self, n=1):
        res = []
        for _ in range(n):
            res.append(self.get(self.counter))
            self.counter += 1
        return res

