import warnings

import numpy as np

from astropy.wcs import WCS
from astropy.io.fits import Header

from .utils import header_archive_to_table
from .config import Config


__all__ = [
    "HeaderFactory",
    "ArchivedHeader",
]


def make_wcs(center_coords=(351., -5.), rotation=0, pixscale=0.2, shape=None):
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
    scale = 0.1 / 3600.0  # arcsec/pixel to deg/pix

    if shape is not None:
        wcs.pixel_shape = shape
        wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    else:
        wcs.wcs.crpix = [0, 0]
    wcs.wcs.crval = center_coords
    wcs.wcs.cunit = ['deg', 'deg']
    wcs.wcs.cd = [[-scale * np.cos(rho), scale * np.sin(rho)],
                  [scale * np.sin(rho), scale * np.cos(rho)]]
    wcs.wcs.radesys = 'ICRS'
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    return wcs


class HeaderFactory:
    primary_template = {
        "EXTNAME": "PRIMARY",
        "NAXIS": 0,
        "BITPIX": 8,
        "OBS-MJD": 58914.0,
        "NEXTEND": 3,
        "OBS-LAT": -30.166,
        "OBS-LONG": -70.814,
        "OBS-ELEV": 2200,
        "OBSERVAT": "CTIO",
    }

    ext_template = {"NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096, "CRPIX1": 1024, "CPRIX2": 2048, "BITPIX": 32}

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

    def __init__(self, metadata, mutables=None, callbacks=None, config=None, **kwargs):
        cards = [] if metadata is None else metadata
        self.header = Header(cards=cards)

        self.mutables = mutables
        self.callbacks = callbacks
        self.__validate_mutables()

        self.is_dynamic = self.mutables is not None
        self.counter = 0

    def mock(self, n=1):
        headers = []
        # This can't be vectorized because callbacks may share global state
        for i in range(n):
            if self.is_dynamic:
                header = self.header.copy()
                for i, mutable in enumerate(self.mutables):
                    header[mutable] = self.callbacks[i](header[mutable])
            else:
                header = self.header
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
    def from_ext_template(cls, overrides=None, mutables=None, callbacks=None, wcs=None, shape=None):
        ext_template = cls.ext_template.copy()

        if shape is not None:
            ext_template["NAXIS1"] = shape[0]
            ext_template["NAXIS2"] = shape[1]
            ext_template["CRPIX1"] = shape[0] // 2
            ext_template["CRPIX2"] = shape[1] // 2

        if wcs is None:
            wcs = make_wcs(
                shape=(ext_template["NAXIS1"], ext_template["NAXIS2"]),

            )

        hdr = cls.gen_header(base=ext_template, overrides=overrides, wcs=wcs)
        return cls(hdr, mutables, callbacks)


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

    def __init__(self, archive_name, fname, config=None, **kwargs):
        super().__init__(config, **kwargs)
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
                header[k] = self.lexical_cast(v, f)
            headers.append(header)
        return headers

    def mock(self, n=1):
        res = []
        for _ in range(n):
            res.append(self.get(self.counter))
            self.counter += 1
        return res
