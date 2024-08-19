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
    "WCSFactory",
    "HeaderFactory",
    "ArchivedHeader",
]

class WCSFactory:
    """WCS Factory.

    Used to generate collections of ICRS TAN WCS from parameters, or as a way to
    update the given header with a new WCS.

    The new WCS is generated by inheriting the header and updating its properties,
    or by completely overwriting the Header WCS and replacing it with a new WCS.

    Attribute
    ---------
    current : `int`
        Counter of the number of mocked WCSs.
    template : `WCS`
        Template WCS.

    Parameters
    ----------
    pointing : `tuple`, optional
        Ra and Dec w.r.t ICRS coordinate system, in decimal degrees.
    rotation : `float`, optional
        Rotation, in degrees, from ICRS equator (ecliptic).
    pixscale : `float`, optional
        Pixel scale, in arcseconds per pixel.
    dither_pos : `bool`, optional
        Dither positions of mocked WCSs.
    dither_rot : `bool`, optional
        Dither rotations of mocked WCSs.
    dither_amplitudes : `tuple`, optional
        A set of 3 values, the amplitude of dither in ra direction, the
        amplitude of dither in dec direction and the amplitude of dither in
        rotations. In decimal degrees.
    cycle : `list[WCS]`, optional
        A list of pre-created WCS objects through which to iterate.

    Examples
    --------
    >>> from astropy.io.fits import Header
    >>> import kbmod.mocking as kbmock
    >>> wcsf = kbmock.WCSFactory(pointing=(10, 10), rotation=45)
    >>> wcsf.mock(Header())
    WCSAXES =                    2 / Number of coordinate axes
    CRPIX1  =                  0.0 / Pixel coordinate of reference point
    CRPIX2  =                  0.0 / Pixel coordinate of reference point
    PC1_1   =  -3.928369684292E-05 / Coordinate transformation matrix element
    PC1_2   =  3.9283723288914E-05 / Coordinate transformation matrix element
    PC2_1   =  3.9283723288914E-05 / Coordinate transformation matrix element
    PC2_2   =   3.928369684292E-05 / Coordinate transformation matrix element
    CDELT1  =                  1.0 / [deg] Coordinate increment at reference point
    CDELT2  =                  1.0 / [deg] Coordinate increment at reference point
    CUNIT1  = 'deg'                / Units of coordinate increment and value
    CUNIT2  = 'deg'                / Units of coordinate increment and value
    CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
    CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
    CRVAL1  =                 10.0 / [deg] Coordinate value at reference point
    CRVAL2  =                 10.0 / [deg] Coordinate value at reference point
    LONPOLE =                180.0 / [deg] Native longitude of celestial pole
    LATPOLE =                 10.0 / [deg] Native latitude of celestial pole
    MJDREF  =                  0.0 / [d] MJD of fiducial time
    RADESYS = 'ICRS'               / Equatorial coordinate system,
    """
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
    def gen_wcs(cls, pointing, rotation=0, pixscale=1, shape=None):
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
        >>> import kbmod.mocking as kbmock
        >>> [kbmock.WCSFactory.gen_wcs((10, 10), rot, 0.2) for rot in (0, 90)]
        [WCS Keywords

        Number of WCS axes: 2
        CTYPE : 'RA---TAN' 'DEC--TAN'
        CRVAL : 10.0 10.0
        CRPIX : 0.0 0.0
        PC1_1 PC1_2  : -5.555555555555556e-05 0.0
        PC2_1 PC2_2  : 0.0 5.555555555555556e-05
        CDELT : 1.0 1.0
        NAXIS : 0  0, WCS Keywords

        Number of WCS axes: 2
        CTYPE : 'RA---TAN' 'DEC--TAN'
        CRVAL : 10.0 10.0
        CRPIX : 0.0 0.0
        PC1_1 PC1_2  : 3.7400283533421276e-11 5.555555555554297e-05
        PC2_1 PC2_2  : 5.555555555554297e-05 -3.7400283533421276e-11
        CDELT : 1.0 1.0
        NAXIS : 0  0]
        """
        wcs = WCS(naxis=2)
        rho = rotation*0.0174533 # deg to rad
        scale = pixscale  / 3600.0  # arcsec/pixel to deg/pix

        if shape is not None:
            wcs.pixel_shape = shape
            wcs.wcs.crpix = [shape[1] // 2, shape[0] // 2]
        else:
            wcs.wcs.crpix = [0, 0]
        wcs.wcs.crval = pointing
        wcs.wcs.cunit = ['deg', 'deg']
        wcs.wcs.pc = np.array([
            [-scale * np.cos(rho), scale * np.sin(rho)],
            [scale * np.sin(rho), scale * np.cos(rho)]
        ])
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        return wcs

    def next(self):
        """Iteratively return WCS from the cycle."""
        for wcs in self.cycle:
            yield wcs

    def update_from_header(self, header):
        """Update WCS template using a header, only updates the cards template
        shares with the given header.

        Updates the template WCS in-place.
        """
        t = self.template.to_header()
        t.update(header)
        self.template = WCS(t)

    def update_headers(self, headers):
        """Update the headers, in-place, with a new mocked WCS.

        If the header contains a WCS, it is updated to match the template.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
             Header

        Returns
        -------
        header : `astropy.io.fits.Header`
            Update header.
        """
        wcs = self.template

        for header in headers:
            if self.cycle is not None:
                wcs = self.next()

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


class HeaderFactory:
    """Mocks a header from a given template.

    Callback functions can be defined for individual header cards which are
    executed and their respective values are updated for each new mocked header.

    A WCS factory can be attached to this factory to update the related WCS
    header keywords.

    Provides two base templates from which to create Headers from, based on
    CTIO observatory location and DECam instrument.

    Can generate headers from dict-like template and card overrides.

    Attributes
    ----------
    is_dynamic : `bool`
        True when factory has a mutable Header cards or a WCS factory.

    Parameters
    ---------
    metadata : `dict-like`
        Header, dict, or a list of cards from which a header is created.
    mutables : `list[str]` or `None`, optional
        A list of strings, matching header card keys, designating them as a
        card that has an associated callback.
    callbacks : `list[func]`, `list[class]` or `None`
        List of callbacks, functions or classes with a ``__call__`` method, that
        will be executed in order to update the respective key in ``mutables``
        See already provided callbacks in `kbmod.mocking.callbacks` module.
    has_wcs : `bool`
        Attach a WCS to each produced header.
    wcs_factory : `WCSFactory` or `None`
        A WCS factory to use, if `None` and `has_wcs` is `True`, uses the default
        WCS Factory. See `WCSFactory` for details.

    Examples
    --------
    >>> import kbmod.mocking as kbmock
    >>> hf = kbmock.HeaderFactory.from_primary_template()
    >>> hf.mock()
    [EXTNAME = 'PRIMARY '
    NAXIS   =                    0
    BITPIX  =                    8
    DATE-OBS= '2021-03-19T00:27:21.140552'
    NEXTEND =                    3
    OBS-LAT =              -30.166
    OBS-LONG=              -70.814
    OBS-ELEV=                 2200
    OBSERVAT= 'CTIO    '                                                            ]
    >>> hf = kbmock.HeaderFactory.from_ext_template()
    >>> hf.mock()
    [NAXIS   =                    2
    NAXIS1  =                 2048
    NAXIS2  =                 4096
    CRPIX1  =               1024.0 / Pixel coordinate of reference point
    CRPIX2  =               2048.0 / Pixel coordinate of reference point
    BITPIX  =                   32
    WCSAXES =                    2 / Number of coordinate axes
    PC1_1   = -5.5555555555556E-05 / Coordinate transformation matrix element
    PC2_2   =  5.5555555555556E-05 / Coordinate transformation matrix element
    CDELT1  =                  1.0 / [deg] Coordinate increment at reference point
    CDELT2  =                  1.0 / [deg] Coordinate increment at reference point
    CUNIT1  = 'deg'                / Units of coordinate increment and value
    CUNIT2  = 'deg'                / Units of coordinate increment and value
    CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
    CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
    CRVAL1  =                351.0 / [deg] Coordinate value at reference point
    CRVAL2  =                 -5.0 / [deg] Coordinate value at reference point
    LONPOLE =                180.0 / [deg] Native longitude of celestial pole
    LATPOLE =                 -5.0 / [deg] Native latitude of celestial pole
    MJDREF  =                  0.0 / [d] MJD of fiducial time
    RADESYS = 'ICRS'               / Equatorial coordinate system                   ]
    """
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
    """Template for the Primary header content."""

    ext_template = {"NAXIS": 2, "NAXIS1": 2048, "NAXIS2": 4096, "CRPIX1": 1024, "CRPIX2": 2048, "BITPIX": 32}
    """Template of an image-like extension header."""

    def __validate_mutables(self):
        """Validate number of mutables is number of callbacks, and that designated
        mutable cards exist in the given header template.
        """
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
        """Mocks headers, executing callbacks and creating WCS as necessary.

        Parameters
        ----------
        n : `int`
            Number of headers to mock.

        Returns
        -------
        headers : `list[Header]`
            Mocked headers.
        """
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
                self.wcs_factory.update_headers([header])
            headers.append(header)
            self.counter += 1
        return headers

    @classmethod
    def gen_header(cls, base, overrides=None, wcs=None):
        """Generate a header from a base template and overrides.

        If a WCS is given, and the header contains only a partially defined
        WCS, updates only the missing WCS cards and values.

        Parameters
        ----------
        base : `Header` or `dict-like`
            Header or a dict-like base template for the header.
        overrides : `Header`, `dict-like` or `None`, optional
            Keys and values that will either be updated or extended to the base
            template.
        wcs : `astropy.wcs.WCS` or `None`, optional
            WCS template to use to update the header values.

        Returns
        -------
        header : `astropy.io.fits.Header`
            A header.
        """
        header = Header(base)
        if overrides is not None:
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
        """Create a header assuming the default template of a PRIMARY header.

        Override, or extend the default template with keys and values in override.
        Attach callbacks to mutable cards.

        Parameters
        ----------
        overrides : `dict-like` or `None`, optional
            A header, or different dict-like, object used to override the base
            template keys and values.
        mutables : `list[str]` or `None`, optional
            List of card keys designated as mutable.
        callbacks : `list[callable]` or `None`, optional
            List of callable functions or classes that match the mutables.

        Returns
        -------
        factory : `HeaderFactory`
            Header factory.
        """
        hdr = cls.gen_header(base=cls.primary_template, overrides=overrides)
        return cls(hdr, mutables, callbacks)

    @classmethod
    def from_ext_template(cls, overrides=None, mutables=None, callbacks=None, shape=None,
                          wcs=None):
        """Create an extension header assuming the default template of an image
        like header.

        Override, or extend the default template with keys and values in override.
        Attach callbacks to mutable cards.

        Parameters
        ----------
        overrides : `dict-like` or `None`, optional
            A header, or different dict-like, object used to override the base
            template keys and values.
        mutables : `list[str]` or `None`, optional
            List of card keys designated as mutable.
        callbacks : `list[callable]` or `None`, optional
            List of callable functions or classes that match the mutables.
        shape : `tuple` or `None`, optional
            Update the template description of data dimensions and the reference
            pixel.
        wcs : `astropy.wcs.WCS` or `None`, optional
            WCS Factory to use.

        Returns
        -------
        factory : `HeaderFactory`
            Header factory.
        """
        ext_template = cls.ext_template.copy()

        if shape is not None:
            ext_template["NAXIS1"] = shape[0]
            ext_template["NAXIS2"] = shape[1]
            ext_template["CRPIX1"] = shape[0] // 2
            ext_template["CRPIX2"] = shape[1] // 2

        hdr = cls.gen_header(base=ext_template, overrides=overrides)
        return cls(hdr, mutables, callbacks, has_wcs=True, wcs_factory=wcs)


class ArchivedHeader(HeaderFactory):
    """Archived header factory.

    Archived headers are those that were produced with the modified version of
    AstroPy's ``fitsheader`` utility available in this module. See

        archiveheaders -h

    for more details. To produce an archive, with KBMOD installed, execute the
    following:

        archiveheaders *fits > archive.ecsv | tar -cf headers.tar.bz2 archive.ecsv

    Attributes
    ----------
    lexical_type_map : `dict`
        A map between the serialized names of built-in types and the built-in
        types. Used to cast the serialized card values before creating a Header.
    compression : `str`
        By default it's assumed the TAR archive was compressed with the ``bz2``
        compression algorithm.
    format : `str`
        The format in which the file of serialized header cards was written in.
        An AstroPy By default, this is assumed to be ``ascii.ecsv``

    Parameters
    ----------
    archive_name : `str`
        Name of the TAR archive containing serialized headers.
    fname : `str`
        Name of the file, within the archive, containing the headers.
    external : `bool`
        When `True`, file will be searched for relative to the current working
        directory. Otherwise, the file is searched for within the header archive
        provided with this module.
    """
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
    """Compression used to compress the archived headers."""

    format = "ascii.ecsv"
    """Format of the archive, and AstroPy'S ASCII module valid identifier of a format."""

    def __init__(self, archive_name, fname, external=False):
        super().__init__({})
        self.table = header_archive_to_table(archive_name, fname, self.compression, self.format, external=external)

        # Create HDU groups for easier iteration
        self.table = self.table.group_by("filename")
        self.n_hdus = len(self.table)

    def lexical_cast(self, value, vtype):
        """Cast str literal of a type to the type itself. Supports just
        the builtin Python types.
        """
        if vtype in self.lexical_type_map:
            return self.lexical_type_map[vtype](value)
        return value

    def get_item(self, group_idx, hdr_idx):
        """Get an extension of an HDUList within the archive without
        incrementing the mocking counter.

        Parameters
        ----------
        group_idx : `int`
            Index of the HDUList to fetch from the archive.
        hdr_idx : `int`
            Index of the extension within the HDUList.

        Returns
        -------
        header : `Header`
            Header of the given extension of the targeted HDUList.
        """
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
        """Get an HDUList within the archive without incrementing the mocking counter.

        Parameters
        ----------
        group_idx : `int`
            Index of the HDUList to fetch from the archive.

        Returns
        -------
        headers : `list[Header]`
            All headers of the targeted HDUList.
        """
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
        """Mock all headers within n HDULists.

        Parameters
        ----------
        n : `int`
            Number of HDUList units which headers we want mocked.

        Returns
        -------
        headers : `list[list[Header]]`
            A list of containing headers of each extension of an HDUList.
        """
        res = []
        for _ in range(n):
            res.append(self.get(self.counter))
            self.counter += 1
        return res

