"""A collection of utility functions for working with WCS in KBMOD."""

import astropy.coordinates
import astropy.units
import astropy.wcs
import json
import numpy


def calc_ecliptic_angle(wcs, center_pixel=(1000, 2000), step=12):
    """Projects an unit-vector parallel with the ecliptic onto the image
    and calculates the angle of the projected unit-vector in the pixel space.

    Parameters
    ----------
    wcs : ``astropy.wcs.WCS``
        World Coordinate System object.
    center_pixel : tuple, array-like
        Pixel coordinates of image center.
    step : ``float`` or ``int``
        Size of step, in arcseconds, used to find the pixel coordinates of
        the second pixel in the image parallel to the ecliptic.

    Returns
    -------
    suggested_angle : ``float``
        Angle the projected unit-vector parallel to the eclipticc closes
        with the image axes. Used to transform the specified search angles,
        with respect to the ecliptic, to search angles within the image.

    Notes
    -----
    It is not neccessary to calculate this angle for each image in an
    image set if they have all been warped to a common WCS.
    """
    # pick a starting pixel approximately near the center of the image
    # convert it to ecliptic coordinates
    start_pixel = numpy.array(center_pixel)
    start_pixel_coord = astropy.coordinates.SkyCoord.from_pixel(start_pixel[0], start_pixel[1], wcs)
    start_ecliptic_coord = start_pixel_coord.geocentrictrueecliptic

    # pick a guess pixel by moving parallel to the ecliptic
    # convert it to pixel coordinates for the given WCS
    guess_ecliptic_coord = astropy.coordinates.SkyCoord(
        start_ecliptic_coord.lon + step * astropy.units.arcsec,
        start_ecliptic_coord.lat,
        frame="geocentrictrueecliptic",
    )
    guess_pixel_coord = guess_ecliptic_coord.to_pixel(wcs)

    # calculate the distance, in pixel coordinates, between the guess and
    # the start pixel. Calculate the angle that represents in the image.
    x_dist, y_dist = numpy.array(guess_pixel_coord) - start_pixel
    return numpy.arctan2(y_dist, x_dist)


def extract_wcs_from_hdu_header(header):
    """Read an WCS from the an HDU header and do basic validity checking.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        The header from which to read the data.

    Returns
    -------
    curr_wcs : `astropy.wcs.WCS`
        The WCS or None if it does not exist.
    """
    # Check that we have (at minimum) the CRVAL and CRPIX keywords.
    # These are necessary (but not sufficient) requirements for the WCS.
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None
    if "CRPIX1" not in header or "CRPIX2" not in header:
        return None

    if "DIMM1" in header and "DIMM2" in header:
        naxis1 = header["DIMM1"]
        naxis2 = header["DIMM2"]
    else:
        naxis1 = None
        naxis2 = None

    curr_wcs = astropy.wcs.WCS(header)
    if curr_wcs is None:
        return None
    if curr_wcs.naxis != 2:
        return None

    if curr_wcs is not None and naxis1 is not None:
        curr_wcs.array_shape = (naxis2, naxis1)

    return curr_wcs


def append_wcs_to_hdu_header(wcs, header):
    """Append the WCS fields to an existing HDU header.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS` or `dict`
        The WCS to use or a dictionary with the necessary information.
    header : `astropy.io.fits.Header`
        The header to which to append the data.
    """
    if wcs is not None:
        if type(wcs) is dict:
            wcs_map = wcs
        else:
            wcs_map = wcs.to_header()

            # shhh... don't tell astropy we're doing this
            # (astropy will refuse to store a "NAXIS[1/2]" key)
            if wcs.array_shape is not None:
                naxis2, naxis1 = wcs.array_shape
                header["DIMM1"] = naxis1
                header["DIMM2"] = naxis2

        for key in wcs_map:
            header[key] = wcs_map[key]


def serialize_wcs(wcs):
    """Convert a WCS into a JSON string.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS` or None
        The WCS to convert.

    Returns
    -------
    wcs_str : `str`
        The serialized WCS. Returns an empty string if wcs is None.
    """
    if wcs is None:
        return ""

    # Since AstroPy's WCS does not output NAXIS, we need to manually add those.
    header = wcs.to_header(relax=True)
    header["NAXIS1"], header["NAXIS2"] = wcs.pixel_shape
    return json.dumps(dict(header))


def deserialize_wcs(wcs_str):
    """Convert a JSON string into a WCS object.

    Parameters
    ----------
    wcs_str : `str`
        The serialized WCS.

    Returns
    -------
    wcs : `astropy.wcs.WCS` or None
        The resulting WCS or None if no data is provided.
    """
    if wcs_str == "" or wcs_str.lower() == "none":
        return None

    wcs_dict = json.loads(wcs_str)
    wcs = astropy.wcs.WCS(wcs_dict)
    wcs.pixel_shape = (wcs_dict["NAXIS1"], wcs_dict["NAXIS2"])
    return wcs


def make_fake_wcs(center_ra, center_dec, height, width, deg_per_pixel=None):
    """Create a fake WCS given basic information. This is not a realistic
    WCS in terms of astronomy, but can provide a place holder for many tests.

    Parameters
    ----------
    center_ra : `float`
        The center of the pointing in RA (degrees)
    center_dev : `float`
        The center of the pointing in DEC (degrees)
    height : `int`
        The image height (pixels).
    width : `int`
        The image width (pixels).
    deg_per_pixel : `float`, optional
        The angular resolution of a pixel (degrees).

    Returns
    -------
    result : `astropy.wcs.WCS`
        The resulting WCS.
    """
    wcs_dict = {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRVAL1": center_ra,
        "CRVAL2": center_dec,
        "CRPIX1": height / 2.0,
        "CRPIX2": width / 2.0,
        "CTYPE1A": "LINEAR  ",
        "CTYPE2A": "LINEAR  ",
        "CUNIT1A": "PIXEL   ",
        "CUNIT2A": "PIXEL   ",
        "NAXIS1": width,
        "NAXIX2": height,
    }

    if deg_per_pixel is not None:
        wcs_dict["CDELT1"] = deg_per_pixel
        wcs_dict["CDELT2"] = deg_per_pixel

    wcs = astropy.wcs.WCS(wcs_dict)
    wcs.array_shape = (height, width)
    return wcs


def wcs_fits_equal(wcs_a, wcs_b):
    """Test if two WCS objects are equal at the level they would be
    written to FITS headers. Treats a pair of None values as equal.

    Parameters
    ----------
    wcs_a : `astropy.wcs.WCS`
        The WCS of the first object.
    wcs_b : `astropy.wcs.WCS`
        The WCS of the second object.
    """
    # Handle None data.
    if wcs_a is None and wcs_b is None:
        return True
    if wcs_a is None:
        return False
    if wcs_b is None:
        return False

    header_a = wcs_a.to_header()
    header_b = wcs_b.to_header()
    if len(header_a) != len(header_b):
        return False

    # Check the derived FITs header data have the same keys and values.
    for key in header_a:
        if not key in header_b:
            return False
        if header_a[key] != header_b[key]:
            return False

    # Check that we correctly kept the shape of the matrix.
    if wcs_a.array_shape != wcs_b.array_shape:
        return False
    if wcs_a.pixel_shape != wcs_b.pixel_shape:
        return False

    return True
