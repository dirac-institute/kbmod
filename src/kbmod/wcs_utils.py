"""A collection of utility functions for working with WCS in KBMOD."""

import astropy.coordinates
import astropy.units
import astropy.wcs
import numpy


def construct_wcs_tangent_projection(
    ref_val,
    img_shape=[4096, 4096],
    ref_pix=None,
    pixel_scale: astropy.units.Quantity = None,
    image_fov: astropy.units.Quantity = None,
    solve_for_image_fov: bool = False,
):
    """Construct a WCS tangent projection.

    Parameters
    ----------
    ref_val : astropy.coordinates.SkyCoord
        The reference coordinate of the tangent projection.
        If None is given the reference coordinate is set to (0, -40) degrees.
    img_shape : list[int], optional
        The shape of the image, by default [4096, 4096]
    ref_pix : list[float], optional
        The reference pixel of the tangent projection, by default None
        If None is given, the image center pixel is used.
        Note that the values are one based in keeping with the astropy.wcs.WCS convention
        which contrasts with the zero based convention for the pixel coordinates.
    pixel_scale : astropy.units.Quantity, optional
        The pixel scale of the image, by default None
        If None and image_fov is provided, the pixel scale is calculated from the image field of view.
        If None and image_fov is not provided, the pixel scale is set to 0.2 arcsec/pixel to match LSST.
    image_fov : astropy.units.Quantity, optional
        The field of view of the first image dimension, by default None
        If provided the pixel scale is calculated from the image shape and the field of view
    solve_for_image_fov : bool, optional
        If True, image_fov is provided, and pixel_scale is None then a (slow) search is made
        to make the horizontal field of view match image_fov. This should be analytic but
        as yet no such solution is contained. The default is False.

    Returns
    -------
    astropy.wcs.WCS
        The WCS tangent projection.
    """
    if min(img_shape) < 1:
        raise ValueError("The image shape must be greater than zero.")
    if ref_val is None:
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=-40 * astropy.units.deg, frame="icrs"
        )
    if ref_pix is None:
        # this is the central one based pixel coordinate
        ref_pix = [((img_shape[0] - 1) / 2 + 1), ((img_shape[0] - 1) / 2 + 1)]
    enable_fov_solution = False
    if pixel_scale is None:
        if image_fov is None:
            pixel_scale = 0.2 * astropy.units.arcsec / astropy.units.pix
        else:
            # not convinced this is correct. It does not seem to result in an image fov of image_fov
            # but then I could be measuring the image fov incorrectly. The tangent performed no better.
            pixel_scale = image_fov / max(img_shape) / astropy.units.pix
            enable_fov_solution = solve_for_image_fov
    # Define the WCS transformation
    wcstan = _wcs_tangent_projection(ref_val, img_shape, ref_pix, pixel_scale)
    if enable_fov_solution:
        wcstan = solve_wcs_for_fov(wcstan, image_fov)
    return wcstan


def _wcs_tangent_projection(ref_val, img_shape, ref_pix, pixel_scale):
    """Construct a WCS tangent projection.
    Similar to construct_wcs_tangent_projection but without the checks, default values, and image_fov parameter.

    Parameters
    ----------
    ref_val : astropy.coordinates.SkyCoord
        The reference coordinate of the tangent projection.
    img_shape : list[int]
        The shape of the image.
    ref_pix : list[float]
        The one based reference pixel of the tangent projection.
    pixel_scale : astropy.units.Quantity
        The pixel scale of the image.

    Returns
    -------
    astropy.wcs.WCS
        The WCS tangent projection.
    """
    pixel_scale_value = pixel_scale.to(astropy.units.deg / astropy.units.pix).value
    wcstan = astropy.wcs.WCS(naxis=2)
    wcstan.array_shape = img_shape
    wcstan.wcs.crpix = ref_pix
    wcstan.wcs.cdelt = [-pixel_scale_value, pixel_scale_value]
    wcstan.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcstan.wcs.crval = [ref_val.ra.deg, ref_val.dec.deg]
    return wcstan


def solve_wcs_for_fov(wcs, fov):
    """Solve for the pixel scale of the WCS given the field of view.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS to solve for.
    fov : astropy.units.Quantity
        The field of view of the first image dimension to solve for.

    Returns
    -------
    astropy.wcs.WCS
        The WCS with the pixel scale adjusted to match the field of view.

    Notes
    -----
    This is a slow iterative solution. It should be possible to solve this analytically.
    The method starts with the pixel scale of the reference pixel and then expands the
    lower and upper search bounds by a factor of two until the field of view is bracketed.
    Then a bisection search is performed to find the pixel scale that matches the field of view.
    An iteration limit is imposed to prevent an infinite loop.
    """
    # Assuming that the pixel scale is the arc of the reference pixel.
    ref_pixel = wcs.wcs.crpix
    image_size = wcs.array_shape
    cdelt_original = wcs.wcs.cdelt[1]
    tolerance = 1e-12
    iterations = 1000

    refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, cdelt_original)

    # early out check so we don't waste time if the fov is already close.
    if abs(refsep2[0] - fov).value < tolerance:
        return wcs

    # start with bounds equal to the pixel scale. If the fov is already close
    # then this will be the solution.
    pixel_scale_bounds = [cdelt_original, cdelt_original]
    # only one of the bounds needs to be expanded
    if fov <= refsep2[0]:
        # expand the lower bound by delta (doubling each iteration) until the fov is bracketed
        delta = cdelt_original / 16.0
        while 0 < iterations and tolerance < abs(refsep2[0] - fov).value and fov <= refsep2[0]:
            delta *= 2.0
            pixel_scale_bounds[0] -= delta
            refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale_bounds[0])
            iterations -= 1
    else:
        # expand the upper bound by delta (doubling each iteration) until the fov is bracketed
        delta = cdelt_original / 16.0
        while 0 < iterations and tolerance < abs(refsep2[0] - fov).value and refsep2[0] < fov:
            delta *= 2.0
            pixel_scale_bounds[1] += delta
            refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale_bounds[1])
            iterations -= 1

    # assert(pixel_scale_bounds[0] < cdelt_original <= pixel_scale_bounds[1])
    previous_pixel_scale = cdelt_original
    pixel_scale = sum(pixel_scale_bounds) / 2.0
    # assert(previous_pixel_scale != pixel_scale)
    # The first test will always be true. Uses a bisection search to find the pixel scale that results in the fov.
    # Terminate when the results is within tolerance or there is no numerical change in the pixel scale or
    # a maximum number of iterations is reached.
    while 0 < iterations and tolerance < abs(refsep2[0] - fov).value and previous_pixel_scale != pixel_scale:
        refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale)
        if refsep2[0] < fov:
            pixel_scale_bounds[0] = pixel_scale
        elif fov < refsep2[0]:
            pixel_scale_bounds[1] = pixel_scale
        previous_pixel_scale = pixel_scale
        pixel_scale = sum(pixel_scale_bounds) / 2.0
        iterations -= 1
    return wcs


def calc_actual_image_fov(wcs):
    """Calculate the actual image field of view in degrees calculated for
    the wcs measured from image edge to edge along lines passing through
    the reference pixel.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to calculate the field of view for.

    Returns
    -------
    astropy.units.Quantity
        The field of view in degrees measured edge to edge across
        both image dimensions and through the reference pixel.

    Notes
    -----
    While this will work for any WCS, it is intended to be used with
    tangent plane projections where the reference pixel is on the image.
    """
    ref_pixel = wcs.wcs.crpix
    image_size = wcs.array_shape
    return _calc_actual_image_fov(wcs, ref_pixel, image_size)


def _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale):
    """Update the WCS with the new pixel scale and calculate the field of view.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to udate and to calculate the field of view for.
    ref_pixel : list
        The one based reference pixel of the WCS used for the calculation.
    image_size : list
        The size of the image in pixels to use for the calculation.
        It should be the same size as the image used to create the WCS.
    pixel_scale : float
        The pixel scale to use for the calculation.

    Returns
    -------
    astropy.units.Quantity
        The field of view in degrees measured from edge to edge of
        both image dimensions and through the reference pixel.

    Notes
    -----
    This function is intended for DRY internal use. It modifies
    the WCS in place.
    """
    wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
    return _calc_actual_image_fov(wcs, ref_pixel, image_size)


def _calc_actual_image_fov(wcs, ref_pixel, image_size):
    """Calculate the actual image field of view in degrees.
    Measure for both image dimensions through the reference pixel.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to calculate the field of view for.
    ref_pixel : list
        The one based reference pixel of the WCS used for the calculation.
    image_size : list
        The size of the image in pixels to use for the calculation.
        It should be the same size as the image used to create the WCS.

    Returns
    -------
    astropy.units.Quantity
        The field of view in degrees measured for
        both dimensions.

    Notes
    -----
    This function is intended for DRY internal use. It does not
    modify the WCS object.
    """

    # The image offset is used to get the pixel area edge from
    # the reference position which is at the pixel center.
    image_offset = -0.5

    skyvallist = wcs.pixel_to_world(
        [
            0 + image_offset,
            image_size[0] + image_offset,
            ref_pixel[0] - 1 + image_offset,
            ref_pixel[0] - 1 + image_offset,
        ],
        [
            ref_pixel[1] - 1 + image_offset,
            ref_pixel[1] - 1 + image_offset,
            0 + image_offset,
            image_size[1] + image_offset,
        ],
    )
    refsep2 = [skyvallist[0].separation(skyvallist[1]), skyvallist[2].separation(skyvallist[3])]
    return refsep2


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

    Note
    ----
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
    --------
    curr_wcs : `astropy.wcs.WCS`
        The WCS or None if it does not exist.
    """
    # Check that we have (at minimum) the CRVAL and CRPIX keywords.
    # These are necessary (but not sufficient) requirements for the WCS.
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return None
    if "CRPIX1" not in header or "CRPIX2" not in header:
        return None

    curr_wcs = astropy.wcs.WCS(header)
    if curr_wcs is None:
        return None
    if curr_wcs.naxis != 2:
        return None

    return curr_wcs


def wcs_from_dict(data):
    """Extract a WCS from a fictionary of the HDU header keys/values.
    Performs very basic validity checking.

    Parameters
    ----------
    data : `dict`
        A dictionary containing the WCS header information.

    Returns
    -------
    wcs : `astropy.wcs.WCS`
        The WCS to convert.
    """
    # Check that we have (at minimum) the CRVAL and CRPIX keywords.
    # These are necessary (but not sufficient) requirements for the WCS.
    if "CRVAL1" not in data or "CRVAL2" not in data:
        return None
    if "CRPIX1" not in data or "CRPIX2" not in data:
        return None

    curr_wcs = astropy.wcs.WCS(data)
    if curr_wcs is None:
        return None
    if curr_wcs.naxis != 2:
        return None

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

        for key in wcs_map:
            header[key] = wcs_map[key]


def wcs_to_dict(wcs):
    """Convert a WCS to a dictionary (via a FITS header).

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        The WCS to convert.

    Returns
    -------
    result : `dict`
        A dictionary containing the WCS header information.
    """
    result = {}
    if wcs is not None:
        wcs_header = wcs.to_header()
        for key in wcs_header:
            result[key] = wcs_header[key]
    return result


def make_fake_wcs_info(center_ra, center_dec, height, width, deg_per_pixel=None):
    """Create a fake WCS dictionary given basic information. This is not a realistic
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
    wcs_dict : `dict`
        A dictionary with the information needed for a WCS.
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
    }

    if deg_per_pixel is not None:
        wcs_dict["CDELT1"] = deg_per_pixel
        wcs_dict["CDELT2"] = deg_per_pixel

    return wcs_dict


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
    wcs_dict = make_fake_wcs_info(center_ra, center_dec, height, width, deg_per_pixel)
    return astropy.wcs.WCS(wcs_dict)


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

    return True
