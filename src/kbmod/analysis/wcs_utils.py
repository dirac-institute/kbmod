import astropy.wcs
import astropy.units
import astropy.coordinates
import numpy

# if True then use when image_fov is provided with no pixel_scale solve for a pixel_scale that results in the image_fov
solve_for_image_fov = False
# astropy convention is that the center of the pixel is where the pixel value is measured. This is meant to offset from
# the pixel sampling point to the minimum valued edge of the pixel. That is half a pixel down.
image_offset = -0.5


def construct_wcs_tangent_projection(
    ref_val: astropy.coordinates.SkyCoord,
    img_shape: list[int] = [4096, 4096],
    ref_pix=None,
    pixel_scale: astropy.units.Quantity = None,
    image_fov: astropy.units.Quantity = None,
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
        The field of view of the larger image dimension, by default None
        If provided the pixel scale is calculated from the image shape and the field of view
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
    _solve_for_image_fov = False
    if pixel_scale is None:
        if image_fov is None:
            pixel_scale = 0.2 * astropy.units.arcsec / astropy.units.pix
        else:
            # not convinced this is correct. It does not seem to result in an image fov of image_fov
            # but then I could be measuring the image fov incorrectly. The tangent performed no better.
            pixel_scale = image_fov / max(img_shape) / astropy.units.pix
            _solve_for_image_fov = solve_for_image_fov
    # Define the WCS transformation
    wcstan = _wcs_tangent_projection(ref_val, img_shape, ref_pix, pixel_scale)
    if _solve_for_image_fov:
        wcstan = solve_wcs_for_fov(wcstan, image_fov)
    return wcstan


def _wcs_tangent_projection(ref_val, img_shape, ref_pix, pixel_scale):
    """Construct a WCS tangent projection.
    Similar to construct_wcs_tangent_projection but without the checks, default values, and image_fov parameter.
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
    """Solve for the pixel scale of the WCS given the field of view."""
    # Assuming that the pixel scale is the arc of the reference pixel.
    ref_pixel = wcs.wcs.crpix
    image_size = wcs.array_shape
    pixel_scale_mark = wcs.wcs.cdelt[1]
    tolerance = 1e-12
    iterations = 1000
    pixel_scale_bounds = [pixel_scale_mark, pixel_scale_mark]
    refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale_mark)
    delta = pixel_scale_mark / 16.0
    while 0 < iterations and tolerance < abs(refsep2[0] - fov).value and fov <= refsep2[0]:
        delta *= 2.0
        pixel_scale_bounds[0] -= delta
        refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale_bounds[0])
        iterations -= 1
    delta = pixel_scale_mark / 16.0
    while 0 < iterations and tolerance < abs(refsep2[0] - fov).value and refsep2[0] < fov:
        delta *= 2.0
        pixel_scale_bounds[1] += delta
        refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale_bounds[1])
        iterations -= 1
    pixel_scale = -pixel_scale_mark
    while 0 < iterations and tolerance < abs(refsep2[0] - fov).value and pixel_scale_mark != pixel_scale:
        pixel_scale_mark = pixel_scale
        pixel_scale = sum(pixel_scale_bounds) / 2.0
        refsep2 = _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale)
        if refsep2[0] < fov:
            pixel_scale_bounds[0] = pixel_scale
        elif fov < refsep2[0]:
            pixel_scale_bounds[1] = pixel_scale
        iterations -= 1
    return wcs


def calc_actual_image_fov(wcs):
    ref_pixel = wcs.wcs.crpix
    image_size = wcs.array_shape
    return _calc_actual_image_fov(wcs, ref_pixel, image_size)


def _update_calc_fov(wcs, ref_pixel, image_size, pixel_scale):
    wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
    return _calc_actual_image_fov(wcs, ref_pixel, image_size)


def _calc_actual_image_fov(wcs, ref_pixel, image_size):
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
