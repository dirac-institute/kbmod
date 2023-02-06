import astropy.units as u
import astropy.coordinates as astroCoords
import numpy as np

from astropy.time import Time
from numpy.linalg import lstsq


def calc_ecliptic_angle(wcs, center_pixel=(1000, 2000), step=12):
    """Projects an unit-vector parallel with the ecliptic onto the image
    and calculates the angle of the projected unit-vector in the pixel
    space.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        World Coordinate System object.
    center_pixel : tuple, array-like
        Pixel coordinates of image center.
    step : float or int
        Size of step, in arcseconds, used to find the pixel coordinates of
            the second pixel in the image parallel to the ecliptic.

    Returns
    -------
    ec_angle : float
        Angle the projected unit-vector parallel to the ecliptic
        closes with the image axes. Used to transform the specified
        search angles, with respect to the ecliptic, to search angles
        within the image.

    Note
    ----
    It is not neccessary to calculate this angle for each image in an
    image set if they have all been warped to a common WCS.

    See Also
    --------
    run_search.do_gpu_search
    """
    # pick a starting pixel approximately near the center of the image
    # convert it to ecliptic coordinates
    start_pixel = np.array(center_pixel)
    start_pixel_coord = astroCoords.SkyCoord.from_pixel(start_pixel[0], start_pixel[1], wcs)
    start_ecliptic_coord = start_pixel_coord.geocentrictrueecliptic

    # pick a guess pixel by moving parallel to the ecliptic
    # convert it to pixel coordinates for the given WCS
    guess_ecliptic_coord = astroCoords.SkyCoord(
        start_ecliptic_coord.lon + step * u.arcsec,
        start_ecliptic_coord.lat,
        frame="geocentrictrueecliptic",
    )
    guess_pixel_coord = guess_ecliptic_coord.to_pixel(wcs)

    # calculate the distance, in pixel coordinates, between the guess and
    # the start pixel. Calculate the angle that represents in the image.
    x_dist, y_dist = np.array(guess_pixel_coord) - start_pixel
    return np.arctan2(y_dist, x_dist)


def calc_barycentric_corr(img_info, dist):
    """This function calculates the barycentric corrections between
    each image and the first.

    The barycentric correction is the shift in x,y pixel position expected for
    an object that is stationary in barycentric coordinates, at a barycentric
    radius of dist au. This function returns a linear fit to the barycentric
    correction as a function of position on the first image.

    Parameters
    ----------
    img_info : `kbmod.search.ImageInfo`
        ImageInfo
    dist : `float`
        Distance to object from barycenter in AU.
    """
    wcslist = [img_info.stats[i].wcs for i in range(img_info.num_images)]
    mjdlist = np.array(img_info.get_all_mjd())
    x_size = img_info.get_x_size()
    y_size = img_info.get_y_size()

    # make grid with observer-centric RA/DEC of first image
    xlist, ylist = np.mgrid[0:x_size, 0:y_size]
    xlist = xlist.flatten()
    ylist = ylist.flatten()
    cobs = wcslist[0].pixel_to_world(xlist, ylist)

    # Convert this grid to barycentric x,y,z, assuming distance r
    # [obs_to_bary_wdist()]
    with astroCoords.solar_system_ephemeris.set("de432s"):
        obs_pos = astroCoords.get_body_barycentric("earth", Time(mjdlist[0], format="mjd"))
    cobs.representation_type = "cartesian"

    # barycentric distance of observer
    r2_obs = obs_pos.x * obs_pos.x + obs_pos.y * obs_pos.y + obs_pos.z * obs_pos.z

    # calculate distance r along line of sight that gives correct
    # barycentric distance
    # |obs_pos + r * cobs|^2 = dist^2
    # obs_pos^2 + 2r (obs_pos dot cobs) + cobs^2 = dist^2
    dot = obs_pos.x * cobs.x + obs_pos.y * cobs.y + obs_pos.z * cobs.z
    bary_dist = dist * u.au
    r = -dot + np.sqrt(bary_dist * bary_dist - r2_obs + dot * dot)

    # barycentric coordinate is observer position + r * line of sight
    cbary = astroCoords.SkyCoord(
        obs_pos.x + r * cobs.x,
        obs_pos.y + r * cobs.y,
        obs_pos.z + r * cobs.z,
        representation_type="cartesian",
    )

    baryCoeff = np.zeros((len(wcslist), 6))
    for i in range(1, len(wcslist)):  # corections for wcslist[0] are 0
        # hold the barycentric coordinates constant and convert to new frame
        # by subtracting the observer's new position and converting to RA/DEC and pixel
        # [bary_to_obs_fast()]
        with astroCoords.solar_system_ephemeris.set("de432s"):
            obs_pos = astroCoords.get_body_barycentric("earth", Time(mjdlist[i], format="mjd"))
        c = astroCoords.SkyCoord(
            cbary.x - obs_pos.x, cbary.y - obs_pos.y, cbary.z - obs_pos.z, representation_type="cartesian"
        )
        c.representation_type = "spherical"
        pix = wcslist[i].world_to_pixel(c)

        # do linear fit to get coefficients
        ones = np.ones_like(xlist)
        A = np.stack([ones, xlist, ylist], axis=-1)
        coef_x, _, _, _ = lstsq(A, (pix[0] - xlist))
        coef_y, _, _, _ = lstsq(A, (pix[1] - ylist))
        baryCoeff[i, 0:3] = coef_x
        baryCoeff[i, 3:6] = coef_y

    return baryCoeff
