import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS, ICRS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.optimize import minimize


def correct_parallax(coord, obstime, point_on_earth, guess_distance):
    """Calculate the parallax corrected postions for a given object at a given time and distance from Earth.

    Attributes
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time` or `string`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    guess_distance : `float`
        The guess distance to the object from Earth.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the pointin ICRS.

    References
    ----------
    .. [1] `Jupyter Notebook <https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb>`_
    """
    loc = (
        point_on_earth.x.to(u.m).value,
        point_on_earth.y.to(u.m).value,
        point_on_earth.z.to(u.m).value,
    ) * u.m

    # line of sight from earth to the object,
    # the object has an unknown distance from earth
    los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))

    cost = lambda d: np.abs(
        guess_distance
        - GCRS(ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=d * u.AU, obstime=obstime, obsgeoloc=loc)
        .transform_to(ICRS())
        .distance.to(u.AU)
        .value
    )

    fit = minimize(
        cost,
        (guess_distance,),
    )

    answer = GCRS(
        ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=fit.x[0] * u.AU, obstime=obstime, obsgeoloc=loc
    ).transform_to(ICRS())

    return answer


def fit_barycentric_wcs(
    original_wcs, width, height, distance, obstime, point_on_earth, npoints=10, seed=None
):
    """Given a ICRS WCS and an object's distance from the Sun,
    return a new WCS that has been corrected for parallax motion.

    Attributes
    ----------
    original_wcs : `astropy.wcs.WCS`
        The image's WCS.
    width : `int`
        The image's width (typically NAXIS1).
    height : `int`
        The image's height (typically NAXIS2).
    distance : `float`
        The distance of the object from the sun, in AU.
    obstime : `astropy.time.Time` or `string`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    npoints : `int`
        The number of randomly sampled points to use during the WCS fitting.
        Typically, the more points the higher the accuracy. The four corners
        of the image will always be included, so setting npoints = 0 will mean
        just using the corners.
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
        the seed that `numpy.random.default_rng` will use.

    Returns
    ----------
    An `astropy.wcs.WCS` representing the original image in "Explicity Barycentric Distance" (EBD)
    space, i.e. where the points have been corrected for parallax.
    """
    rng = np.random.default_rng(seed)

    sampled_x_points = np.array([0, 0, width, width])
    sampled_y_points = np.array([0, height, height, 0])
    if npoints > 0:
        sampled_x_points = np.append(sampled_x_points, rng.random(npoints) * width)
        sampled_y_points = np.append(sampled_y_points, rng.random(npoints) * height)

    sampled_ra, sampled_dec = original_wcs.all_pix2world(sampled_x_points, sampled_y_points, 0)

    sampled_coordinates = SkyCoord(sampled_ra, sampled_dec, unit="deg")

    ebd_corrected_points = []
    for coord in sampled_coordinates:
        ebd_corrected_points.append(correct_parallax(coord, obstime, point_on_earth, distance))

    ebd_corrected_points = SkyCoord(ebd_corrected_points)
    xy = (sampled_x_points, sampled_y_points)
    ebd_wcs = fit_wcs_from_points(
        xy, ebd_corrected_points, proj_point="center", projection="TAN", sip_degree=3
    )
    return ebd_wcs
