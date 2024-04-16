import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS, ICRS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.optimize import minimize


def correct_parallax(coord, obstime, point_on_earth, heliocentric_distance):
    """Calculate the parallax corrected postions for a given object at a given time and distance from Earth.

    Attributes
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time` or `string`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    heliocentric_distance : `float`
        The guess distance to the object from the Sun.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS, and the best fit geocentric distance (float).

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

    cost = lambda geocentric_distance: np.abs(
        heliocentric_distance
        - GCRS(ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=geocentric_distance * u.AU, obstime=obstime, obsgeoloc=loc)
        .transform_to(ICRS())
        .distance.to(u.AU)
        .value
    )

    fit = minimize(
        cost,
        (heliocentric_distance,),
    )

    answer = SkyCoord(
        ra=los_earth_obj.ra, dec=los_earth_obj.dec, distance=fit.x[0] * u.AU, obstime=obstime, obsgeoloc=loc, frame="gcrs"
    ).transform_to(ICRS())

    return answer, fit.x[0]

def invert_correct_parallax(coord, obstime, point_on_earth, geocentric_distance, heliocentric_distance):
    """Calculate the original ICRS coordinates of a point in EBD space, i.e. a result from `correct_parallax`.

    Attributes
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time` or `string`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    geocentric_distance : `float`
        The distance from Earth to the object (generally a result from `correct_parallax`).

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS.

    References
    ----------
    .. [1] `Jupyter Notebook <https://github.com/maxwest-uw/notebooks/blob/main/uncorrecting_parallax.ipynb>`_
    """
    loc = (
        point_on_earth.x.to(u.m).value,
        point_on_earth.y.to(u.m).value,
        point_on_earth.z.to(u.m).value,
    ) * u.m
    icrs_with_dist = ICRS(ra=coord.ra, dec=coord.dec, distance=heliocentric_distance * u.au)

    gcrs_no_dist = icrs_with_dist.transform_to(GCRS(obsgeoloc=loc, obstime=obstime))
    gcrs_with_dist = GCRS(
        ra=gcrs_no_dist.ra,
        dec=gcrs_no_dist.dec,
        distance=geocentric_distance,
        obsgeoloc=loc,
        obstime=obstime
    )

    original_icrs = gcrs_with_dist.transform_to(ICRS())
    return original_icrs


def fit_barycentric_wcs(
    original_wcs, width, height, heliocentric_distance, obstime, point_on_earth, npoints=10, seed=None
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
    heliocentric_distance : `float`
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
    geocentric_distances = []
    for coord in sampled_coordinates:
        coord, geo_dist = correct_parallax(coord, obstime, point_on_earth, heliocentric_distance)
        ebd_corrected_points.append(coord)
        geocentric_distances.append(geo_dist)

    ebd_corrected_points = SkyCoord(ebd_corrected_points)
    xy = (sampled_x_points, sampled_y_points)
    ebd_wcs = fit_wcs_from_points(
        xy, ebd_corrected_points, proj_point="center", projection="TAN", sip_degree=3
    )
    geocentric_distance = np.average(geocentric_distances)

    return ebd_wcs, geocentric_distance
