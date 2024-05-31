import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS, ICRS, get_body_barycentric
from astropy.wcs.utils import fit_wcs_from_points
from scipy.optimize import minimize


def correct_parallax(coord, obstime, point_on_earth, heliocentric_distance, geocentric_distance=None, method=None, use_bounds=False):
    """Calculate the parallax corrected postions for a given object at a given time and distance from Earth.

    Parameters
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time` or `string`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    heliocentric_distance : `float`
        The guess distance to the object from the Sun.
    geocentric_distance : `float` or `None` (optional)
        If the geocentric distance to be corrected for is already known,
        you can pass it in here. This will avoid the computationally expensive
        minimizer call.
    method : `string` (optional)
        The minimization algorithm to use. Default is "Nelder-Mead".
    use_bounds : `bool` (optional)
        If True, the minimizer will be bounded heliocentric distance +/- 1.02.
        Default is True.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS, and the best fit geocentric distance (float).

    References
    ----------
    .. [1] `Jupyter Notebook <https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb>`_
    """
    loc = (point_on_earth.to_geocentric()) * u.m

    # line of sight from earth to the object,
    # the object has an unknown distance from earth
    los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))

    if geocentric_distance is None:
        cost = lambda geocentric_distance: np.abs(
            heliocentric_distance
            - GCRS(
                ra=los_earth_obj.ra,
                dec=los_earth_obj.dec,
                distance=geocentric_distance * u.AU,
                obstime=obstime,
                obsgeoloc=loc,
            )
            .transform_to(ICRS())
            .distance.to(u.AU)
            .value
        )

        # range of geocentric distances to search. 1.02 is Earth aphelion in au.
        bounds = None
        if use_bounds:
            bounds = [(max(0., heliocentric_distance-1.02), heliocentric_distance+1.02)]

        fit = minimize(
            cost,
            (heliocentric_distance,),
            method=method,
            bounds=bounds,
        )
        geocentric_distance = fit.x[0]

    answer = SkyCoord(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=geocentric_distance * u.AU,
        obstime=obstime,
        obsgeoloc=loc,
        frame="gcrs",
    ).transform_to(ICRS())

    return answer, geocentric_distance


def correct_parallax2(coord, obstime, point_on_earth, heliocentric_distance):
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
    # Compute the Earth location relative to the barycenter.
    # times = Time(obstime, format="mjd")
    
    # Compute the Earth's location in to cartesian space centered the barycenter.
    # This is an approximate position. Is it good enough?
    earth_pos_cart = get_body_barycentric("earth", obstime)
    ex = earth_pos_cart.x.value + point_on_earth.x.to(u.au).value
    ey = earth_pos_cart.y.value + point_on_earth.y.to(u.au).value
    ez = earth_pos_cart.z.value + point_on_earth.z.to(u.au).value

    # Compute the unit vector of the pointing.
    loc = (point_on_earth.to_geocentric()) * u.m
    los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))

    pointings_cart = los_earth_obj.cartesian
    vx = pointings_cart.x.value
    vy = pointings_cart.y.value
    vz = pointings_cart.z.value

    # Solve the quadratic equation for the ray leaving the earth and intersecting
    # a sphere around the sun (0, 0, 0) with radius = heliocentric_distance
    a = vx * vx + vy * vy + vz * vz
    b = 2 * vx * ex + 2 * vy * ey + + 2 * vz * ez
    c = ex * ex + ey * ey + ez * ez - heliocentric_distance * heliocentric_distance
    disc = b * b - 4 * a * c

    if (disc < 0):
        return None, -1.0
    
    # Since the ray will be starting from within the sphere (we assume the 
    # heliocentric_distance is at least 1 AU), one of the solutions should be positive
    # and the other negative. We only use the positive one.
    dist = (-b + np.sqrt(disc))/(2 * a)

    answer = SkyCoord(
        ra=los_earth_obj.ra, # this was coord.ra
        dec=los_earth_obj.dec, # this was coord.dec
        distance=dist * u.AU,
        obstime=obstime,
        obsgeoloc=loc,
        frame="gcrs",
    ).transform_to(ICRS())

    return answer, dist

def correct_parallax3(coord, obstime, point_on_earth, heliocentric_distance, geocentric_distance=None):
    """This is the implementation that Dino implemented here (Section 3):
    https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb

    Conceptually similar to the scipy minimizer approach, but just uses a static range of distances to search.
    It seems like it's producing reasonable ra/dec values, but I believe there is
    a bug in the distance, because it differes from the scipy minimizer approach
    by a factor of 0-1au. See the heliocentric_reproejc_smanap notebook for a plot.
    """

    loc = (point_on_earth.to_geocentric()) * u.m

    # line of sight from earth to the object,
    # the object has an unknown distance from earth
    los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))

    guess_dists = np.arange(heliocentric_distance-1.02, heliocentric_distance+1.02, 0.0001)
    guesses = GCRS(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=guess_dists*u.AU,
        obstime=obstime,
        obsgeoloc=loc
    ).transform_to(ICRS())

    deltad = np.abs(heliocentric_distance-guesses.distance.value)
    minidx= min(deltad) == deltad
    answer = guesses[minidx]

    # we'll make a new object so that it returns numbers not a list
    res = ICRS(
        ra = answer.ra[0],
        dec = answer.dec[0],
        distance=answer.distance[0]
    )
    
    return res, answer.distance[0].value

def invert_correct_parallax(coord, obstime, point_on_earth, geocentric_distance, heliocentric_distance):
    """Calculate the original ICRS coordinates of a point in EBD space, i.e. a result from `correct_parallax`.

    Parameters
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The EBD coordinate that we want to find the original position of in non parallax corrected space of.
    obstime : `astropy.time.Time` or `string`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    geocentric_distance : `float`
        The distance from Earth to the object (generally a result from `correct_parallax`).
    heliocentric_distance : `float`
        The distance from the solar system barycenter to the object (generally an input for `correct_parallax`).

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in ICRS. corresponding to the
    position in the original observation (before `correct_parallax`).

    References
    ----------
    .. [1] `Jupyter Notebook <https://github.com/maxwest-uw/notebooks/blob/main/uncorrecting_parallax.ipynb>`_
    """
    loc = (point_on_earth.to_geocentric()) * u.m

    icrs_with_dist = ICRS(ra=coord.ra, dec=coord.dec, distance=heliocentric_distance * u.au)

    gcrs_no_dist = icrs_with_dist.transform_to(GCRS(obsgeoloc=loc, obstime=obstime))
    gcrs_with_dist = GCRS(
        ra=gcrs_no_dist.ra, dec=gcrs_no_dist.dec, distance=geocentric_distance, obsgeoloc=loc, obstime=obstime
    )

    original_icrs = gcrs_with_dist.transform_to(ICRS())
    return original_icrs


def fit_barycentric_wcs(
    original_wcs, width, height, heliocentric_distance, obstime, point_on_earth, npoints=10, seed=None
):
    """Given a ICRS WCS and an object's distance from the Sun,
    return a new WCS that has been corrected for parallax motion.

    Parameters
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
    space, i.e. where the points have been corrected for parallax, as well as the average best fit
    geocentric distance of the object.
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


def transform_wcses_to_ebd(
    wcs_list, width, height, heliocentric_distance, obstimes, point_on_earth, npoints=10, seed=None
):
    """Transform a set of WCSes (for instance, a `WorkUnit.per_image_wcs`) into EBD space.

    Parameters
    ----------
    wcs_list : List of `astropy.wcs.WCS`
        The image's WCS.
    width : `int`
        The image's width (typically NAXIS1).
    height : `int`
        The image's height (typically NAXIS2).
    heliocentric_distance : `float`
        The distance of the object from the sun, in AU.
    obstimes : list of `astropy.time.Time`s or `string`s
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
    A list of `astropy.wcs.WCS` objects in "Explicity Barycentric Distance" (EBD)
    space, i.e. where the points have been corrected for parallax, as well as a list of
    the average best fits for geocentric distance of the object.
    """
    transformed_wcses = []
    geocentric_dists = []

    for w, t in zip(wcs_list, obstimes):
        transformed_wcs, geo_dist = fit_barycentric_wcs(
            w, width, height, heliocentric_distance, t, point_on_earth, npoints, seed
        )
        transformed_wcses.append(transformed_wcs)
        geocentric_dists.append(geo_dist)

    return transformed_wcses, geocentric_dists
