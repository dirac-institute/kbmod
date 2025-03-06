import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS, ICRS, solar_system_ephemeris, get_body_barycentric
from astropy.time import Time
from astropy.wcs.utils import fit_wcs_from_points


__all__ = [
    "correct_parallax",
    "invert_correct_parallax",
    "fit_barycentric_wcs",
    "transform_wcses_to_ebd",
    "correct_parallax_geometrically_vectorized",
    "invert_correct_parallax_vectorized",
]


def correct_parallax(
    coord,
    obstime,
    point_on_earth,
    heliocentric_distance,
    geocentric_distance=None,
    use_minimizer=False,
    method=None,
    use_bounds=False,
):
    """Calculate the parallax corrected postions for a given object at a given
    time, observation location on Earth, and user defined distance from the Sun.

    By default, this function will use the geometric solution for objects beyond 1au.
    If the distance is less than 1au, the function will use the scipy minimizer
    to find the best geocentric distance.

    To explicitly use the minimizer, set `use_minimizer=True`.

    Parameters
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    heliocentric_distance : `float`
        The guess distance to the object from the Sun in AU.
    geocentric_distance : `float` or `None` (optional)
        If the geocentric distance to be corrected for is already known,
        you can pass it in here. This will avoid the computationally expensive
        minimizer call. In AU.
    use_minimizer : `bool` (optional)
        If True, the minimizer will be used to find the best fit geocentric distance.
        Default is False.
    method : `string` (optional)
        The minimization algorithm to use. Default is None, allow Scipy to choose
        the best method.
    use_bounds : `bool` (optional)
        If True, the minimizer will be bounded heliocentric distance +/- 1.02.
        Default is True.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in
    ICRS, and the best fit geocentric distance (float).

    """
    if use_minimizer or heliocentric_distance < 1.02:
        return correct_parallax_with_minimizer(
            coord, obstime, point_on_earth, heliocentric_distance, geocentric_distance, method, use_bounds
        )
    else:
        return correct_parallax_geometrically(coord, obstime, point_on_earth, heliocentric_distance)


def correct_parallax_with_minimizer(
    coord,
    obstime,
    point_on_earth,
    heliocentric_distance,
    geocentric_distance=None,
    method=None,
    use_bounds=False,
):
    """Calculate the parallax corrected postions for a given object at a given time and distance from Earth.

    Parameters
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    heliocentric_distance : `float`
        The guess distance to the object from the Sun in AU.
    geocentric_distance : `float` or `None` (optional)
        If the geocentric distance to be corrected for is already known,
        you can pass it in here. This will avoid the computationally expensive
        minimizer call. In AU.
    method : `string` (optional)
        The minimization algorithm to use. Default is "Nelder-Mead".
    use_bounds : `bool` (optional)
        If True, the minimizer will be bounded heliocentric distance +/- 1.02.
        Default is True.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in
    ICRS, and the best fit geocentric distance (float).

    References
    ----------
    .. [1] `Jupyter Notebook <https://github.com/DinoBektesevic/region_search_example/blob/main/02_accounting_parallax.ipynb>`_
    """
    loc = (point_on_earth.to_geocentric()) * u.m

    # line of sight from earth to the object,
    # the object has an unknown distance from earth
    los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))

    if geocentric_distance is None:
        # Only import the scipy module if we are going to use it.
        from scipy.optimize import minimize

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
            bounds = [(max(0.0, heliocentric_distance - 1.02), heliocentric_distance + 1.02)]

        fit = minimize(
            cost,
            (heliocentric_distance,),
            method=method,
            bounds=bounds,
        )
        geocentric_distance = fit.x[0]

    # If we are given a barycentric distance less than 1 au the object can end up on the other
    # side of the Earth from the observatory.
    if geocentric_distance <= 0:
        return None, -1

    answer = SkyCoord(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=geocentric_distance * u.AU,
        obstime=obstime,
        obsgeoloc=loc,
        frame="gcrs",
    ).transform_to(ICRS())

    return answer, geocentric_distance


def correct_parallax_geometrically(coord, obstime, point_on_earth, heliocentric_distance):
    """Calculate the parallax corrected postions for a given object at a given time,
    position on Earth, and a hypothetical distance from the Sun.

    This geometric solution is only applicable for objects beyond the 1au. It is
    generally faster than the scipy minimizer approach.

    Attributes
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    heliocentric_distance : `float`
        The guess distance to the object from the Sun in AU.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in
    ICRS, and the best fit geocentric distance (float).
    """

    # Compute the Earth's location in cartesian space centered on the barycenter.
    # Also take into account the point on Earth where the observation was made.
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
    # a sphere centered on the barycenter at (0, 0, 0) with radius = heliocentric_distance
    a = vx * vx + vy * vy + vz * vz
    b = 2 * vx * ex + 2 * vy * ey + 2 * vz * ez
    c = ex * ex + ey * ey + ez * ez - heliocentric_distance * heliocentric_distance
    disc = b * b - 4 * a * c

    if disc < 0:
        # The ray from the observatory never hits a sphere at that distance.
        return None, -1.0

    # A ray can intersect a sphere at 0, 1, or 2 points. We use the closest distance
    # that is not negative (since a negative distance is through the Earth).
    dist1 = (-b + np.sqrt(disc)) / (2 * a)
    dist2 = (-b - np.sqrt(disc)) / (2 * a)
    assert dist1 > dist2

    if dist2 > 0.0:
        dist = dist2
    elif dist1 > 0.0:
        dist = dist1
    else:
        # The barycentric distance puts this object on the wrong side of the Earth.
        return None, -1.0

    answer = SkyCoord(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=dist * u.AU,
        obstime=obstime,
        obsgeoloc=loc,
        frame="gcrs",
    ).transform_to(ICRS())

    return answer, dist


def correct_parallax_geometrically_vectorized(
    ra, dec, mjds, heliocentric_distance, point_on_earth=None, return_geo_dists=True
):
    """Calculate the parallax corrected postions for a given object at a given time,
    position on Earth, and a hypothetical distance from the Sun.

    This geometric solution is only applicable for objects beyond the 1au.
    If the parallax correction failed, that is returned an unphysical result, the
    coordinate distance is set to 0AU.

    Given MJDs do not need to be the same, the returned parallax corrected coordinates
    will be returned in the same order of the given coordinates and times.

    Attributes
    ----------
    ra : `list[float]`
        Right Ascension in decimal degrees
    dec : `list[float]`
        Declination in decimal degrees
    mjds : `list[float]`
        MJD timestamps of the times the ``ra`` and ``dec`` were recorded.
    heliocentric_distance : `float`
        The guess distance to the object from the Sun in AU.
    point_on_earth : `EarthLocation` or `None`, optional
        Observation is returned from the geocenter by default. Provide an
        EarthLocation if you want to also account for the position of the
        observatory. If not provided, assumed to be geocenter.
    return_geo_dists : `bool`, default: `True`
        Return calculated geocentric distances (in AU).

    Returns
    ----------
    parallax_corrected: `astropy.coordinates.SkyCoord`
        `SkyCoord` vector of parallax corrected coordinates.
    geocentric_distances: `list`, optional
        A list of calculated geocentric distances. Returned for
        compatibility.
    """
    # Processing has to be batched over same times since Earth
    # position changes at each time stamp
    unique_obstimes = np.unique(mjds)

    # fetch ephemeris of planet earth
    earth_ephems_x = np.zeros((len(unique_obstimes),))
    earth_ephems_y = np.zeros((len(unique_obstimes),))
    earth_ephems_z = np.zeros((len(unique_obstimes),))
    # figure out what coordinate this is pointing to - center of mass, or?
    # TODO: redo the calls to make use of JPL ephems with all the obstime at once
    # because there could be a lot of timestamps here
    with solar_system_ephemeris.set("de432s"):
        for i, ot in enumerate(Time(unique_obstimes, format="mjd")):
            earth_coords = get_body_barycentric("earth", ot)
            earth_ephems_x[i] = earth_coords.x.to(u.AU).value
            earth_ephems_y[i] = earth_coords.y.to(u.AU).value
            earth_ephems_z[i] = earth_coords.z.to(u.AU).value

    # calculate the ephemeris of a particular location on earth, f.e. CTIO
    location_ephems_x = earth_ephems_x + point_on_earth.x.to(u.AU).value
    location_ephems_y = earth_ephems_y + point_on_earth.y.to(u.AU).value
    location_ephems_z = earth_ephems_z + point_on_earth.z.to(u.AU).value

    # Move the given ICRS coordinates back to GCRS, this change of
    # coordinate origins changes the RA, DEC so that its line of sight
    # now contains the object, which isn't true for ICRS coordinate
    # Copy the cartesian elements of the LOS ray for use later
    point_on_earth_geocentric = point_on_earth.geocentric * u.m
    mjd_times = Time(mjds, format="mjd")
    los_earth_obj = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, obstime=mjd_times).transform_to(
        GCRS(obsgeoloc=point_on_earth_geocentric)
    )
    los_earth_obj_cart = los_earth_obj.cartesian

    # Now that we have the ray elements, re-create the SkyCoords with but with an
    # added distance placeholder, because that alters the underlying coordinate
    # representation to SphericalRepresentation. Avoid data copying
    los_earth_obj = SkyCoord(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=np.zeros((len(ra),)) * u.AU,
        obstime=los_earth_obj.obstime,
        obsgeoloc=los_earth_obj.obsgeoloc,
        frame="gcrs",
        copy=False,
    )

    # for each ephemeris calculate the geocentric distance
    # that matches the given heliocentric distance guess and update
    # our LOS coordinate with these new distances
    for obstime, ext, eyt, ezt in zip(
        unique_obstimes, location_ephems_x, location_ephems_y, location_ephems_z
    ):
        mjd_mask = mjds == obstime
        vx = los_earth_obj_cart.x[mjd_mask].value
        vy = los_earth_obj_cart.y[mjd_mask].value
        vz = los_earth_obj_cart.z[mjd_mask].value

        # Solve the quadratic equation for the ray leaving the earth and intersecting
        # a sphere centered on the barycenter at (0, 0, 0) with radius = heliocentric_distance
        a = vx * vx + vy * vy + vz * vz
        b = 2 * vx * ext + 2 * vy * eyt + 2 * vz * ezt
        c = ext * ext + eyt * eyt + ezt * ezt - heliocentric_distance * heliocentric_distance
        disc = b * b - 4 * a * c

        invalid_result_mask = disc < 0
        disc[disc < 0] = 0

        # Since the ray will be starting from within the sphere (we assume the
        # heliocentric_distance is at least 1 au), one of the solutions should be positive
        # and the other negative. We only use the positive one.
        dist = (-b + np.sqrt(disc)) / (2 * a)
        # we can't set the distance to negative in SkyCOord without a lot of wrangling
        # the distance of 0 should correspond to UnitSphericalRepresentation
        dist[invalid_result_mask] = 0.0
        los_earth_obj.distance[mjd_mask] = dist * u.AU

    # finally, transform the coordinates with the new distance guesses
    # back to ICRS, which will now include the correct parallax correction
    # Don't return geo dists if the user doesn't want them
    if return_geo_dists:
        return los_earth_obj.transform_to(ICRS()), los_earth_obj.distance
    return los_earth_obj.transform_to(ICRS())


def invert_correct_parallax_vectorized(coords, obstimes, point_on_earth=None):
    """Converts a given ICRS coordinate with distance into an ICRS coordinate,
    without accounting for the reflex correction.

    ICRS coordinates corrected for reflex motion of the Earth are ICRS coordinates
    that are aware of the finite distance to the object, and thus able to account
    for it during transformation.
    ICRS coordinates that are reported by observers on Earth, assume that distances
    to all objects are ~infinity, Thus, the "true" ICRS coordinate of an object
    with a finite distance is converted without the appropriate correction for the
    parallax. Note the loose use of "true ICRS coordinate" as it is the "wrong"
    ICRS coordinate for all other objects.
    This function takes an ICRS coordinate capable of accounting for the parallax
    due to the finite distance to the object, and returns an ICRS coordinate as it
    would be reported by an observer from Earth at a given time, if they did not
    know how their distance to the object.

    Parameters
    ----------
    coords : `SkyCoord`
        True coordinates.
    obstimes : `Time` or `list[float]`
        Timestamps of observations of the object in MJD.
    point_on_earth : `EarthLocation` or `None`, optional
        Observation is returned from the geocenter by default. Provide an
        EarthLocation if you want to also account for the position of the
        observatory.

    Returns
    -------
    icrs : `SkyCoord`
        ICRS coordinate of the object as it would be observed from Earth at
        the given timestamp(s) without knowing the distance to the object.
    """
    obstimes = Time(obstimes, format="mjd") if not isinstance(obstimes, Time) else obstimes
    obsgeoloc = point_on_earth.geocentric * u.m
    los = coords.transform_to(GCRS(obstime=obstimes, obsgeoloc=obsgeoloc))
    los_earth_obj = SkyCoord(
        ra=los.ra,
        dec=los.dec,
        obsgeoloc=los.obsgeoloc,
        obstime=los.obstime,
        frame="gcrs",
        copy=False,
    )
    return los_earth_obj.icrs


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
        The distance from Earth to the object in AU (generally a result from `correct_parallax`).
    heliocentric_distance : `float`
        The distance from the solar system barycenter to the object in AU
        (generally an input for `correct_parallax`).

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
    return SkyCoord(ra=original_icrs.ra, dec=original_icrs.dec, unit="deg")


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
        The image's width (typically NAXIS1) in pixels.
    height : `int`
        The image's height (typically NAXIS2) in pixels.
    heliocentric_distance : `float`
        The distance of the object from the sun, in AU.
    obstime : `astropy.time.Time`
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
        The image's width (typically NAXIS1) in pixels.
    height : `int`
        The image's height (typically NAXIS2) in pixels.
    heliocentric_distance : `float`
        The distance of the object from the sun, in AU.
    obstimes : list of `astropy.time.Time`s
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
