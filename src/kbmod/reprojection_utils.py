import astropy.units as u
import numpy as np
from astropy.coordinates import (
    SkyCoord,
    GCRS,
    ICRS,
    UnitSphericalRepresentation,
)
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.time import Time
from astropy.wcs.utils import fit_wcs_from_points, skycoord_to_pixel

import warnings

__all__ = [
    "correct_parallax",
    "fit_barycentric_wcs",
    "transform_wcses_to_ebd",
    "correct_parallax_geometrically_vectorized",
    "invert_correct_parallax_vectorized",
]


def correct_parallax(
    coord,
    obstime,
    point_on_earth,
    barycentric_distance,
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
    barycentric_distance : `float`
        The guess distance to the object from the solar system's barycenter in AU.
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
        If True, the minimizer will be bounded barycentric distance +/- 1.02.
        Default is False.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in
    ICRS, and the best fit geocentric distance (float).

    """
    new_coord = None
    geo_dist = -1.0

    # Try the geometric solution first for objects beyond 1au and fall back to the minimizer if it fails
    # (or we can't use the geometric solution because the object is too close).
    if not use_minimizer and barycentric_distance > 1.02:
        new_coord, geo_dist = correct_parallax_geometrically(
            coord, obstime, point_on_earth, barycentric_distance
        )
    if new_coord is None or geo_dist < 0.0:
        new_coord, geo_dist = correct_parallax_with_minimizer(
            coord, obstime, point_on_earth, barycentric_distance, geocentric_distance, method, use_bounds
        )

    return new_coord, geo_dist


def correct_parallax_with_minimizer(
    coord,
    obstime,
    point_on_earth,
    barycentric_distance,
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
    barycentric_distance : `float`
        The guess distance to the object from the Sun in AU.
    geocentric_distance : `float` or `None` (optional)
        If the geocentric distance to be corrected for is already known,
        you can pass it in here. This will avoid the computationally expensive
        minimizer call. In AU.
    method : `string` (optional)
        The minimization algorithm to use. Default is "Nelder-Mead".
    use_bounds : `bool` (optional)
        If True, the minimizer will be bounded barycentric distance +/- 1.02.
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
            barycentric_distance
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
            bounds = [(max(0.0, barycentric_distance - 1.02), barycentric_distance + 1.02)]

        fit = minimize(
            cost,
            (barycentric_distance,),
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


def _observer_barycentric_position(obstime, obsgeoloc):
    """Return the observer's barycentric position in AU, shape (3,) or (N, 3).

    This is the vector Astropy itself adds and subtracts in its GCRS<->ICRS
    transformations (``astrom["eb"]``), so the ray-sphere intersection and the
    final coordinate transformation share one observer position.
    """
    return erfa_astrom.get().apcs(GCRS(obstime=obstime, obsgeoloc=obsgeoloc))["eb"]


def correct_parallax_geometrically(coord, obstime, point_on_earth, barycentric_distance):
    """Calculate the parallax corrected postions for a given object at a given time,
    position on Earth, and a hypothetical distance from the Sun.

    This geometric solution is only applicable for objects beyond the 1au. It is
    generally faster than the scipy minimizer approach.

    The returned coordinate lies on the sphere of radius ``barycentric_distance``, so
    `invert_correct_parallax_vectorized` with that distance is its exact inverse.

    Attributes
    ----------
    coord : `astropy.coordinate.SkyCoord`
        The coordinate to be corrected for.
    obstime : `astropy.time.Time`
        The observation time.
    point_on_earth : `astropy.coordinate.EarthLocation`
        The location on Earth of the observation.
    barycentric_distance : `float`
        The guess distance to the object from the Sun in AU.

    Returns
    ----------
    An `astropy.coordinate.SkyCoord` containing the ra and dec of the point in
    ICRS, and the best fit geocentric distance (float).
    """
    loc = (point_on_earth.to_geocentric()) * u.m
    eb = _observer_barycentric_position(obstime, loc)
    los_earth_obj = coord.transform_to(GCRS(obstime=obstime, obsgeoloc=loc))

    # Astropy's GCRS->ICRS removes aberration and light deflection from the GCRS direction
    # before placing the point at the given distance from `eb`, so the intersection must use
    # that de-aberrated (ICRS) ray, not the GCRS one. Start from the input direction, then
    # refine once with the exact direction Astropy places the point along.
    ray = coord.icrs.represent_as(UnitSphericalRepresentation).to_cartesian().xyz.value
    for _ in range(2):
        # Solve |eb + dist * ray| = barycentric_distance (ray is a unit vector).
        b = 2 * np.dot(ray, eb)
        c = np.dot(eb, eb) - barycentric_distance * barycentric_distance
        disc = b * b - 4 * c

        if disc < 0:
            # The ray from the observatory never hits a sphere at that distance.
            return None, -1.0

        # A ray can intersect a sphere at 0, 1, or 2 points. We use the closest distance
        # that is not negative (since a negative distance is through the Earth).
        dist1 = (-b + np.sqrt(disc)) / 2
        dist2 = (-b - np.sqrt(disc)) / 2
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
        ray = answer.cartesian.xyz.to_value(u.AU) - eb
        ray /= np.linalg.norm(ray)

    return answer, dist


def correct_parallax_geometrically_vectorized(
    ra, dec, mjds, barycentric_distance, point_on_earth, return_geo_dists=True
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
    barycentric_distance : `float`
        The guess distance to the object from the Sun in AU.
    point_on_earth : `EarthLocation`
        Observation is returned from the geocenter by default. Provide an
        EarthLocation if you want to also account for the position of the
        observatory.
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
    # Move the given ICRS coordinates to GCRS. The final GCRS->ICRS transformation
    # below removes aberration and light deflection again before placing each point
    # at its distance from the observer's barycentric position `eb`.
    point_on_earth_geocentric = point_on_earth.geocentric * u.m
    mjd_times = Time(mjds, format="mjd")
    eb = np.broadcast_to(
        _observer_barycentric_position(mjd_times, point_on_earth_geocentric), (len(mjd_times), 3)
    )
    los_earth_obj = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, obstime=mjd_times).transform_to(
        GCRS(obsgeoloc=point_on_earth_geocentric)
    )

    # Intersect the sphere along the de-aberrated (ICRS) ray from `eb`. Start from the
    # input direction, then refine once with the exact direction Astropy places each
    # point along.
    ray = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).cartesian.xyz.value.T
    for _ in range(2):
        # Solve |eb + dist * ray| = barycentric_distance (rays are unit vectors).
        b = 2 * np.sum(ray * eb, axis=1)
        c = np.sum(eb * eb, axis=1) - barycentric_distance * barycentric_distance
        disc = b * b - 4 * c

        invalid_result_mask = disc < 0
        disc[invalid_result_mask] = 0

        # A ray can intersect a sphere at 0, 1, or 2 points. We use the closest distance
        # that is not negative (since a negative distance is through the Earth).
        dist1 = (-b + np.sqrt(disc)) / 2
        dist2 = (-b - np.sqrt(disc)) / 2
        dist = np.max([dist1, dist2], axis=0)
        invalid_result_mask = invalid_result_mask | (dist <= 0.0)
        dist[invalid_result_mask] = 0.0

        corrected = SkyCoord(
            ra=los_earth_obj.ra,
            dec=los_earth_obj.dec,
            distance=dist * u.AU,
            obstime=los_earth_obj.obstime,
            obsgeoloc=los_earth_obj.obsgeoloc,
            frame="gcrs",
        ).transform_to(ICRS())
        # Keep the previous ray for invalid results (dist = 0 puts the point at `eb`).
        new_ray = corrected.cartesian.xyz.to_value(u.AU).T - eb
        norm = np.linalg.norm(new_ray, axis=1, keepdims=True)
        ray = np.where(norm > 0, new_ray / np.where(norm > 0, norm, 1.0), ray)

    # Don't return geo dists if the user doesn't want them
    if return_geo_dists:
        return corrected, dist * u.AU
    return corrected


def invert_correct_parallax_vectorized(coords, obstimes, point_on_earth=None):
    """Converts a given ICRS coordinate with barycentric distance into an ICRS coordinate,
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


def fit_barycentric_wcs(
    original_wcs, width, height, barycentric_distance, obstime, point_on_earth, npoints=10, seed=None
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
    barycentric_distance : `float`
        The distance of the object from the solar system's barycenter, in AU.
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
        coord, geo_dist = correct_parallax(coord, obstime, point_on_earth, barycentric_distance)
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
    wcs_list, width, height, barycentric_distance, obstimes, point_on_earth, npoints=10, seed=None
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
    barycentric_distance : `float`
        The distance of the object from the solar system's barycenter, in AU.
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
            w, width, height, barycentric_distance, t, point_on_earth, npoints, seed
        )
        transformed_wcses.append(transformed_wcs)
        geocentric_dists.append(geo_dist)

    return transformed_wcses, geocentric_dists


def image_positions_to_original_icrs(
    image_indices,
    positions,
    reprojected_wcs,
    original_wcses,
    all_times,
    observatory,
    input_format="xy",
    output_format="xy",
    filter_in_frame=True,
    reprojection_frame="original",
    barycentric_distance=None,
    geocentric_distances=None,
    per_image_indices=None,
    image_locations=None,
):
    """Method to transform image positions in EBD reprojected images
    into coordinates in the orignal ICRS frame of reference.

    Parameters
    ----------
    image_indices : `numpy.array`
        The `ImageStackPy` indices to transform coordinates.
    positions : `list` of `astropy.coordinates.SkyCoord`s or `tuple`s
        The positions to be transformed.
    reprojected_wcs : `astropy.wcs.WCS`
        The WCS of the reprojected image in EBD space.
    original_wcses : `list` of `astropy.wcs.WCS`
        The WCSes of the original images in ICRS space.
    all_times : `list` of mjds.
        The observation times of the original images in MJD.
    observatory: `astropy.coordinates.EarthLocation`
        The observatory location.
    input_format : `str`
        The input format for the positions. Either 'xy' or 'radec'.
        If 'xy' is given, positions must be in the format of a
        `tuple` with two float or integer values, like (x, y).
        If 'radec' is given, positions must be in the format of
        a `astropy.coordinates.SkyCoord`.
    output_format : `str`
        The output format for the positions. Either 'xy' or 'radec'.
        If 'xy' is given, positions will be returned in the format of a
        `tuple` with two `int`s, like (x, y).
        If 'radec' is given, positions will be returned in the format of
        a `astropy.coordinates.SkyCoord`.
    filter_in_frame : `bool`
        Whether or not to filter the output based on whether they fit within the
        original `constituent_image` frame. If `True`, only results that fall within
        the bounds of the original WCS will be returned.
    reprojection_frame : `str`
        The frame of reference to use for reprojection. Either 'ebd' or 'original'.
        If 'ebd' is given, barycentric_distance, geoncentric_distances, and
        per_image_indices must be provided.
    barycentric_distance : `float` or `None`
        The guess distance from the solar system barycenter to the "objects" in AU.
    geocentric_distances : `list` of `float`s or `None`
        The geocentric distances to the objects in AU. If `reprojection_frame` is 'ebd',
        this must be provided. If `None`, the function will raise an error.
    per_image_indices : `dict` or `None`
        A dictionary mapping image indices to the indices of the constituent images
        used to track which images have been mosaicked together.
    image_locations : `list` of `tuple`s or `None`
        A list of tuples containing the URI strings of the constituent images
        matched to the positions. If `None`, the function will return only the
        transformed positions with both image indices.

    Returns
    -------
    positions : `list` of `astropy.coordinates.SkyCoord`s or `tuple`s
        The transformed positions. If `filter_in_frame` is true, each
        element of the result list will also be a tuple with the
        URI string of the constituent image matched to the position.
    """
    # input value validation
    if input_format not in ["xy", "radec"]:
        raise ValueError(f"input format must be 'xy' or 'radec' , '{input_format}' provided")
    if input_format == "xy":
        if not all(isinstance(i, tuple) and len(i) == 2 for i in positions):
            raise ValueError("positions in incorrect format for input_format='xy'")
    if input_format == "radec" and not all(isinstance(i, SkyCoord) for i in positions):
        raise ValueError("positions in incorrect format for input_format='radec'")
    if len(positions) != len(image_indices):
        raise ValueError(f"wrong number of inputs, expected {len(image_indices)}, got {len(positions)}")
    if output_format not in ["xy", "radec"]:
        raise ValueError(f"output format must be 'xy' or 'radec' , '{output_format}' provided")
    if reprojection_frame not in ["ebd", "original"]:
        raise ValueError(f"reprojection frame must be 'ebd' or 'original', '{reprojection_frame}' provided")
    if reprojection_frame == "ebd" and any(
        [
            barycentric_distance is None,
            geocentric_distances is None,
            per_image_indices is None,
        ]
    ):
        raise ValueError(
            "barycentric_distance or geocentric_distances must be provided when reprojection_frame is 'ebd'"
        )

    position_reprojected_coords = positions

    # convert to radec if input is xy
    # TODO: I should probably rename ebd_wcs
    if input_format == "xy":
        radec_coords = []
        for pos in positions:
            ebd_wcs = reprojected_wcs
            ra, dec = ebd_wcs.all_pix2world(pos[0], pos[1], 0)
            radec_coords.append(SkyCoord(ra=ra, dec=dec, unit="deg"))
        position_reprojected_coords = radec_coords

    # invert the parallax correction if in ebd space
    original_coords = position_reprojected_coords
    if reprojection_frame == "ebd":
        bary_dist = barycentric_distance
        obstimes = [all_times[i] for i in image_indices]

        # transfer into a combined SkyCoord
        combined_coord = SkyCoord(position_reprojected_coords, distance=bary_dist * u.AU)

        inverted_coords = invert_correct_parallax_vectorized(
            coords=combined_coord, obstimes=obstimes, point_on_earth=observatory
        )
        original_coords = [sc for sc in inverted_coords]

    if output_format == "radec" and not filter_in_frame:
        return original_coords

    # convert coordinates into original pixel positions
    positions = []
    for i in image_indices:
        inds = per_image_indices[i]
        coord = original_coords[i]
        pos = []
        for j in inds:
            con_wcs = original_wcses[j]
            con_image = image_locations[j] if image_locations is not None else (i, j)
            height, width = con_wcs.array_shape
            x, y = skycoord_to_pixel(coord, con_wcs)
            x, y = float(x), float(y)
            if output_format == "xy":
                result_coord = (x, y)
            else:
                result_coord = coord
            to_allow = (y >= 0.0 and y < height and x >= 0 and x < width) or (not filter_in_frame)
            if to_allow:
                pos.append((result_coord, con_image))
        if len(pos) == 0:
            positions.append(None)
        elif len(pos) > 1:
            positions.append(pos)
            if filter_in_frame:
                warnings.warn(
                    f"ambiguous image origin for coordinate {i}, including all potential constituent images.",
                    Warning,
                )
        else:
            positions.append(pos[0])
    return positions
