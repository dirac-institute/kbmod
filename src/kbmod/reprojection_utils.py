import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ICRS
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
