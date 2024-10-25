from astropy.coordinates import EarthLocation
from astropy.table import Table


from kbmod.reproject_utils import correct_parallax_geometrically_vectorized
from kbmod.search import Trajectory
from kbmod.util_functions import get_matched_obstimes


def load_fakes_data(filename):
    """Load the fakes data from a CSV file.

    Parameters
    ----------
    filename : `str`
        The file with the fakes data in it.

    Returns
    -------
    fakes : `astropy.table.Table`
        The table with the fakes data.
    """
    fakes = Table.read(filename, format="csv")
    fakes.sort(["ORBITID", "mjd_mid"])  # Sort by the unique ORBITID for each fake and then observation time
    return fakes


def add_parallax_corrected_ra_dec(fake_table, distance, observatory=None):
    """Add RA and DEC for the parallax correct observation at a given distance.

    Parameters
    ----------
    fake_table : `astropy.table.Table`
        The table with the fakes data.
    distance: `float` or list-like
        The guess distance to the object from the Sun in AU.
    observatory : `str` or `None`, optional
        The name of the observatory to use during parallax correction. If not specified, the
        code uses the center of the earth.

    Returns
    -------
    fake_table : `astropy.table.Table`
        The table with the fakes data.
    """
    if distance is None or distance < 1.0:
        raise ValueError(f"Invalid heliocentric distance {distance}. Value must be >= 1.0.")

    new_ra_col = f"RA_{guess_dist}"
    new_dec_col = f"Dec_{guess_dist}"
    if new_ra_col in fake_table.colnames and new_dec_col in fake_table.colnames:
        # Nothing to do. The data already exists.
        return fake_table

    # Correct all of the observations for parallax.
    point_on_earth = None if observatory is None else EarthLocation.of_site(observatory)
    new_fakes = correct_parallax_geometrically_vectorized(
        fake_table["RA"],
        fake_table["DEC"],
        fake_table["mjd_mid"],
        distance,
        point_on_earth=point_on_earth,
        return_geo_dists=False,
    )
    fake_table[new_ra_col] = new_fakes.ra.deg
    fake_table[new_dec_col] = new_fakes.dec.deg

    return fake_table


def compute_pixels_of_fake(
    fake_table,
    fake_orbit_id,
    workunit,
    distance=None,
):
    """Compute the pixel coordinates of the fake in each image in which it occurs.

    Parameters
    ----------
    fake_table : `astropy.table.Table`
        The table with the fakes data.
    fake_orbit_id : `str`
        The orbit ID of the fake to analyze.
    workunit : `WorkUnit`
        The WorkUnit to use for analysis. Needed for the WCS.
    distance : `float` or `None`, optional
        The heliocentric guess distance to use. If None then does not perform parallax correction.
        Default: None

    Returns
    -------
    x_pos, y_pos: `numpy.ndarray`
        Arrays of the X and Y pixel positions respectively.
    """
    # Extract the single object to fit.
    match = fake_table[fake_table["ORBITID"] == fake_orbit_id]
    if len(match) <= 1:
        raise ValueError("Insufficient matches to fit trajectory. Require >= 2 observations.")
    match.sort("mjd_mid")

    # Get the parallax corrected information if needed.
    if distance is not None:
        ra_col = f"RA_{guess_dist}"
        dec_col = f"Dec_{guess_dist}"
        if new_ra_col not in match.colnames or new_dec_col not in match.colnames:
            match = add_parallax_corrected_ra_dec(match, distance)
    else:
        ra_col = "RA"
        dec_col = "Dec"

    x_pos, y_pos = workunit.get_pixel_coordinates(match[ra_col], match[dec_col], match["mjd_mid"])
    return x_pos, y_pos
