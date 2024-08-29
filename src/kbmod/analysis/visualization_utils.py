from astropy.time import Time


def mjd_to_day(mjd):
    """Takes an mjd and converts it into a day in
    calendar date format.

    Parameters
    ----------
    mjd : `float`
        mjd format date.

    Returns
    ----------
    A `str` with a calendar date, in the format YYYY-MM-DD.
    e.g., mjd=60000 -> '2023-02-25'
    """
    return str(Time(mjd, format="mjd").to_value("datetime")).split()[0]
