"""A set of helper geometric functions"""

import numpy as np


def ang2unitvec(ra, dec):
    """Convert an angular direction (RA, dec) to a unit vector (x, y, z).

    Parameters
    ----------
    ra : `float` or numpy array
       The right ascension in degrees.
    dec : `float` or numpy array
        The declination in degrees.

    Returns
    -------
    (x, y, z) : `tuple`
        The unit vector as a tuple.
    """
    ra_rad: float = ra * (np.pi / 180.0)
    dec_rad: float = dec * (np.pi / 180.0)

    x = np.cos(ra_rad) * np.cos(dec_rad)
    y = np.sin(ra_rad) * np.cos(dec_rad)
    z = np.sin(dec_rad)

    return (x, y, z)


def unitvec2ang(x, y, z):
    """Convert a unit vector (x, y, z) to an angular direction (RA, dec).

    Parameters
    ----------
    x : `float` or numpy array
    y : `float` or numpy array
    z : `float` or numpy array

    Returns
    -------
    ra : `float` or numpy array
       The right ascension(s) in degrees.
    dec : `float` or numpy array
        The declination(s) in degrees.
    """
    ra = (180.0 / np.pi) * np.arctan2(y, x)
    ra = ra % 360.0  # Only positive values
    dec = (180.0 / np.pi) * np.arcsin(z)
    return ra, dec


def angular_distance(pts1, pts2):
    """Compute the pairwise angular offset of D-dimensional
    points with a common scale and origin.

    Parameters
    ----------
    pts1 : numpy array
        A (K, D) matrix with the K points from the first set.
    pts2 : numpy array
        A (K, D) matrix with the K points from the second set.

    Returns
    -------
    ang_dist : numpy array
        A length K array with the pairwise distances.

    Raises
    ------
    ValueError if the matrices are not the compatible shapes.
    """
    if pts1.shape != pts2.shape:
        raise ValueError("Incompatible shapes {pts1.shape} vs {pts2.shape}.")

    length1 = np.linalg.norm(pts1, axis=1)
    length2 = np.linalg.norm(pts2, axis=1)
    scaled_dot = np.sum(pts1 * pts2, axis=1) / (length1 * length2)
    ang_dist = np.arccos(np.clip(scaled_dot, -1.0, 1.0))
    return ang_dist
