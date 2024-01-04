import numpy as np
import pyoorb as oo

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


class PyoorbOrbit(object):
    """A helper class for pyoorb that wraps functionality.

    Uses an orbit in a fixed representation (Cartesian and UTC) at a fixed epoch (2000)
    to simplify predictions and avoid conversions.
    """

    _init_run = False

    def __init__(self, cart_elements):
        """Create an orbit from cartesian elements.

        Attributes
        ----------
        cart_elements : `numpy.ndarray`
            An array of orbit elements with the following values.
            0 - The integer object id
            1 - x
            2 - y
            3 - z
            4 - dx
            5 - dy
            6 - dz
            7 - orbital element type (must be 1 for Cartesian)
            8 - Epoch in MJD (must be 51544.5 for MJD2000)
            9 - time scale type (must be 1 for UTC)
            10 - target absolute magnitude
            12 - target slope parameter
        """
        if len(cart_elements) != 12:
            raise ValueError(f"Wrong number of elements: Expected 12. Found {len(self.cart_elements)})")
        if cart_elements[7] != 1:
            raise ValueError(f"Non-cartesian coordinates provided {self.cart_elements[7]}")
        if cart_elements[8] != 51544.5:
            raise ValueError(f"Invalid epoch used {self.cart_elements[8]}")
        if cart_elements[9] != 1:
            raise ValueError(f"Invalid time scale type {self.cart_elements[9]}")
        self.cart_elements = cart_elements
        self.orb_array = np.array([cart_elements], dtype=np.double, order="F")

    @classmethod
    def _safe_initialize(cls):
        """Initialize the pyoorb library. This needs to be done exactly
        once for ALL objects.
        """
        if not PyoorbOrbit._init_run:
            if oo.pyoorb.oorb_init() != 0:
                raise Exception("Unable to initialize pyoorb")
            PyoorbOrbit._init_run = True

    @classmethod
    def from_kepler_elements(cls, a, e, i, longnode, argper, mean_anomaly, abs_mag=10.0, slope_g=0.0):
        """Create an orbit from the Keplerian elements."""
        cls._safe_initialize()

        orbits_kepler = np.array(
            [
                [
                    0,  # ID Number (unused)
                    a,
                    e,
                    i,
                    longnode,
                    argper,
                    mean_anomaly,
                    3,  # Element types = Keplerian
                    51544.5,  # epoch = MJD2000
                    1,  # time scale = UTC
                    abs_mag,
                    slope_g,
                ]
            ],
            dtype=np.double,
            order="F",
        )

        # Convert the orbit to cartesian coordinates (Note in_element_type is the code for the
        # output type).
        orbits_cart, err = oo.pyoorb.oorb_element_transformation(in_orbits=orbits_kepler, in_element_type=1)
        if err != 0:
            raise Exception(f"Error in transformation {err}")

        # Create a single
        return PyoorbOrbit(orbits_cart[0])

    @classmethod
    def from_cartesian_elements(cls, x, y, z, dx, dy, dz, abs_mag=10.0, slope_g=0.0):
        """Create an orbit from the Cartesian elements."""
        orbit_cart = np.array(
            [
                0,  # ID Number (unused)
                x,
                y,
                z,
                dx,
                dy,
                dz,
                1,  # Element types = Cartesian
                51544.5,  # epoch = MJD2000
                1,  # time scale = UTC
                abs_mag,
                slope_g,
            ],
            dtype=np.double,
            order="F",
        )
        return PyoorbOrbit(orbit_cart)

    def get_ephemerides(self, mjds, obscode="I11"):
        """Compute the object's position at given times.

        Parameters
        ----------
        mjds : `list` or `numpy.ndarray`
            Observation times in MJD.
        obscode : `str`
            The observatory code.

        Returns
        -------
        A list of SkyCoord (one for each time) of the predicted sky positions.
        """
        timescales = [1] * len(mjds)
        ephem_dates = np.array(list(zip(mjds, timescales)), dtype=np.double, order="F")

        ephs, err = oo.pyoorb.oorb_ephemeris_basic(
            in_orbits=self.orb_array, in_dynmodel="N", in_obscode=obscode, in_date_ephems=ephem_dates
        )
        if err != 0:
            raise Exception(f"Error in ephem {err}")

        return [SkyCoord(ephs[0, i, 1], ephs[0, i, 2], unit="deg") for i in range(len(mjds))]

    def get_ephem_pixels(self, wcs, mjds, obscode="I11"):
        """Compute the object's position at different times given the WCS for each time.

        Parameters
        ----------
        wcs : `astropy.wcs.WCS` or `list`
            The WCS(s) to use when computing the pixels. Must be either a single WCS
            or a list of the same length as mjds.
        """
        num_times = len(mjds)
        ephem = self.get_ephemerides(mjds, obscode)

        # Support the wcs as a list of WCS or as a global one to use at all times.
        if type(wcs) is list:
            if len(wcs) != num_times:
                raise ValueError(f"Wrong number of WCS provided. Expected {num_times}. Found {len(wcs)}")
        else:
            wcs = [wcs] * num_times

        # Generate the pixel coordinates from each WCS + (RA, dec).
        results = []
        for i in range(num_times):
            x, y = wcs[i].world_to_pixel(ephem[i])
            results.append((x.item(), y.item()))
        return results
