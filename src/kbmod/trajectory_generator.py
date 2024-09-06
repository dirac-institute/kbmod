import abc
import logging
import random
import traceback

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.search import Trajectory


logger = logging.getLogger(__name__)


def create_trajectory_generator(config):
    """Create a TrajectoryGenerator object given a dictionary
    of configuration options. The generator class is specified by
    the 'name' entry, which must exist and match the class name of one
    of the subclasses of ``TrajectoryGenerator``.

    Parameters
    ----------
    config : `dict` or `SearchConfiguration`
        The dictionary of generator parameters.

    Returns
    -------
    gen : `TrajectoryGenerator`
        A TrajectoryGenerator object.

    Raises
    ------
    Raises a ``KeyError`` if the name entry is missing or the correct parameters
    do not exist.
    """
    # Check if we are dealing with a top level configuration.
    if type(config) is SearchConfiguration:
        if config["generator_config"] is None:
            # We are dealing with a legacy configuration file.
            gen = KBMODV1SearchConfig(
                v_arr=config["v_arr"],
                ang_arr=config["ang_arr"],
                average_angle=config["average_angle"],
            )
            return gen
        else:
            # We are dealing with a top level configuration.
            config = config["generator_config"]

    if "name" not in config:
        raise KeyError("The trajectory generator configuration must contain a name field.")

    name = config["name"]
    if name not in TrajectoryGenerator.generators:
        raise KeyError("Trajectory generator {name} is undefined.")
    logger.info(f"Creating trajectory generator of type {name}")

    return TrajectoryGenerator.generators[name](**config)


class TrajectoryGenerator(abc.ABC):
    """A base class for defining search strategies by generating
    candidate trajectories for each pixel.

    Implementations of TrajectoryGenerator must:
    1) override generate() to provide new samples,
    2) cannot be infinite
    """
    generators = {}
    """Registry of generator names."""

    @classmethod
    def from_config(cls, config):
        if isinstance(config, SearchConfiguration):
            if config["generator_config"] is None:
                # We are dealing with a legacy configuration file.
                gen = KBMODV1SearchConfig(
                    v_arr=config["v_arr"],
                    ang_arr=config["ang_arr"],
                    average_angle=config["average_angle"],
                )
                return gen
        else:
            # We are dealing with a top level configuration.
            config = config["generator_config"]

        if "name" not in config:
            raise KeyError("The trajectory generator configuration must contain a name field.")

        name = config["name"]
        if name not in TrajectoryGenerator.generators:
            raise KeyError("Trajectory generator {name} is undefined.")
        logger.info(f"Creating trajectory generator of type {name}")

        return cls.generators[name](**config)

    def __init_subclass__(cls, **kwargs):
        # Register all subclasses in a dictionary mapping class name to the
        # class object, so we can programmatically create objects from the names.
        super().__init_subclass__(**kwargs)
        cls.generators[cls.__name__] = cls

    def __enter__(self, *args, **kwargs):
        return self.generate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            traceback.print_tb(exc_tb)
            raise exc_val

    def __next__(self, *args, **kwargs):
        return next(self.generate(*args, **kwargs))

    def __iter__(self, *args, **kwargs):
        return self.generate()

    @abc.abstractmethod
    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            a ``Trajectory`` to test at each pixel.
        """
        raise NotImplementedError()

    def to_table(self):
        """Generate the trajectories and put them into
        an astropy table for analysis.

        Returns
        -------
        results : `astropy.table.Table`
            The different trajectories to try.
        """
        data_dict = {"x": [], "y": [], "vx": [], "vy": []}
        for trj in self:
            data_dict["x"].append(trj.x)
            data_dict["y"].append(trj.y)
            data_dict["vx"].append(trj.vx)
            data_dict["vy"].append(trj.vy)
        return Table(data_dict)


class SingleVelocitySearch(TrajectoryGenerator):
    """Search a single velocity from each pixel."""

    def __init__(self, vx, vy, **kwargs):
        """Create a class SingleVelocitySearch.

        Parameters
        ----------
        vx : `float`
            The velocity in x pixels (pixels per day).
        vy : `float`
            The velocity in y pixels (pixels per day).
        """
        super().__init__(**kwargs)
        self.vx = vx
        self.vy = vy

    def __repr__(self):
        return f"SingleVelocitySearch: vx={self.vx}, vy={self.vy}"

    def __str__(self):
        return f"SingleVelocitySearch: vx={self.vx}, vy={self.vy}"

    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            A ``Trajectory`` to test at each pixel.
        """
        yield Trajectory(vx=self.vx, vy=self.vy)


class VelocityGridSearch(TrajectoryGenerator):
    """Search a grid defined by steps in velocity space."""

    def __init__(self, vx_steps, min_vx, max_vx, vy_steps, min_vy, max_vy, *args, **kwargs):
        """Create a class VelocityGridSearch.

        Parameters
        ----------
        vx_steps : `int`
            The number of velocity steps in the x direction.
        min_vx : `float`
            The minimum velocity in the x-dimension (pixels per day).
        max_vx : `float`
            The maximum velocity in the x-dimension (pixels per day).
        vy_steps : `int`
            The number of velocity steps in the y direction.
        min_vy : `float`
            The minimum velocity in the y-dimension (pixels per day).
        max_vy : `float`
            The maximum velocity in the y-dimension (pixels per day).
        """
        super().__init__(*args, **kwargs)

        if vx_steps < 2 or vy_steps < 2:
            raise ValueError("VelocityGridSearch requires at least 2 steps in each dimension")
        if max_vx < min_vx or max_vy < min_vy:
            raise ValueError("Invalid VelocityGridSearch bounds.")

        self.vx_steps = vx_steps
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.vx_stepsize = (self.max_vx - self.min_vx) / float(self.vx_steps - 1)

        self.vy_steps = vy_steps
        self.min_vy = min_vy
        self.max_vy = max_vy
        self.vy_stepsize = (self.max_vy - self.min_vy) / float(self.vy_steps - 1)

    def __repr__(self):
        return (
            "VelocityGridSearch:"
            f"  vx=[{self.min_vx}, {self.max_vx}], {self.vx_steps}"
            f"  vy=[{self.min_vy}, {self.max_vy}], {self.vy_steps}"
        )

    def __str__(self):
        return (
            "VelocityGridSearch:\n"
            f"    Vel X: [{self.min_vx}, {self.max_vx}] in {self.vx_steps} steps.\n"
            f"    Vel Y: [{self.min_vy}, {self.max_vy}] in {self.vy_steps} steps."
        )

    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            A ``Trajectory`` to test at each pixel.
        """
        for vy_i in range(self.vy_steps):
            for vx_i in range(self.vx_steps):
                vx = self.min_vx + vx_i * self.vx_stepsize
                vy = self.min_vy + vy_i * self.vy_stepsize
                yield Trajectory(vx=vx, vy=vy)


class EclipticSearch(TrajectoryGenerator):
    """Search velocities and angles centered on the ecliptic.

    Trajectories are produced by pairing each velocity magnitude (speed) with
    each angle and then calculating the velocity components of the velocity
    vector. The created combinations are the equivalent of:

        >>> velocities = [1, 2]
        >>> angles = [3, 4]
        >>> [(v, a) for a in angles for v in velocities]
        [(1, 3), (2, 3), (1, 4), (2, 4)]

    Parameters
    ----------
    v_range : `Quantity[float, float, float]` or `tuple[float, float, float]`
        Velocity range, as ``[start, stop, step]``, to search. Start and end
        velocities are included in the search. If provided as a list or a tuple, the
        expected units are pixels/day.
    angle_range: `Quantity[float, float, float]` or `tuple[float, float, float]`
        Angle range, as ``[start, stop, step]``, to search. Start and end search
        angle are included in the search. The angles are centered on the
        ``ecliptic_angle``.
    ecliptic_angle: `float` or `None`, optional
        Angle the image closes with the ecliptic. If `None`, then `wcs` must be provided to
        calculate the angle the image is rotated by wrt the ecliptic.
    direction: `str`, optional
        By default the direction is ``prograde``. Setting this to ``retrograde`` returns
        negative search velocities.

    Notes
    -----
    Searching ``[ecliptic-90, ecliptic+90] in retrograde direction is equivalent
    to ``[ecliptic-270, ecliptic-90]``, i.e. ``direction`` wraps the search
    around 360 degrees.

    The velocity is expressed in pixels per day because internally the timestamps
    are handled as MJD timestamps, i.e. fractional days.

    Examples
    --------
    >>> from kbmod.trajectory_generators import EclipticSearch
    >>> trajectories = list(EclipticSearch([0, 1, 1], [0, 90, 90], ecliptic_angle=0))
    >>> for t in trajectories:
    ...     print(f"{t.vx:5.2}{t.vy:5.2}")
    ...
    0.0  0.0
    1.0  0.0
    0.0  0.0
    6.1e-17  1.0

    or as a context manager:

    >>> with EclipticSearch([0, 1, 1], [0, 90, 90], ecliptic_angle=45) as gen:
    ...     for t in gen:
    ...         print(f"{t.vx:.2} {t.vy:.2}")
    ...
    0.0 0.0
    0.71 0.71
    -0.0 0.0
    -0.71 0.71
    """
    def __init__(self, v_range, angle_range, wcs=None, ecliptic_angle=None, direction="prograde", **kwargs):
        if isinstance(v_range, u.Quantity):
            v_range = v_range.to(u.pixel/u.day).value
        self.vmin, self.vmax, self.vstep = np.array(v_range)
        # make end inclusive
        self.vmax += self.vstep/4

        if ecliptic_angle is None and wcs is None:
            raise ValueError("Either the ecliptic angle or an WCS must be provided.")
        self.ecliptic_angle = self.calc_ecliptic_angle(wcs) if ecliptic_angle is None else np.deg2rad(ecliptic_angle)

        if isinstance(angle_range, u.Quantity):
            v_range = angle_range.to(u.deg).value
        self.amin, self.amax, self.astep = np.deg2rad(angle_range)
        self.amax += self.astep/4

        self.direction = direction

    def calc_ecliptic_angle(self, wcs, center_pixel=(1000, 2000), step=12):
        """Projects an unit-vector parallel with the ecliptic onto the image
        and calculates the angle of the projected unit-vector in the pixel space.

        Parameters
        ----------
        wcs : ``astropy.wcs.WCS``
            World Coordinate System object.
        center_pixel : tuple, array-like
            Pixel coordinates of image center.
        step : ``float`` or ``int``
            Size of step, in arcseconds, used to find the pixel coordinates of
            the second pixel in the image parallel to the ecliptic.

        Returns
        -------
        ecliptic_angle : ``float``
            Angle the projected unit-vector parallel to the eclipticc closes
            with the image axes. Used to transform the specified search angles,
            with respect to the ecliptic, to search angles within the image.

        Note
        ----
        It is not necessary to calculate this angle for each image in an
        image set if they have all been warped to a common WCS.
        """
        # pick a starting pixel approximately near the center of the image
        # convert it to ecliptic coordinates
        start_pixel = np.array(center_pixel)
        start_pixel_coord = SkyCoord.from_pixel(start_pixel[0], start_pixel[1], wcs)
        start_ecliptic_coord = start_pixel_coord.geocentrictrueecliptic

        # pick a guess pixel by moving parallel to the ecliptic
        # convert it to pixel coordinates for the given WCS
        guess_ecliptic_coord = SkyCoord(
            start_ecliptic_coord.lon + step * u.arcsec,
            start_ecliptic_coord.lat,
            frame="geocentrictrueecliptic",
        )
        guess_pixel_coord = guess_ecliptic_coord.to_pixel(wcs)

        # calculate the distance, in pixel coordinates, between the guess and
        # the start pixel. Calculate the angle that represents in the image.
        x_dist, y_dist = np.array(guess_pixel_coord) - start_pixel
        return np.arctan2(y_dist, x_dist)

    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            A ``Trajectory`` to test at each pixel.
        """
        v = np.arange(self.vmin, self.vmax, self.vstep)
        theta = np.arange(self.ecliptic_angle+self.amin, self.ecliptic_angle+self.amax, self.astep)

        sign = 1 if self.direction == "prograde" else -1
        vx = sign*v*np.cos(theta)[:, np.newaxis]
        vy = sign*v*np.sin(theta)[:, np.newaxis]

        for vx, vy in zip(vx.ravel(), vy.ravel()):
            yield Trajectory(vx=vx, vy=vy)


class KBMODV1Search(TrajectoryGenerator):
    """Search a grid defined by velocities and angles."""

    def __init__(self, vel_steps, min_vel, max_vel, ang_steps, min_ang, max_ang, *args, **kwargs):
        """Create a class KBMODV1Search.

        Parameters
        ----------
        vel_steps : `int`
            The number of velocity steps.
        min_vel : `float`
            The minimum velocity magnitude (in pixels per day)
        max_vel : `float`
            The maximum velocity magnitude (in pixels per day)
        ang_steps : `int`
            The number of angle steps.
        min_ang : `float`
            The minimum angle (in radians)
        max_ang : `float`
            The maximum angle (in radians)
        """
        super().__init__(*args, **kwargs)

        if vel_steps < 1 or ang_steps < 1:
            raise ValueError("KBMODV1Search requires at least 1 step in each dimension")
        if max_vel < min_vel or max_ang < min_ang:
            raise ValueError("Invalid KBMODV1Search bounds.")

        self.vel_steps = vel_steps
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.vel_stepsize = (self.max_vel - self.min_vel) / float(self.vel_steps)

        self.ang_steps = ang_steps
        self.min_ang = min_ang
        self.max_ang = max_ang
        self.ang_stepsize = (self.max_ang - self.min_ang) / float(self.ang_steps)

    def __repr__(self):
        return (
            "KBMODV1Search:"
            f" v=[{self.min_vel}, {self.max_vel}), {self.vel_steps}"
            f" a=[{self.min_ang}, {self.max_ang}), {self.ang_steps}"
        )

    def __str__(self):
        return (
            "KBMODV1Search:\n"
            f"    Vel: [{self.min_vel}, {self.max_vel}) in {self.vel_steps} steps.\n"
            f"    Ang: [{self.min_ang}, {self.max_ang}) in {self.ang_steps} steps."
        )

    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            A ``Trajectory`` to test at each pixel.
        """
        for ang_i in range(self.ang_steps):
            for vel_i in range(self.vel_steps):
                curr_ang = self.min_ang + ang_i * self.ang_stepsize
                curr_vel = self.min_vel + vel_i * self.vel_stepsize

                vx = np.cos(curr_ang) * curr_vel
                vy = np.sin(curr_ang) * curr_vel

                yield Trajectory(vx=vx, vy=vy)


class KBMODV1SearchConfig(KBMODV1Search):
    """Search a grid defined by velocities and angles in the format of the configuration file."""

    def __init__(self, v_arr, ang_arr, average_angle, *args, **kwargs):
        """Create a class KBMODV1SearchConfig.

        Parameters
        ----------
        v_arr : `list`
            A triplet of the minimum velocity to use (in pixels per day), and the maximum velocity
            magnitude (in pixels per day), and the number of velocity steps to try.
        ang_arr : `list`
            A triplet of the minimum angle offset (in radians), and the maximum angle offset
            (in radians), and the number of angles to try.
        average_angle : `float`
            The central angle to search around. Should align with the ecliptic in most cases.
        """
        if len(v_arr) != 3:
            raise ValueError("KBMODV1SearchConfig requires v_arr to be length 3")
        if len(ang_arr) != 3:
            raise ValueError("KBMODV1SearchConfig requires ang_arr to be length 3")
        if average_angle is None:
            raise ValueError("KBMODV1SearchConfig requires a valid average_angle.")

        ang_min = average_angle - ang_arr[0]
        ang_max = average_angle + ang_arr[1]
        super().__init__(v_arr[2], v_arr[0], v_arr[1], ang_arr[2], ang_min, ang_max, *args, **kwargs)


class RandomVelocitySearch(TrajectoryGenerator):
    """Search a grid defined by min/max bounds on pixel velocities."""

    def __init__(self, min_vx, max_vx, min_vy, max_vy, max_samples=1_000_000, *args, **kwargs):
        """Create a class KBMODV1Search.

        Parameters
        ----------
        min_vx : `float`
            The minimum velocity magnitude (in pixels per day)
        max_vx : `float`
            The maximum velocity magnitude (in pixels per day)
        min_vy : `float`
            The minimum velocity magnitude (in pixels per day)
        max_vy : `float`
            The maximum velocity magnitude (in pixels per day)
        max_samples : `int`
            The maximum number of samples to generate. Used to avoid
            infinite loops in KBMOD code.
        """
        super().__init__(*args, **kwargs)
        if max_vx < min_vx or max_vy < min_vy:
            raise ValueError("Invalid RandomVelocitySearch bounds.")
        if max_samples <= 0:
            raise ValueError(f"Invalid maximum samples.")

        self.min_vx = min_vx
        self.max_vx = max_vx
        self.min_vy = min_vy
        self.max_vy = max_vy
        self.samples_left = max_samples

    def __repr__(self):
        return (
            "RandomVelocitySearch:"
            f" vx=[{self.min_vx}, {self.max_vx}]"
            f" vy=[{self.min_vy}, {self.max_vy}]"
        )

    def __str__(self):
        return self.__repr__()

    def reset_sample_count(self, max_samples):
        """Reset the counter of samples left.

        Parameters
        ----------
        max_samples : `int`
            The maximum number of samples to generate.
        """
        if max_samples <= 0:
            raise ValueError(f"Invalid maximum samples.")
        self.samples_left = max_samples

    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            a ``Trajectory`` to test at each pixel.
        """
        while self.samples_left > 0:
            self.samples_left -= 1
            vx = self.min_vx + random.random() * (self.max_vx - self.min_vx)
            vy = self.min_vy + random.random() * (self.max_vy - self.min_vy)
            yield Trajectory(vx=vx, vy=vy)
