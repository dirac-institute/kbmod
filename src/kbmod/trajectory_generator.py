import abc
import copy
import logging
import math
import random

from astropy.table import Table

from kbmod.configuration import SearchConfiguration
from kbmod.search import Trajectory


def create_trajectory_generator(config, **kwargs):
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
            raise ValueError("Missing generator_config parameter.")
        config = config["generator_config"]

    if "name" not in config:
        raise KeyError("The trajectory generator configuration must contain a name field.")

    name = config["name"]
    if name not in TrajectoryGenerator.generators:
        raise KeyError("Trajectory generator {name} is undefined.")
    logger = logging.getLogger(__name__)
    logger.info(f"Creating trajectory generator of type {name}")

    # Add any keyword arguments to the params, overriding current values.
    params = copy.deepcopy(config)
    params.update(kwargs)
    logger.debug(str(params))

    return TrajectoryGenerator.generators[name](**params)


class TrajectoryGenerator(abc.ABC):
    """A base class for defining search strategies by generating
    candidate trajectories for each pixel.

    Implementations of TrajectoryGenerator must:
    1) override generate() to provide new samples,
    2) cannot be infinite
    """

    generators = {}  # A mapping of class name to class object for subclasses.

    def __init__(self, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        # Register all subclasses in a dictionary mapping class name to the
        # class object, so we can programmatically create objects from the names.
        super().__init_subclass__(**kwargs)
        cls.generators[cls.__name__] = cls

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is None:
            return True

    def __next__(self, *args, **kwargs):
        return next(self.generate(*args, **kwargs))

    def __iter__(self, *args, **kwargs):
        return self.generate()

    def initialize(self, *args, **kwargs):
        """Performs any setup needed for this generator"""
        pass

    def close(self, *args, **kwargs):
        """Performs any cleanup needed for this generator"""
        pass

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

    def __init__(self, vx_steps, min_vx, max_vx, vy_steps, min_vy, max_vy, **kwargs):
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
        super().__init__(**kwargs)

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


class KBMODV1Search(TrajectoryGenerator):
    """Search a grid defined by velocities and angles."""

    def __init__(self, vel_steps, min_vel, max_vel, ang_steps, min_ang, max_ang, **kwargs):
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
        super().__init__(**kwargs)

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

                vx = math.cos(curr_ang) * curr_vel
                vy = math.sin(curr_ang) * curr_vel

                yield Trajectory(vx=vx, vy=vy)


class KBMODV1SearchConfig(KBMODV1Search):
    """Search a grid defined by velocities and angles in the format of the legacy configuration file."""

    def __init__(self, v_arr, ang_arr, average_angle=None, computed_ecliptic_angle=None, **kwargs):
        """Create a class KBMODV1SearchConfig.

        Parameters
        ----------
        v_arr : `list`
            A triplet of the minimum velocity to use (in pixels per day), and the maximum velocity
            magnitude (in pixels per day), and the number of velocity steps to try.
        ang_arr : `list`
            A triplet of the minimum angle offset (in radians), and the maximum angle offset
            (in radians), and the number of angles to try.
        average_angle : `float`, optional
            The central angle to search around. Should align with the ecliptic in most cases.
        computed_ecliptic_angle : `float`, optional
            An override for the computed ecliptic from a WCS (in the units defined in
            ``angle_units``). This parameter is ignored if ``force_ecliptic`` is given.
        """
        if len(v_arr) != 3:
            raise ValueError("KBMODV1SearchConfig requires v_arr to be length 3")
        if len(ang_arr) != 3:
            raise ValueError("KBMODV1SearchConfig requires ang_arr to be length 3")
        if average_angle is None:
            if computed_ecliptic_angle is None:
                raise ValueError(
                    "KBMODV1SearchConfig requires a valid average_angle or computed_ecliptic_angle."
                )
                average_angle = computed_ecliptic_angle

        ang_min = average_angle - ang_arr[0]
        ang_max = average_angle + ang_arr[1]
        super().__init__(v_arr[2], v_arr[0], v_arr[1], ang_arr[2], ang_min, ang_max, **kwargs)


class EclipticSearch(TrajectoryGenerator):
    """Search a grid defined by velocities and angles relative to the ecliptic.

    Attributes
    ----------
    ecliptic_angle : `float`
        The angle to use for the ecliptic in the image (in the units defined in ``angle_units``).
    vel_steps : `int`
        The number of velocity steps.
    min_vel : `float`
        The minimum velocity magnitude (in pixels per day)
    max_vel : `float`
        The maximum velocity magnitude (in pixels per day)
    ang_steps : `int`
        The number of angle steps.
    min_ang_offset : `float`
        The minimum offset of the search angle relative from the ecliptic_angle (in radians).
    max_ang_offset : `float`
        The maximum offset of the search angle relative from the ecliptic_angle (in radians).
    min_ang : `float`
        The minimum search angle in the image (ecliptic_angle + min_ang_offset) in radians.
    max_ang : `float`
        The maximum search angle in the image (ecliptic_angle + max_ang_offset) in radians.
    """

    def __init__(
        self,
        vel_steps=0,
        min_vel=0.0,
        max_vel=0.0,
        ang_steps=0,
        min_ang_offset=0.0,
        max_ang_offset=0.0,
        angle_units="radians",
        force_ecliptic=None,
        computed_ecliptic_angle=None,
        **kwargs,
    ):
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
        min_ang_offset : `float`
            The minimum offset of the search angle relative from the ecliptic_angle
            (in the units defined in ``angle_units``).
        max_ang_offset : `float`
            The maximum offset of the search angle relative from the ecliptic_angle
            (in the units defined in ``angle_units``).
        angle_units : `str`
            The units for the angle. Must be one of radians or degrees.
            Default: 'radians'
        force_ecliptic : `float`, optional
            An override for the ecliptic as given in the config (in the units defined in
            ``angle_units``). This angle takes precedence over ``computed_ecliptic``.
        computed_ecliptic_angle : `float`, optional
            An override for the computed ecliptic from a WCS (in the units defined in
            ``angle_units``). This parameter is ignored if ``force_ecliptic`` is given.
        """
        super().__init__(**kwargs)

        if force_ecliptic is not None:
            ecliptic_angle = force_ecliptic
        elif computed_ecliptic_angle is not None:
            ecliptic_angle = computed_ecliptic_angle
        else:
            logger = logging.getLogger(__name__)
            logger.warning("No ecliptic angle provided. Using 0.0.")
            ecliptic_angle = 0.0

        if vel_steps < 1 or ang_steps < 1:
            raise ValueError("EclipticSearch requires at least 1 step in each dimension")
        if max_vel < min_vel:
            raise ValueError("Invalid EclipticSearch bounds.")

        self.vel_steps = vel_steps
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.vel_stepsize = (self.max_vel - self.min_vel) / float(self.vel_steps - 1)

        # Convert the angles into radians.
        if angle_units[:3] == "rad":
            self.ecliptic_angle = ecliptic_angle
            self.min_ang_offset = min_ang_offset
            self.max_ang_offset = max_ang_offset
        elif angle_units[:3] == "deg":
            deg_to_rad = math.pi / 180.0
            self.ecliptic_angle = deg_to_rad * ecliptic_angle
            self.min_ang_offset = deg_to_rad * min_ang_offset
            self.max_ang_offset = deg_to_rad * max_ang_offset
        else:
            raise ValueError(f"Unknown angular units {angle_units}")

        self.ang_steps = ang_steps
        self.min_ang = self.ecliptic_angle + self.min_ang_offset
        self.max_ang = self.ecliptic_angle + self.max_ang_offset
        self.ang_stepsize = (self.max_ang - self.min_ang) / float(self.ang_steps - 1)

    def __repr__(self):
        return (
            "EclipticSearch:"
            f" v=[{self.min_vel}, {self.max_vel}], {self.vel_steps}"
            f" a=[{self.min_ang}, {self.max_ang}], {self.ang_steps}"
        )

    def __str__(self):
        return f"""EclipticSearch:
               Vel: [{self.min_vel}, {self.max_vel}] in {self.vel_steps} steps.
               Ang:
                   Ecliptic = {self.ecliptic_angle}
                   Offsets = {self.min_ang_offset} to {self.max_ang_offset}
                   [{self.min_ang}, {self.max_ang}] in {self.ang_steps} steps."""

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

                vx = math.cos(curr_ang) * curr_vel
                vy = math.sin(curr_ang) * curr_vel

                yield Trajectory(vx=vx, vy=vy)


class RandomVelocitySearch(TrajectoryGenerator):
    """Search a grid defined by min/max bounds on pixel velocities."""

    def __init__(self, min_vx, max_vx, min_vy, max_vy, max_samples=1_000_000, **kwargs):
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
        super().__init__(**kwargs)
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
