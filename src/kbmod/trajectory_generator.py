import abc
import copy
import logging
import math
import random

import astropy.units as u

from astropy.table import Table

from kbmod.configuration import SearchConfiguration
from kbmod.search import Trajectory


logger = logging.getLogger(__name__)


def create_trajectory_generator(config, work_unit=None, **kwargs):
    """Create a TrajectoryGenerator object given a dictionary
    of configuration options. The generator class is specified by
    the 'name' entry, which must exist and match the class name of one
    of the subclasses of ``TrajectoryGenerator``.

    Parameters
    ----------
    config : `dict` or `SearchConfiguration`
        The dictionary of generator parameters.
    work_unit : `WorkUnit`, optional
        A WorkUnit to provide additional information about the data that
        can be used to derive parameters that depend on the input.

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
    logger.info(f"Creating trajectory generator of type {name}")

    # Add any keyword arguments to the params, overriding current values.
    params = copy.deepcopy(config)
    params.update(kwargs)
    logger.debug(str(params))

    return TrajectoryGenerator.generators[name](**params, work_unit=work_unit)


class TrajectoryGenerator(abc.ABC):
    """A base class for defining search strategies by generating
    candidate trajectories for each pixel.

    Implementations of TrajectoryGenerator must:
    1) override generate() to provide new samples,
    2) cannot be infinite
    """

    generators = {}  # A mapping of class name to class object for subclasses.

    def __init__(self, work_unit=None, **kwargs):
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

    def __init__(self, v_arr, ang_arr, average_angle=None, work_unit=None, **kwargs):
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
        work_unit : `WorkUnit`, optional
            A WorkUnit to provide additional information about the data that
            can be used to derive parameters that depend on the input.
        """
        if len(v_arr) != 3:
            raise ValueError("KBMODV1SearchConfig requires v_arr to be length 3")
        if len(ang_arr) != 3:
            raise ValueError("KBMODV1SearchConfig requires ang_arr to be length 3")
        if average_angle is None:
            if work_unit is None:
                raise ValueError(
                    "KBMODV1SearchConfig requires a valid average_angle or a WorkUnit with a WCS."
                )
                average_angle = work_unit.compute_ecliptic_angle()

        ang_min = average_angle - ang_arr[0]
        ang_max = average_angle + ang_arr[1]
        super().__init__(v_arr[2], v_arr[0], v_arr[1], ang_arr[2], ang_min, ang_max, **kwargs)


class EclipticCenteredSearch(TrajectoryGenerator):
    """Search a grid defined by velocities and angles relative to the ecliptic.

    Attributes
    ----------
    ecliptic_angle : `float`
        The angle to use for the ecliptic in the image (in the units defined in ``angle_units``).
    velocities : `list`
        A triplet of the minimum velocity to use (in pixels per day), and the maximum velocity
        magnitude (in pixels per day), and the number of velocity steps to try.
    angles : `list`
        A triplet of the minimum angle offset (in radians), and the maximum angle offset (in radians),
        and the number of angles to try.
    min_ang : `float`
        The minimum search angle in the image (ecliptic_angle + min_ang_offset) in radians.
    max_ang : `float`
        The maximum search angle in the image (ecliptic_angle + max_ang_offset) in radians.
    """

    def __init__(
        self,
        velocities=[0.0, 0.0, 0],
        angles=[0.0, 0.0, 0],
        angle_units="radian",
        velocity_units="pix / d",
        given_ecliptic=None,
        work_unit=None,
        **kwargs,
    ):
        """Create a class EclipticCenteredSearch.

        Parameters
        ----------
        velocities : `list`
            A triplet of the minimum velocity to use (in pixels per day), and the maximum velocity
            magnitude (in pixels per day), and the number of velocity steps to try.
        angles : `list`
            A triplet of the minimum angle offset (in the units defined in ``angle_units``), the maximum
            angle offset (in the units defined in ``angle_units``), and the number of angles to try.
        angle_units : `str`
            The units for the angle.
            Default: 'radian'
        velocity_units : `str`
            The units for the angle.
            Default: 'pix / d'
        given_ecliptic : `float`, optional
            An override for the ecliptic as given in the config (in the units defined in
            ``angle_units``). This angle takes precedence over ``computed_ecliptic``.
        work_unit : `WorkUnit`, optional
            A WorkUnit to provide additional information about the data that
            can be used to derive parameters that depend on the input.
        """
        super().__init__(**kwargs)
        ang_units = u.Unit(angle_units)
        vel_units = u.Unit(velocity_units)

        if given_ecliptic is not None:
            self.ecliptic_angle = (given_ecliptic * ang_units).to(u.rad).value
        elif work_unit is not None:
            # compute_ecliptic_angle() always produces radians.
            self.ecliptic_angle = work_unit.compute_ecliptic_angle()
        else:
            logger.warning("No ecliptic angle provided. Using 0.0.")
            self.ecliptic_angle = 0.0

        if len(angles) != 3:
            raise ValueError("Invalid angles parameter. Expected a length 3 list.")
        if len(velocities) != 3:
            raise ValueError("Invalid velocity parameter. Expected a length 3 list.")
        if velocities[2] < 1 or angles[2] < 1:
            raise ValueError("EclipticCenteredSearch requires at least 1 step in each dimension")
        if velocities[1] < velocities[0]:
            raise ValueError(f"Invalid EclipticCenteredSearch bounds: {velocities}")

        self.velocities = [
            (velocities[0] * vel_units).to(u.pixel / u.day).value,
            (velocities[1] * vel_units).to(u.pixel / u.day).value,
            velocities[2],
        ]
        self.vel_stepsize = (velocities[1] - velocities[0]) / float(velocities[2] - 1)

        # Compute the angle bounds and step size in radians.
        self.angles = [
            (angles[0] * ang_units).to(u.rad).value,
            (angles[1] * ang_units).to(u.rad).value,
            angles[2],
        ]
        self.min_ang = self.ecliptic_angle + self.angles[0]
        self.max_ang = self.ecliptic_angle + self.angles[1]
        self.ang_stepsize = (self.max_ang - self.min_ang) / float(self.angles[2] - 1)

    def __repr__(self):
        return (
            "EclipticSearch:"
            f" v=[{self.velocities[0]}, {self.velocities[1]}], {self.velocities[2]}"
            f" a=[{self.min_ang}, {self.max_ang}], {self.ang_steps}"
        )

    def __str__(self):
        return f"""EclipticSearch:
               Vel: [{self.velocities[0]}, {self.velocities[1]}] in {self.velocities[2]} steps.
               Ang:
                   Ecliptic = {self.ecliptic_angle}
                   Offsets = {self.angles[0]} to {self.angles[1]}
                   [{self.min_ang}, {self.max_ang}] in {self.angles[2]} steps."""

    def generate(self, *args, **kwargs):
        """Produces a single candidate trajectory to test.

        Returns
        -------
        candidate : `Trajectory`
            A ``Trajectory`` to test at each pixel.
        """
        for ang_i in range(self.angles[2]):
            for vel_i in range(self.velocities[2]):
                curr_ang = self.min_ang + ang_i * self.ang_stepsize
                curr_vel = self.velocities[0] + vel_i * self.vel_stepsize

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
