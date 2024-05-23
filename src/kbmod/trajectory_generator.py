import abc
import math
import random

from astropy.table import Table

from kbmod.search import Trajectory


class TrajectoryGenerator(abc.ABC):
    """A base class for defining search strategies by generating
    candidate trajectories for each pixel.

    Implementations of TrajectoryGenerator must:
    1) override generate() to provide new samples,
    2) cannot be infinite
    """

    def __init__(self, *args, **kwargs):
        pass

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

    def __init__(self, vx, vy, *args, **kwargs):
        """Create a class SingleVelocitySearch.

        Parameters
        ----------
        vx : `float`
            The velocity in x pixels (pixels per day).
        vy : `float`
            The velocity in y pixels (pixels per day).
        """
        super().__init__(*args, **kwargs)
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
            The minimum velocity magnitude (in pixels per day)
        max_vx : `float`
            The maximum velocity magnitude (in pixels per day)
        vy_steps : `int`
            The number of velocity steps in the y direction.
        min_vy : `float`
            The minimum velocity magnitude (in pixels per day)
        max_vy : `float`
            The maximum velocity magnitude (in pixels per day)
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

                vx = math.cos(curr_ang) * curr_vel
                vy = math.sin(curr_ang) * curr_vel

                yield Trajectory(vx=vx, vy=vy)


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
