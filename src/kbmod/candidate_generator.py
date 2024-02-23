import abc
import math

from kbmod.search import Trajectory
from kbmod.trajectory_utils import make_trajectory


class CandidateGenerator(abc.ABC):
    """A base class for defining search strategies by generating
    candidate trajectories for each pixel."""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def debug_string(self):
        """Returns a debug string"""
        pass

    @abc.abstractmethod
    def get_candidate_trajectories(self):
        """Produces a list of candidate trajectories to test.

        Returns
        -------
        candidates : `list`
            The list of candidate trajectories to test.
        """
        pass


class SingleVelocitySearch(CandidateGenerator):
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

    def debug_string(self):
        """Returns a debug string"""
        return f"SingleVelocitySearch: vx={self.vx}, vy={self.vy}"

    def get_candidate_trajectories(self):
        """Produces a list of candidate trajectories to test.

        Returns
        -------
        candidates : `list`
            The list of candidate trajectories to test.
        """
        return [make_trajectory(vx=self.vx, vy=self.vy)]


class VelocityGridSearch(CandidateGenerator):
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

        self.vx_steps = vx_steps
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.vy_steps = vy_steps
        self.min_vy = min_vy
        self.max_vy = max_vy

    def debug_string(self):
        """Returns a debug string"""
        return (
            "VelocityGridSearch:\n"
            f"    Vel X: [{self.min_vx}, {self.max_vx}] in {self.vx_steps} steps.\n"
            f"    Vel Y: [{self.min_vy}, {self.max_vy}] in {self.vy_steps} steps."
        )

    def get_candidate_trajectories(self):
        """Produces a list of candidate trajectories to test.

        Returns
        -------
        candidates : `list`
            The list of candidate trajectories to test.
        """
        vx_stepsize = (self.max_vx - self.min_vx) / float(self.vx_steps - 1)
        vy_stepsize = (self.max_vy - self.min_vy) / float(self.vy_steps - 1)
        candidates = []
        for vy_i in range(self.vy_steps):
            for vx_i in range(self.vx_steps):
                vx = self.min_vx + vx_i * vx_stepsize
                vy = self.min_vy + vy_i * vy_stepsize
                candidates.append(make_trajectory(vx=vx, vy=vy))
        return candidates


class KBMODV1Search(CandidateGenerator):
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

        self.vel_steps = vel_steps
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.ang_steps = ang_steps
        self.min_ang = min_ang
        self.max_ang = max_ang

    def debug_string(self):
        """Returns a debug string"""
        return (
            "KBMODV1Search:\n"
            f"    Vel: [{self.min_vel}, {self.max_vel}) in {self.vel_steps} steps.\n"
            f"    Ang: [{self.min_ang}, {self.max_ang}) in {self.ang_steps} steps."
        )

    def get_candidate_trajectories(self):
        """Produces a list of candidate trajectories to test.

        Returns
        -------
        candidates : `list`
            The list of candidate trajectories to test.
        """
        ang_stepsize = (self.max_ang - self.min_ang) / float(self.ang_steps)
        vel_stepsize = (self.max_vel - self.min_vel) / float(self.vel_steps)
        candidates = []
        for ang_i in range(self.ang_steps):
            for vel_i in range(self.vel_steps):
                curr_ang = self.min_ang + ang_i * ang_stepsize
                curr_vel = self.min_vel + vel_i * vel_stepsize

                vx = math.cos(curr_ang) * curr_vel
                vy = math.sin(curr_ang) * curr_vel

                candidates.append(make_trajectory(vx=vx, vy=vy))
        return candidates
