class Trajectory:
    """A class to store the (linear) trajectory of a moving object.

    Attributes
    ----------
    x : `int`
        The initial x-coordinate of the trajectory (in pixels).
    y : `int`
        The initial y-coordinate of the trajectory (in pixels).
    vx : `float`
        The x-velocity of the trajectory (in pixels per day).
    vy : `float`
        The y-velocity of the trajectory (in pixels per day).
    flux : `float`
        The flux of the trajectory.
    lh : `float`
        The likelihood of the trajectory.
    obs_count : `int`
        The number of valid observations in the trajectory.
    """

    def __init__(self, x=0, y=0, vx=0.0, vy=0.0, flux=0.0, lh=0.0, obs_count=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.flux = flux
        self.lh = lh
        self.obs_count = obs_count

    def __str__(self):
        return f"Trajectory: x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}, lh={self.lh}, obs_count={self.obs_count}"

    def __eq__(self, other):
        return (
            self.x == other.x
            and self.y == other.y
            and self.vx == other.vx
            and self.vy == other.vy
            and self.flux == other.flux
            and self.lh == other.lh
            and self.obs_count == other.obs_count
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.lh < other.lh

    def __le__(self, other):
        return self.lh <= other.lh

    def __gt__(self, other):
        return self.lh > other.lh

    def __ge__(self, other):
        return self.lh >= other.lh

    def get_xy_pos(self, time, centered=True):
        """Get the x-position of the trajectory at a given time.

        Parameters
        ----------
        time : `float`, `np.ndarray`, or `jax.numpy.ndarray`
            The time(s) in days.
        centered : `bool`, optional
            If True, the position is returned at the center of the pixel.

        Returns
        -------
        x_pos : `float`, `np.ndarray`, or `jax.numpy.ndarray`
            The x-position of the trajectory.
        y_pos : `float`, `np.ndarray`, or `jax.numpy.ndarray`
            The y-position of the trajectory.
        """
        x_pos = self.x + time * self.vx
        y_pos = self.y + time * self.vy
        if centered:
            x_pos += 0.5
            y_pos += 0.5
        return x_pos, y_pos
