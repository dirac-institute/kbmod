"""Basic KBMOD benchmarks.

To manually run the benchmarks use: asv run

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""


from kbmod.search import Trajectory


class TimeSuite:
    """A suite of timing functions."""

    def setup(self):
        pass

    def time_runs(self):
        """Empty placeholder for timing runs."""
        trj = Trajectory()
        trj.x = 0
