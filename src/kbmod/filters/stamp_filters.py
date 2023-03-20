import abc

from kbmod.filters.base_filter import *
from kbmod.search import *
from kbmod.result_list import *


class StampPeakFilter(Filter):
    """A filter on how far the stamp's peak is from the center."""

    def __init__(self, stamp_radius, x_thresh, y_thresh, *args, **kwargs):
        """Create a StampPeakFilter.

        Parameters
        ----------
        stamp_radius : ``int``
            The radius of a stamp.
        x_thresh : ``float``
            The number of pixels of offset in the x-direction for filtering.
        y_thresh : ``float``
            The number of pixels of offset in the y-direction for filtering.
        """
        super().__init__(*args, **kwargs)

        self.stamp_radius = stamp_radius
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"PeakOffset_{self.x_thresh}_{self.y_thresh}"

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        the offset of the stamp's peak.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        # Filter any row without a stamp.
        if row.stamp is None:
            return False

        # Check the stamp's width is correct.
        width = 2 * self.stamp_radius + 1
        if len(row.stamp) != width * width:
            raise ValueError("Expected stamp of size {width} by {width}.")

        # Find the peack in the image.
        stamp = row.stamp.reshape([width, width])
        peak_pos = raw_image(stamp).find_peak(True)
        return (
            abs(peak_pos.x - self.stamp_radius) < self.x_thresh
            and abs(peak_pos.y - self.stamp_radius) < self.y_thresh
        )
