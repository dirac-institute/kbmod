"""A series of Filter subclasses for processing basic stamp information.

The filters in this file all operate over simple statistics based on the
stamp pixels.
"""

import abc

from kbmod.search import raw_image, KB_NO_DATA
from kbmod.result_list import ResultRow


class BaseStampFilter(abc.ABC):
    """The base class for the various stamp filters.

    Attributes
    ----------
    stamp_radius : ``int``
        The radius of a stamp.
    width : ``int``
        The width of the stamp.
    """

    def __init__(self, stamp_radius, *args, **kwargs):
        """Store data needed for all stamp filters."""
        super().__init__(*args, **kwargs)

        if stamp_radius <= 0:
            raise ValueError(f"Invalid stamp radius {stamp_radius}.")
        self.stamp_radius = stamp_radius
        self.width = 2 * stamp_radius + 1

    def _check_row_valid(self, row: ResultRow) -> bool:
        """Checks whether a stamp is valid for this filter.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether the row is valid.
        """
        # Filter any row without a stamp.
        if row.stamp is None:
            return False

        # Check the stamp's number of elements is correct.
        # This can be as a square stamp or a linear array.
        if row.stamp.size != self.width * self.width:
            return False

        return True


class StampPeakFilter(BaseStampFilter):
    """A filter on how far the stamp's peak is from the center.

    Attributes
    ----------
    stamp_radius : ``int``
        The radius of a stamp.
    x_thresh : ``float``
        The number of pixels of offset in the x-direction for filtering.
    y_thresh : ``float``
        The number of pixels of offset in the y-direction for filtering.
    """

    def __init__(self, stamp_radius, x_thresh, y_thresh, *args, **kwargs):
        """Create a StampPeakFilter."""
        super().__init__(stamp_radius, *args, **kwargs)
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"StampPeakFilter_{self.x_thresh}_{self.y_thresh}"

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
        # Filter rows without a valid stamp.
        if not self._check_row_valid(row):
            return False

        # Find the peak in the image.
        stamp = row.stamp.reshape([self.width, self.width])
        peak_pos = raw_image(stamp).find_peak(True)
        return (
            abs(peak_pos.x - self.stamp_radius) < self.x_thresh
            and abs(peak_pos.y - self.stamp_radius) < self.y_thresh
        )


class StampMomentsFilter(BaseStampFilter):
    """A filter on how well the stamp's moments match that of a Gaussian.

    Finds the moment j, k (called moment_jk) as:
    ``SUM_x SUM_y (x - center_x) ^ j * (y - center_y) ^ k * stamp[x][y]``

    For example moment_10 is the flux weighted average position of
    each stamp pixel relative to a center of zero.

    Attributes
    ----------
    stamp_radius : ``int``
        The radius of a stamp.
    m01_thresh : ``float``
        The threshold for the j=0, k=1 moment.
    m10_thresh : ``float``
        The threshold for the j=1, k=0 moment.
    m11_thresh : ``float``
        The threshold for the j=1, k=1 moment.
    m02_thresh : ``float``
        The threshold for the j=0, k=2 moment.
    m20_thresh : ``float``
        The threshold for the j=2, k=0 moment.
    """

    def __init__(
        self, stamp_radius, m01_thresh, m10_thresh, m11_thresh, m02_thresh, m20_thresh, *args, **kwargs
    ):
        """Create a StampMomentsFilter."""
        super().__init__(stamp_radius, *args, **kwargs)
        self.m01_thresh = m01_thresh
        self.m10_thresh = m10_thresh
        self.m11_thresh = m11_thresh
        self.m02_thresh = m02_thresh
        self.m20_thresh = m20_thresh

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return (
            f"StampMomentsFilter_m01_{self.m01_thresh}_m10_{self.m10_thresh}"
            f"_m11_{self.m11_thresh}_m02_{self.m02_thresh}_m20_{self.m20_thresh}"
        )

    def keep_row(self, row: ResultRow):
        """Determine whether to keep an individual row based on
        how well the stamp's moments match that of a Gaussian.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        # Filter rows without a valid stamp.
        if not self._check_row_valid(row):
            return False

        # Find the peack in the image.
        stamp = row.stamp.reshape([self.width, self.width])
        moments = raw_image(stamp).find_central_moments()
        return (
            (abs(moments.m01) < self.m01_thresh)
            and (abs(moments.m10) < self.m10_thresh)
            and (abs(moments.m11) < self.m11_thresh)
            and (moments.m20 < self.m20_thresh)
            and (moments.m02 < self.m02_thresh)
        )


class StampCenterFilter(BaseStampFilter):
    """A filter on whether the center of the stamp is a local
    maxima and the percentage of the stamp's total flux in this
    pixel.

    Attributes
    ----------
    stamp_radius : ``int``
        The radius of a stamp.
    local_max : ``bool``
        Require the central pixel to be a local maximum.
    flux_thresh : ``float``
        The fraction of the stamp's total flux that needs to be in
        the center pixel [0.0, 1.0].
    """

    def __init__(self, stamp_radius, local_max, flux_thresh, *args, **kwargs):
        """Create a StampCenterFilter."""
        super().__init__(stamp_radius, *args, **kwargs)
        self.local_max = local_max
        self.flux_thresh = flux_thresh

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"StampCenterFilter_{self.local_max}_{self.flux_thresh}"

    def keep_row(self, row: ResultRow):
        """Determine whether the center pixel meets the filtering criteria.

        Parameters
        ----------
        row : ResultRow
            The row to evaluate.

        Returns
        -------
        bool
           An indicator of whether to keep the row.
        """
        # Filter rows without a valid stamp.
        if not self._check_row_valid(row):
            return False

        # Find the value of the center pixel.
        stamp = row.stamp.flatten()
        center_index = self.width * self.stamp_radius + self.stamp_radius
        center_val = stamp[center_index]

        # Find the total flux in the image and check for other local_maxima
        flux_sum = 0.0
        for i in range(self.width * self.width):
            pix_val = stamp[i]
            if pix_val != KB_NO_DATA:
                flux_sum += pix_val
                if i != center_index and self.local_max and (pix_val >= center_val):
                    return False

        # Check the flux percentage.
        if flux_sum == 0.0:
            return False
        return center_val / flux_sum >= self.flux_thresh
