import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize


class ResultsVisualizer:
    @staticmethod
    def plot_single_stamp(stamp, axes=None, norm=None):
        """Plot a single stamp image.

        Parameters
        ----------
        stamp : np.array
            The numpy array containing the stamp's pixel values.
        axes : matplotlib.axes.Axes
            The axes on which to draw the figure.
        norm : `astropy.visualization.ImageNormalize` or None
            The normalization to use for the image
        """
        # If there is nothing to plot, skip.
        if stamp is None or stamp.size == 0:
            return

        # If the stamp needs to be reshaped, compute the width and reshape.
        stamp_width = stamp.shape[0]
        if len(stamp.shape) == 1:
            stamp_width = int(math.sqrt(stamp.shape[0]))
        if stamp.size != stamp_width * stamp_width:
            raise ValueError("Expected square stamp, but found {stamp.shape}")

        # If no figure was given, create a new one.
        if axes is None:
            fig = plt.figure()
            axes = fig.add_axes([0, 0, 1, 1])

        # Plot the stamp.
        axes.imshow(stamp.reshape(stamp_width, stamp_width), norm=norm)

    @staticmethod
    def plot_stamps(stamps, fig=None, columns=3, normalize=False):
        """Plot multiple stamps in a grid.

        Parameters
        ----------
        stamps : a list of np.arrays
            The list of numpy array containing each stamp's pixel values.
        fig : matplotlib.figure
            The figure to use or None to create a new figure.
        columns : int
            The number of columns to use.
        normalize: `bool`
            Normalize the image using Astropy's `ImageNormalize`
            using `ZScaleInterval` and `AsinhStretch`. `False` by
            default.
        """
        num_stamps = len(stamps)
        num_rows = math.ceil(num_stamps / columns)

        # Create a new figure if needed.
        if fig is None:
            fig = plt.figure()

        for i, stamp in enumerate(stamps):
            ax = fig.add_subplot(num_rows, columns, i + 1)
            ax.set_title(f"Time {i}")
            norm = None
            if normalize:
                norm = ImageNormalize(stamp, interval=ZScaleInterval(), stretch=AsinhStretch())
            ResultsVisualizer.plot_single_stamp(stamp, axes=ax, norm=norm)

    @staticmethod
    def plot_time_series(values, times=None, axes=None, indices=None, title=None):
        """Plot a time series on the graph.

        Parameters
        ----------
        values : a list or np.array of floats
            The array of the values at each time.
        times : a list or np.array of floats
            The array of the time stamps. If None then uses equally spaced points.
        indices : a list np.array of of ints
            The array of which indices are valid. If None then all indices are
            considered valid.
        axes : matplotlib.axes.Axes
            The axes on which to draw the figure.
        title : string
            The title string to use.
        """
        # Do a shallow copy to allow us to transform list into nparray.
        y_values = np.copy(values)

        # If no axes were given, create a new figure.
        if axes is None:
            fig = plt.figure()
            axes = fig.add_axes([0, 0, 1, 1])

        # If no valid indices are given, use them all.
        all_indices = np.linspace(0, len(values) - 1, len(values), dtype=int)
        if indices is None:
            to_use = all_indices
        else:
            to_use = np.copy(indices)
        invalid_indices = np.setdiff1d(all_indices, to_use)

        # If the times are not given, then use linear spacing.
        if times is None:
            x_values = all_indices
        else:
            x_values = np.copy(times)

        # Plot the data with the curve in blue, the valid points as blue dots,
        # and the invalid indices as smaller red dots.
        axes.plot(x_values, y_values, "b")
        axes.plot(x_values[to_use], y_values[to_use], "b.", ms=25)
        axes.plot(x_values[invalid_indices], y_values[invalid_indices], "r.", ms=10)

        if title is not None:
            axes.set_title(title)
