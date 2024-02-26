import math
import matplotlib.pyplot as plt
import numpy as np


class ResultsVisualizer:
    @staticmethod
    def plot_single_stamp(stamp, axes=None):
        """Plot a single stamp image.

        Parameters
        ----------
        stamp : np.array
            The numpy array containing the stamp's pixel values.
        axes : matplotlib.axes.Axes
            The axes on which to draw the figure.
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
        axes.imshow(stamp.reshape(stamp_width, stamp_width))

    @staticmethod
    def plot_stamps(stamps, fig=None, columns=3):
        """Plot multiple stamps in a grid.

        Parameters
        ----------
        stamps : a list of np.arrays
            The list of numpy array containing each stamp's pixel values.
        fig : matplotlib.figure
            The figure to use or None to create a new figure.
        columns : int
            The number of columns to use.
        """
        num_stamps = len(stamps)
        num_rows = math.ceil(num_stamps / columns)

        # Create a new figure if needed.
        if fig is None:
            fig = plt.figure()

        for i in range(num_stamps):
            ax = fig.add_subplot(num_rows, columns, i + 1)
            ResultsVisualizer.plot_single_stamp(stamps[i], axes=ax)
            ax.set_title(f"Time {i}")

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

    @staticmethod
    def plot_result_row(row, times=None, title=None, fig=None):
        """Plot a time series on the graph.

        Parameters
        ----------
        row : ResultRow
            The ResultRow to plot.
        times : a list or np.array of floats
            The array of the time stamps. If None then uses equally spaced points.
        title : string
            The title string to use.
        fig : matplotlib.figure
            The figure to use or None to create a new figure.
        """
        if fig is None:
            fig = plt.figure()

        # Create subfigures on the top and bottom.
        (fig_top, fig_bot) = fig.subfigures(2, 1)

        # In the top subfigure plot the coadded stamp on the left and
        # the light curve on the right.
        (ax_stamp, ax_lc) = fig_top.subplots(1, 2)
        if row.stamp is not None:
            ResultsVisualizer.plot_single_stamp(row.stamp, axes=ax_stamp)
            ax_stamp.set_title("Coadded Stamp")
        else:
            ax_stamp.text(0.5, 0.5, "No Stamp")

        if row.light_curve is not None:
            ResultsVisualizer.plot_time_series(
                row.light_curve,
                times=times,
                indices=row.valid_indices,
                axes=ax_lc,
                title="Lightcurve",
            )
        else:
            ax_lc.text(0.5, 0.5, "No Lightcurve")

        # If there are all_stamps, plot those.
        if row.all_stamps is not None:
            ResultsVisualizer.plot_stamps(row.all_stamps, fig=fig_bot, columns=5)
        else:
            ax = fig_bot.add_axes([0, 0, 1, 1])
            ax.text(0.5, 0.5, "No Individual Stamps")

    @staticmethod
    def plot_starting_pixel_histogram(results, height, width):
        """Plot a histogram of the starting pixels of each found trajectory.

        Parameters
        ----------
        results : `ResultList`
            The results to analyze.
        height : `int`
            The image height in pixels
        width : `int`
            The image width in pixels
        """
        fig, ax = plt.subplots()

        x_vals = results.get_result_values("trajectory.x")
        y_vals = results.get_result_values("trajectory.y")
        _, _, _, img = ax.hist2d(x_vals, y_vals, bins=[height, width])
        fig.colorbar(img, ax=ax)
        ax.set_title("Histogram of Starting Pixel")
