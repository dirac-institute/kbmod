import matplotlib.pyplot as plt
import numpy as np

from kbmod.analysis.plotting import plot_time_series, plot_image
from kbmod.results import Results


class ResultsVisualizer:
    """A class to visualize results from a `Results` object.
    
    Attributes
    ----------
    results : `Results`
        The results data structure containing the analysis results.
    stamp_size : float
        The size of the stamps in inches.
        Default: 2.0
    coadds : list
        A list of coadd stamps present in the results.
    times : `numpy.ndarray`
        The time values for the results.
    idx : `int`
        The index of the current result being analyzed.
    _figure : `matplotlib.figure.Figure`
        The matplotlib figure for displaying results.
    _ax_map : dict[str, `matplotlib.axes.Axes`]
        A mapping of axes names to matplotlib Axes objects.
    """
    def __init__(self, results, stamp_size=2.0):
        self.results = results
        self.idx = 0

        if stamp_size <= 0:
            raise ValueError("stamp_size must be positive.")
        self.stamp_size = stamp_size

        # Extract the time stamps if they are available.
        if results.mjd_mid is not None:
            self.times = results.mjd_mid
        else:
            self.times = np.arange(results.get_num_times())

        # Collect the list of coadds that are present.
        self.coadds = []
        for col_key in results.colnames:
            if col_key.startswith("coadd_"):
                self.coadds.append(col_key)

        self._setup_figure()
        self.plot_all()

    @classmethod
    def from_file(cls, filename, stamp_size=2.0):
        """Create a ResultsVisualizer from a file.

        Parameters
        ----------
        filename : str
            The path to the results file.
        stamp_size : float, optional
            The size of the stamps in inches. Default is 2.0.
        """
        results = Results.from_file(filename, stamp_size=stamp_size)
        return cls(results)

    def _setup_figure(self):
        """Set up the matplotlib figure and axes for visualization."""
        # Compute the width of the figure from the data it contains: 2 inch stats bars
        # and the maximum of a) self.stamp_size inches for each coadded stamp or or 4 inches
        # for each time series curve.
        widths = [2, self.stamp_size, 6]  # Stats, stamps, and data curves.
        total_width = np.sum(widths)

        # Compute the height of the figure from the data it contains: 1 inch nav bar,
        # 4 inches core stats, and space for each row of all stamps.
        coadd_height = self.stamp_size * len(self.coadds)
        curve_height = 6  # 2 inches for each of the three curves.
        core_height = max(coadd_height, curve_height)
        heights = [1, core_height, 1]  # Nav bar, core stats, and placeholder for all stamps.

        if "all_stamps" in self.results.colnames:
            stamps_per_row = np.floor(total_width / self.stamp_size)
            num_stamp_rows = np.ceil(len(self.times) / stamps_per_row)
            heights[2] = num_stamp_rows * self.stamp_size
        total_height = np.sum(heights)

        # Create a nested layout of subfigures.
        self._figure = plt.figure(figsize=(total_width, total_height))
        nav_fig, data_fig, all_stamps_fig = self._figure.subfigures(3, 1, height_ratios=heights, hspace=0.05)
        stat_fig, coadds_fig, curve_fig = data_fig.subfigures(1, 3, width_ratios=widths, wspace=0.1, hspace=0.1)

        # Create the axes for each part of the visualization.
        self._ax_map = {}
        self._ax_map["stats"] = stat_fig.add_subplot(111)
        self._ax_map["nav_bar"] = nav_fig.add_subplot(111)

        psi_ax, phi_ax, lc_ax = curve_fig.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.25})
        self._ax_map["psi_curve"] = psi_ax
        self._ax_map["phi_curve"] = phi_ax
        self._ax_map["lightcurve"] = lc_ax

        for coadd_idx, coadd in enumerate(self.coadds):
            self._ax_map[coadd] = coadds_fig.add_subplot(len(self.coadds), 1, coadd_idx + 1)

        if "all_stamps" in self.results.colnames:
            for idx in range(len(self.times)):
                self._ax_map[f"all_stamps_{idx}"] = all_stamps_fig.add_subplot(
                    num_stamp_rows,
                    stamps_per_row,
                    idx + 1,
                )
        else:
            # Create a single axis for the text saying there are no stamps.
            self._ax_map["all_stamps"] = all_stamps_fig.add_subplot(111)
    
        # Create the navigation bar.
        self._ax_map["nav_bar"].set_axis_off()

    def plot_all(self):
        """Plot all the results in the visualizer."""
        self.plot_stats()
        self.plot_curves()
        self.plot_coadds()
        self.plot_all_stamps()

    def plot_curves(self, psi_axis=None, phi_axis=None, lc_axis=None):
        """Update the time series curves.
        
        Parameters
        ----------
        psi_axis : `matplotlib.axes.Axes`, optional
            The axis to plot the psi curve on. If None, uses the default axis.
        phi_axis : `matplotlib.axes.Axes`, optional
            The axis to plot the phi curve on. If None, uses the default axis.
        lc_axis : `matplotlib.axes.Axes`, optional
            The axis to plot the lightcurve on. If None, uses the default axis.
        """
        if psi_axis is None:
            psi_axis = self._ax_map["psi_curve"]
        if phi_axis is None:
            phi_axis = self._ax_map["phi_curve"]
        if lc_axis is None:
            lc_axis = self._ax_map["lightcurve"]

        # Clear the axes before plotting.
        psi_axis.clear()
        phi_axis.clear()
        lc_axis.clear()

        # If we do not have the time series, then display a message in each box.
        if "psi_curve" not in self.results.colnames or "phi_curve" not in self.results.colnames:
            psi_axis.text(0.1, 0.1, "No PSI data", horizontalalignment='center', verticalalignment='center')
            phi_axis.text(0.1, 0.1, "No PHI data", horizontalalignment='center', verticalalignment='center')
            lc_axis.text(0.1, 0.1, "No Lightcurve data", horizontalalignment='center', verticalalignment='center')
            return

        psi = self.results["psi_curve"][self.idx]
        phi = self.results["phi_curve"][self.idx]

        valid = (phi != 0) & np.isfinite(psi) & np.isfinite(phi)
        if "obs_valid" in self.results.colnames:
            valid = valid & self.results["obs_valid"][self.idx]

        plot_time_series(psi, self.times, indices=valid, ax=psi_axis, title="Psi Values")
        plot_time_series(phi, self.times, indices=valid, ax=phi_axis, title="Phi Values")

        lc = np.full(psi.shape, 0.0)
        lc[valid] = psi[valid] / phi[valid]
        plot_time_series(lc, self.times, indices=valid, ax=lc_axis, title="Lightcurve")        

    def plot_stats(self, stats_axis=None):
        """Update the statistics plot.
        
        Parameters
        ----------
        stats_axis : `matplotlib.axes.Axes`, optional
            The axis to plot the statistics on. If None, uses the default axis.
        """
        if stats_axis is None:
            stats_axis = self._ax_map["stats"]

        # Clear the axis before plotting.
        stats_axis.clear()

        # Extract the statistics data from the row.
        values = extract_results_row_scalars(self.results, self.idx)
        display_str = f"RESULT {self.idx}\n--------------\n"
        for key, value in values.items():
            if key == "uuid":
                display_str += f"\n{key}: ...{value[-8:]}"  # Display last 8 characters of UUID
            elif isinstance(value, float):
                display_str += f"\n{key}: {value:.3f}"
            else:
                display_str += f"\n{key}: {value}"
        stats_axis.text(
            0.1,  # Left-ish
            1.0,  # Top
            display_str,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=12,
        )
        stats_axis.set_axis_off()

    def plot_all_stamps(self, stamps_axis=None):
        """Update the all stamps plot.
        
        Parameters
        ----------
        stamps_axis : `lit[matplotlib.axes.Axes]`, optional
            A list of axes on which to plot each stamp.
        """
        if "all_stamps" not in self.results.colnames:
            # If there are no stamps, display a message.
            if stamps_axis is None:
                stamps_axis = self._ax_map["all_stamps"]
            stamps_axis.clear()
            stamps_axis.text(0.5, 0.5, "Individual stamps not available")
            stamps_axis.set_axis_off()
            return

        curr_stamps = np.asanyarray(self.results["all_stamps"][self.idx])
        num_stamps = curr_stamps.shape[0]
        if stamps_axis is not None and len(stamps_axis) != num_stamps:
            raise ValueError(f"Expected {num_stamps} axes for all_stamps, but got {len(stamps_axis)}.")

        for i in range(num_stamps):
            curr_ax = stamps_axis[i] if stamps_axis is not None else self._ax_map[f"all_stamps_{i}"]
            curr_ax.clear()
            plot_image(
                curr_stamps[i],
                ax=curr_ax,
                title=f"Stamp {i + 1}",
                cmap='gray',
                show_counts=False,
            )
            curr_ax.set_aspect('equal')
            curr_ax.set_axis_off()

    def plot_coadds(self, coadd_axes=None):
        """Update the statistics plot.
        
        Parameters
        ----------
        coadd_axes : `matplotlib.axes.Axes`, optional
            A list of axes on which to plot each stamp.
        """
        if coadd_axes is not None and len(coadd_axes) != len(self.coadds):
            raise ValueError(f"Expected {len(self.coadds)} axes for coadds, but got {len(coadd_axes)}.")

        for coadd_idx, coadd in enumerate(self.coadds):
            curr_ax = coadd_axes[coadd_idx] if coadd_axes is not None else self._ax_map[coadd]
            curr_ax.clear()
            plot_image(
                self.results[coadd][self.idx],
                ax=curr_ax,
                title=coadd,
                cmap='gray',
                show_counts=False,
            )
            curr_ax.set_aspect('equal')
            curr_ax.set_axis_off()

def extract_results_row_scalars(results, idx):
    """Extract the scalar values from a results row.
    
    Parameters
    ----------
    results : `Results`
        The full results data structure.
    idx : `int`
        The integer index of the result row to examine.
    
    Returns
    -------
    values : `dict`
        A mapping of column name to value.
    """
    if idx < 0 or idx >= len(results):
        raise IndexError(f"Index {idx} out of bounds for {len(results)} entries.")
    
    values = {}
    for col_key in results.colnames:
        col_data = results[col_key][idx]
        if np.isscalar(col_data):
            values[col_key] = col_data
    return values