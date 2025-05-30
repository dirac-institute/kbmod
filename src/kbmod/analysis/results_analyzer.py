import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, RadioButtons, TextBox

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
    _controls : dict
        A dictionary mapping control names to their respective widgets.
    """

    _labels = ["Not Classified", "Valid", "Noise", "Unknown"]

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

        # Add a classification column if it does not exist.
        if "user_class" not in self.results.colnames:
            self.results.table["user_class"] = np.full(len(self.results), "Not Classified")

        self._setup_figure()
        self.update_all()

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

    def next_result(self, event=None):
        """Move to the next result in the results set."""
        if self.idx < len(self.results) - 1:
            self.idx += 1
            self.update_all()

    def previous_result(self, event=None):
        """Move to the previous result in the results set."""
        if self.idx > 0:
            self.idx -= 1
            self.update_all()

    def goto_to_id(self, event=None):
        """Go to a specific result by ID."""
        id_str = self._controls["id_box"].text.strip()
        if len(id_str) > 0 and id_str.isdigit():
            id_value = int(id_str)
            if id_value >= 0 or id_value < len(self.results):
                self.idx = id_value
        self.update_all()

    def _setup_figure(self):
        """Set up the matplotlib figure and axes for visualization."""
        # Compute the width of the figure from the data it contains: 2 inch stats bars
        # and the maximum of a) self.stamp_size inches for each coadded stamp or or 4 inches
        # for each time series curve.
        widths = [2, self.stamp_size, 6]  # Stats, stamps, and data curves.
        total_width = np.sum(widths)

        # Compute the height of the figure from the data it contains.
        coadd_height = self.stamp_size * len(self.coadds)
        curve_height = 6  # 2 inches for each of the three curves.
        core_height = max(coadd_height, curve_height)
        heights = [1.0, core_height, 1]  # Nav bar, core stats, and placeholder for all stamps.

        if "all_stamps" in self.results.colnames:
            stamps_per_row = np.floor(total_width / self.stamp_size)
            num_stamp_rows = np.ceil(len(self.times) / stamps_per_row)
            heights[2] = num_stamp_rows * self.stamp_size
        total_height = np.sum(heights)

        # Create a nested layout of subfigures.
        self._figure = plt.figure(figsize=(total_width, total_height))
        self._figure.suptitle(None)
        _, data_fig, all_stamps_fig = self._figure.subfigures(3, 1, height_ratios=heights, hspace=0.05)
        stat_fig, coadds_fig, curve_fig = data_fig.subfigures(
            1, 3, width_ratios=widths, wspace=0.1, hspace=0.1
        )

        # Create the axes for each part of the visualization.
        self._ax_map = {}
        self._ax_map["stats"] = stat_fig.add_subplot(111)

        psi_ax, phi_ax, lc_ax = curve_fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.25})
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

        # Create the navigation bar. This is at the top of the figure.
        self._controls = {}
        button_width = 1.0 / total_width  # Scale for 1 inch width.
        button_height = 0.25 / total_height  # Scale for 0.25 inch height
        button_bottom = 0.95 - button_height
        left_margin = 0.5 / total_width
        step_size = button_width + 0.25 / total_width  # 0.25 inches between buttons

        # Create the naviation buttons (previous and next) and a text box for ID below them.
        prev_axes = plt.axes([left_margin, button_bottom, button_width, button_height], figure=self._figure)
        prev_button = Button(prev_axes, "Previous", color="white")
        prev_button.on_clicked(self.previous_result)
        self._controls["previous"] = prev_button

        next_axes = plt.axes(
            [left_margin + step_size, button_bottom, button_width, button_height], figure=self._figure
        )
        next_buttom = Button(next_axes, "Next", color="white")
        next_buttom.on_clicked(self.next_result)
        self._controls["next"] = next_buttom

        row2_top = button_bottom - (1.5 * button_height)
        id_axes = plt.axes([left_margin, row2_top, button_width, button_height], figure=self._figure)
        id_box = TextBox(id_axes, "ID: ")
        self._controls["id_box"] = id_box

        go_axes = plt.axes(
            [left_margin + step_size, row2_top, button_width, button_height], figure=self._figure
        )
        go_button = Button(go_axes, "Go", color="white")
        go_button.on_clicked(self.goto_to_id)
        self._controls["go"] = go_button

        # Create the buttons to classify the results.
        # radio_height = button_height * len(self._labels)
        # radio_axes = plt.axes(
        #    [
        #        left_margin + 2.0 * step_size,
        #        0.95 - radio_height,
        #        button_width,
        #        radio_height,
        #    ],
        #    figure=self._figure,
        # )
        # radio_axes.set_axis_off()
        # radio_button = RadioButtons(radio_axes, labels=self._labels)
        # self._controls["radio_buttons"] = radio_button

        # Create a textbox for free form notes and an update button.
        notes_ax = plt.axes(
            [
                left_margin + 4.0 * step_size,
                button_bottom + button_height,
                0.95 - left_margin + 4.0 * step_size,
                button_height * 2.0,
            ],
            figure=self._figure,
        )
        notes_box = TextBox(notes_ax, "Notes: ")
        self._controls["notes_box"] = notes_box

        add_note_ax = plt.axes(
            [
                left_margin + 4.0 * step_size,
                button_bottom - (1.5 * button_height),  # Place below the text box
                button_width,
                button_height,
            ],
            figure=self._figure,
        )
        add_button = Button(add_note_ax, "update", color="red")
        add_button.on_clicked(self._update_data)
        self._controls["update"] = add_button

    def update_all(self):
        """Plot all the results in the visualizer."""
        self.plot_stats()
        self.plot_curves()
        self.plot_coadds()
        self.plot_all_stamps()
        self._update_controls()

    def plot_curves(self):
        """Update the time series curves."""
        psi_axis = self._ax_map["psi_curve"]
        phi_axis = self._ax_map["phi_curve"]
        lc_axis = self._ax_map["lightcurve"]

        # Clear the axes before plotting.
        psi_axis.clear()
        phi_axis.clear()
        lc_axis.clear()

        # If we do not have the time series, then display a message in each box.
        if "psi_curve" not in self.results.colnames or "phi_curve" not in self.results.colnames:
            psi_axis.text(0.1, 0.1, "No PSI data", horizontalalignment="center", verticalalignment="center")
            phi_axis.text(0.1, 0.1, "No PHI data", horizontalalignment="center", verticalalignment="center")
            lc_axis.text(
                0.1, 0.1, "No Lightcurve data", horizontalalignment="center", verticalalignment="center"
            )
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

    def plot_stats(self):
        """Update the statistics plot."""
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
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=12,
        )
        stats_axis.set_axis_off()

    def plot_all_stamps(self):
        """Update the all stamps plot."""
        if "all_stamps" not in self.results.colnames:
            # If there are no stamps, display a message.
            stamps_axis = self._ax_map["all_stamps"]
            stamps_axis.clear()
            stamps_axis.text(0.5, 0.5, "Individual stamps not available")
            stamps_axis.set_axis_off()
            return

        curr_stamps = np.asanyarray(self.results["all_stamps"][self.idx])
        num_stamps = curr_stamps.shape[0]

        for i in range(num_stamps):
            curr_ax = self._ax_map[f"all_stamps_{i}"]
            curr_ax.clear()
            plot_image(
                curr_stamps[i],
                ax=curr_ax,
                title=f"Stamp {i + 1}",
                cmap="gray",
                show_counts=False,
            )
            curr_ax.set_aspect("equal")
            curr_ax.set_axis_off()

    def plot_coadds(self):
        """Update the statistics plot."""
        for coadd_idx, coadd in enumerate(self.coadds):
            curr_ax = self._ax_map[coadd]
            curr_ax.clear()
            plot_image(
                self.results[coadd][self.idx],
                ax=curr_ax,
                title=coadd,
                cmap="gray",
                show_counts=False,
            )
            curr_ax.set_aspect("equal")
            curr_ax.set_axis_off()

    def _update_controls(self):
        """Update the controls based on the current result."""
        # Disable/enable the next/prev buttons depending on the current index.
        if self.idx == 0:
            self._controls["previous"].set_active(False)
        else:
            self._controls["previous"].set_active(True)

        if self.idx == len(self.results) - 1:
            self._controls["next"].set_active(False)
        else:
            self._controls["next"].set_active(True)

        # Update the ID box to reflect the current index.
        self._controls["id_box"].set_val(str(self.idx))

        # Set the notes box to the current notes if there are any.
        if "notes" in self.results.colnames:
            current_notes = self.results["notes"][self.idx]
            self._controls["notes_box"].set_val(current_notes)
        else:
            self._controls["notes_box"].set_val("")

        # Update the radio buttons to reflect the current classification.
        # current_classification = self.results["user_class"][self.idx]
        # if current_classification in self._labels:
        #    label_idx = self._labels.index(current_classification)
        # else:
        #    label_idx = 0
        # self._controls["radio_buttons"].set_active(label_idx)

        #    label_idx = self._controls["radio_buttons"].labels.index(current_classification)
        #
        # else:
        #    self._controls["radio_buttons"].set_active(0)


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
