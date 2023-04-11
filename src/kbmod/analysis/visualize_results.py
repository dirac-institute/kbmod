import os
import pickle

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output, display
from kbmod.analysis.create_stamps import *


class VisualizeResults:
    def __init__(self, results_dir_format, im_dir_format, cnn_path="./resnet_2.h5", load_filt_tools=True):
        self.results_dir_format = results_dir_format
        self.im_dir_format = im_dir_format
        self.starting_x_lim = 2048
        if load_filt_tools:
            self.filter_tools = CNNFilter(cnn_path)
        else:
            self.filter_tools = None

    def compare_filter_effectiveness(self):
        num_results = {"cnn_lh15": [], "cnn_lh10": [], "lh10": [], "lh15": []}
        num_except = 0
        ccd_num = np.linspace(1, 62, 62).astype(int)
        ccd_num = ccd_num[ccd_num != 2]
        ccd_num = ccd_num[ccd_num != 61]
        for ccd in ccd_num:
            try:
                stamps = self.show_pg_results(
                    190, ccd, cutoff=0.75, suffix="XSEDE", plot_stamps="none", lh_lim=15
                )
                num_results["cnn_lh15"].append(len(stamps))
                stamps = self.show_pg_results(
                    190, ccd, cutoff=0.75, suffix="XSEDE", plot_stamps="none", lh_lim=10
                )
                num_results["cnn_lh10"].append(len(stamps))
                stamps = self.show_pg_results(
                    190, ccd, cutoff=0.0, suffix="XSEDE", plot_stamps="none", lh_lim=15
                )
                num_results["lh15"].append(len(stamps))
                stamps = self.show_pg_results(
                    190, ccd, cutoff=0.0, suffix="XSEDE", plot_stamps="none", lh_lim=10
                )
                num_results["lh10"].append(len(stamps))
            except:
                num_except += 1
        return num_results

    def plot_res_starting_xy(self, pg_num, suffix="XSEDE", x_lim=[0, 1e6], y_lim=[0, 1e6]):
        stamper = CreateStamps()
        x_loc = []
        y_loc = []
        lh = []
        all_results = []
        ccd_nums = np.linspace(1, 62, 62).astype(int)
        for ccd_num in ccd_nums:
            results_dir = "/epyc/users/smotherh/xsede_results/{0:03}/{1:02d}/".format(pg_num, ccd_num)
            result_filename = os.path.join(results_dir, "results_%s.txt" % suffix)
            result_exists = os.path.isfile(result_filename)
            if result_exists:
                results = stamper.load_results(result_filename)
                all_results.append(results)
                for result in results:
                    x_loc.append(result[2])
                    y_loc.append(result[3])
                    lh.append(result[0])
        all_results = np.concatenate(all_results, axis=0)
        x_loc = np.array(x_loc)
        y_loc = np.array(y_loc)
        lh = np.array(lh)
        x_mask = np.logical_and(x_loc <= x_lim[1], x_loc >= x_lim[0])
        y_mask = np.logical_and(y_loc <= y_lim[1], y_loc >= y_lim[0])
        pos_mask = np.logical_and(x_mask, y_mask)
        mask = np.logical_and(pos_mask, lh > 15)

        plt.hist(x_loc[mask])
        plt.xlabel("Starting x pixel")
        plt.ylabel("Number")
        plt.figure()
        plt.hist(y_loc[mask])
        plt.xlabel("Starting y pixel")
        plt.ylabel("Number")
        print("There are {} results".format(len(x_loc[mask])))

    def recover_known_objects(
        self, results_dir_format, im_dir_format, known_object_data_path, cutoff=0.75, suffix="XSEDE"
    ):
        with open(known_object_data_path, "rb") as f:
            allObjectData = pickle.load(f)

        known_format = "pg{:03d}_ccd{:02d}"
        known_data = []
        for key in allObjectData.keys():
            if (key != "legend") and (int(key[2:5]) > 100):
                known_data.append(np.array([int(key[2:5]), int(key[9:])]))

        all_good_idx = []
        num_found_objects = 0
        found_objects = {}
        missed_objects = {}
        found_vmag = []
        v_mag = []
        num_results_neighbor = []
        res_per_ccd = []
        all_coadd_stamps = []
        good_coadd_stamps = []
        all_stamp_probs = []
        found_stamps = []
        found_results = {}
        exception_list = []
        exception_keys = []
        for pgccd in known_data:
            pg_num = int(pgccd[0])
            ccd_num = int(pgccd[1])

            results_dir = results_dir_format.format(pg_num, ccd_num)
            im_dir = im_dir_format.format(pg_num, ccd_num)

            times_filename = os.path.join(results_dir, "times_%s.txt" % suffix)
            stamper = CreateStamps()
            try:
                object_key = known_format.format(pg_num, ccd_num)
                xy_array = allObjectData[object_key][5]
                v_array = allObjectData[object_key][2]
                v_mag.append(allObjectData[object_key][1])
                times_list = stamper.load_times(times_filename)
                (
                    keep_idx,
                    results,
                    stamper,
                    stamps,
                    all_stamps,
                    lc_list,
                    psi_list,
                    phi_list,
                    lc_index,
                ) = load_stamps(results_dir, im_dir, suffix)
                all_coadd_stamps.append(stamps)
            except Exception as e:
                exception_list.append(e)
                object_key = known_format.format(pg_num, ccd_num)
                exception_keys.append(object_key)
                continue

            if len(stamps) > 0:
                if cutoff > 0 and self.filter_tools is not None:
                    good_idx = self.filter_tools.cnn_filter(np.copy(stamps), cutoff=cutoff)
                else:
                    good_idx = [i for i in range(len(self.stamps))]
            else:
                print("No results found...")
                good_idx = []
            res_per_ccd.append(len(good_idx))
            if len(good_idx) < 1:
                missed_objects[object_key] = allObjectData[object_key]
                print("Failed CCN Filtering. Continuing...")
                continue
            for idx in good_idx:
                good_coadd_stamps.append(stamps[idx])

            if np.count_nonzero(results) != 0:
                if len(lc_list) == 1:
                    results = np.array([results])

                stamps_fig, object_found, found_idx = stamper.target_results(
                    np.array(results)[good_idx],
                    np.array(lc_list)[good_idx],
                    np.array(lc_index)[good_idx],
                    xy_array,
                    stamps=np.copy(stamps[good_idx]),
                    center_thresh=0.00,
                    target_vel=v_array,
                    vel_tol=5,
                    atol=5,
                    title_info="pg_num={:03d}, ccd={:02d}".format(*pgccd.astype(int)),
                )
                if object_found:
                    num_found_objects += 1
                    found_stamps.append(stamps[good_idx][found_idx])
                    found_vmag.append(allObjectData[object_key][1])
                    found_objects[object_key] = allObjectData[object_key]
                    found_results[object_key] = results[good_idx][found_idx]
                else:
                    missed_objects[object_key] = allObjectData[object_key]
            else:
                missed_objects[object_key] = allObjectData[object_key]
        print("Found {} objects".format(len(found_objects)))
        print("Missed {} objects".format(len(missed_objects)))
        print("Exceptions {}".format(len(exception_keys)))
        return (found_objects, found_results, v_mag, found_vmag)

    def plot_completeness(self, found_vmag, v_mag, limiting_mag):
        fig = plt.figure(figsize=[12, 8])
        percent_recovered = len(found_vmag) / len(v_mag)
        plt.hist(v_mag, color="tab:blue", bins="fd")
        plt.hist(found_vmag, range=[np.min(v_mag), np.max(v_mag)], color="tab:orange", bins="fd")
        plt.xlabel("V", fontsize=20)
        plt.ylabel("Number of Objects", fontsize=20)
        plt.axvline(limiting_mag, color="black", lw=4, ls="--")
        _ = plt.title("Known Object Recovery\n{:.3f} Completeness".format(percent_recovered), fontsize=20)
        plt.legend(
            [r"Faintest single-image 10$\sigma$ depth", "All Objects", "Recovered Objects"], fontsize=16
        )

        return fig

    def compare_results(self, found_objects, found_results):
        found_pos = []
        found_vel = []
        found_speed = []
        pred_pos = []
        pred_vel = []
        pred_speed = []

        arcsec_per_pixel = 0.27

        for key in found_objects:
            found_x = found_results[key][0][2]
            found_y = found_results[key][0][3]
            found_vx = found_results[key][0][4]
            found_vy = found_results[key][0][5]
            found_pos.append([found_x, found_y])
            found_vel.append([found_vx, found_vy])
            found_speed.append(np.linalg.norm([found_vx, found_vy]))
            pred_pos.append(found_objects[key][5])
            pred_vel.append(found_objects[key][2])
            pred_speed.append(np.linalg.norm(found_objects[key][2]))
        found_pos = np.array(found_pos) * arcsec_per_pixel
        found_vel = np.array(found_vel) * arcsec_per_pixel / 24
        found_speed = np.array(found_speed) * arcsec_per_pixel / 24
        pred_pos = np.array(pred_pos) * arcsec_per_pixel
        pred_vel = np.array(pred_vel) * arcsec_per_pixel / 24
        pred_speed = np.array(pred_speed) * arcsec_per_pixel / 24

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=[12, 8])

        ax[0, 0].scatter(found_pos[:, 0], found_pos[:, 1], c="tab:blue")
        ax[0, 0].scatter(pred_pos[:, 0], pred_pos[:, 1], c="tab:orange")
        ax[0, 0].set_xlabel("Initial X Position (arcsec)", fontsize=16)
        ax[0, 0].set_ylabel("Initial Y Position (arcsec)", fontsize=16)
        ax[0, 0].legend(["Recovered Position", "Predicted Position"], fontsize=12)
        for i in range(len(found_pos)):
            ax[0, 0].plot([found_pos[i, 0], pred_pos[i, 0]], [found_pos[i, 1], pred_pos[i, 1]], c="k")

        ax[0, 1].scatter(found_vel[:, 0], found_vel[:, 1], c="tab:blue")
        ax[0, 1].scatter(pred_vel[:, 0], pred_vel[:, 1], c="tab:orange")
        ax[0, 1].set_xlabel("X Velocity (arcsec/hr)", fontsize=16)
        ax[0, 1].set_ylabel("Y Velocity (arcsec/hr)", fontsize=16)
        ax[0, 1].legend(["Recovered Velocity", "Predicted Velocity"], fontsize=12)

        for i in range(len(found_vel)):
            ax[0, 1].plot([found_vel[i, 0], pred_vel[i, 0]], [found_vel[i, 1], pred_vel[i, 1]], c="k")
        deltaSpeed = np.linalg.norm(found_vel - pred_vel, axis=1)
        print("Media Speed Residual: {:.3e}".format(np.median(deltaSpeed)))
        ax[1, 1].hist(deltaSpeed, bins="fd")
        ax[1, 1].axvline(np.median(deltaSpeed), color="black", lw=4, ls="--")
        ax[1, 1].set_xlabel("Speed Residual (arcsec/hr)", fontsize=16)
        ax[1, 1].set_ylabel("Number of Objects", fontsize=16)
        ax[1, 1].legend(["Median Residual"], fontsize=12)

        # _=plt.hist(pred_speed)
        deltaPos = np.linalg.norm(found_pos - pred_pos, axis=1)
        ax[1, 0].hist(deltaPos, bins="fd")  # "
        print("Median Position Residual: {:.3e}".format(np.median(deltaPos)))
        ax[1, 0].axvline(np.median(deltaPos), color="black", lw=4, ls="--")
        ax[1, 0].set_xlabel("Position Residual (arcsec)", fontsize=16)
        ax[1, 0].set_ylabel("Number of Objects", fontsize=16)
        ax[1, 0].legend(["Median Residual"], fontsize=12)
        fig.suptitle("Recovered Results vs Predicted Results", fontsize=20)

        for ax0 in ax.reshape(-1):
            ax0.tick_params(labelsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return (fig, ax)

    def plot_init(self):
        with self.out:
            clear_output(wait=True)
            i = self.good_idx[0]
            self.stamps_fig = self.stamper.plot_all_stamps(
                self.results[i],
                np.array(self.lc_list)[i],
                np.array(self.lc_index)[i],
                np.copy(self.stamps[i]),
                np.copy(self.all_stamps[i]),
                stamp_index=i,
                compare_SNR=False,
                show_fig=False,
            )
            display(self.stamps_fig)
            self.stamps_fig_prev = self.stamps_fig
            if len(self.good_idx) > 1:
                i = self.good_idx[1]
                self.stamps_fig_next = self.stamper.plot_all_stamps(
                    self.results[i],
                    np.array(self.lc_list)[i],
                    np.array(self.lc_index)[i],
                    np.copy(self.stamps[i]),
                    np.copy(self.all_stamps[i]),
                    stamp_index=i,
                    compare_SNR=False,
                    show_fig=False,
                )
        self.counter.value = "{:3d}/{:3d} | {:2d}".format(
            self.current_index + 1, len(self.good_idx), self.current_ccd
        )

    def plot_next(self, b):
        self.current_index += 1
        if self.current_index < len(self.good_idx):
            self.stamps_fig_prev = self.stamps_fig
            i = self.good_idx[self.current_index]
            with self.out:
                clear_output(wait=True)
                self.stamps_fig = self.stamps_fig_next
                display(self.stamps_fig)
                self.counter.value = "{:3d}/{:3d} | {:2d}".format(
                    self.current_index + 1, len(self.good_idx), self.current_ccd
                )
            if self.current_index + 1 < len(self.good_idx):
                i = self.good_idx[self.current_index + 1]
                self.stamps_fig_next = self.stamper.plot_all_stamps(
                    self.results[i],
                    np.array(self.lc_list)[i],
                    np.array(self.lc_index)[i],
                    np.copy(self.stamps[i]),
                    np.copy(self.all_stamps[i]),
                    stamp_index=i,
                    compare_SNR=False,
                    show_fig=False,
                )
        else:
            self.current_index -= 1
            print("No more results in this ccd")

    def plot_prev(self, b):
        self.current_index -= 1
        if self.current_index >= 0:
            i = self.good_idx[self.current_index]
            self.stamps_fig_next = self.stamps_fig
            with self.out:
                clear_output(wait=True)
                self.stamps_fig = self.stamps_fig_prev
                display(self.stamps_fig)
                self.counter.value = "{:3d}/{:3d} | {:2d}".format(
                    self.current_index + 1, len(self.good_idx), self.current_ccd
                )
            if self.current_index >= 1:
                i = self.good_idx[self.current_index - 1]
                self.stamps_fig_prev = self.stamper.plot_all_stamps(
                    self.results[i],
                    np.array(self.lc_list)[i],
                    np.array(self.lc_index)[i],
                    np.copy(self.stamps[i]),
                    np.copy(self.all_stamps[i]),
                    stamp_index=i,
                    compare_SNR=False,
                    show_fig=False,
                )
        else:
            self.current_index += 1
            print("Already displaying first result")

    def save_current_fig(self, b):
        self.stamps_fig.savefig(
            "./saved_trajectory_images/{:03d}_{:02d}_{}_{:07d}".format(
                self.current_pg, self.current_ccd, self.suffix, self.good_idx[self.current_index]
            )
        )

    def _load_stamps(self):
        results_dir = self.results_dir_format.format(self.current_pg, self.current_ccd)
        im_dir = self.im_dir_format.format(self.current_pg, self.current_ccd)
        times_filename = os.path.join(results_dir, "times_%s.txt" % self.suffix)
        stamper = CreateStamps()

        times_list = stamper.load_times(times_filename)
        (
            self.keep_idx,
            self.results,
            self.stamper,
            self.stamps,
            self.all_stamps,
            self.lc_list,
            self.psi_list,
            self.phi_list,
            self.lc_index,
        ) = load_stamps(results_dir, im_dir, self.suffix)
        if len(self.lc_list) == 1:
            self.results = np.array([self.results])

    def _run_filter(self):
        """
        Filters the results based on likelihood, x value, and the
        CNN (if load_filt_tools == True). Save the indices that pass
        all three filters to good_idx.
        """
        result_lh = np.array([result["lh"] for result in self.results])
        result_x = np.array([result["x"] for result in self.results])
        lh_idx = np.where(result_lh >= self.lh_lim)[0]
        edge_idx = np.where(result_x <= self.starting_x_lim)[0]

        # Perform the CNN filtering only if cutoff != 0 and
        # the filter tools were loaded.
        if self.cutoff == 0 or self.filter_tools is None:
            stamp_idx = [i for i in range(len(self.stamps))]
        else:
            stamp_idx = self.filter_tools.cnn_filter(np.copy(self.stamps), cutoff=self.cutoff)

        # The indices to use are the ones that pass all three filters.
        self.good_idx = np.intersect1d(np.intersect1d(lh_idx, stamp_idx), edge_idx)

    def _next_ccd(self, b):
        self.current_ccd += 1
        self.ccd_data = False
        while (self.current_ccd <= 62) and (self.ccd_data == False):
            try:
                self._load_stamps()
                self._run_filter()
                self.current_index = 0
                if len(self.good_idx) > 0:
                    self.plot_init()
                else:
                    with self.out:
                        clear_output()
                    self.counter.value = "{:3d}/{:3d} | {:2d}".format(
                        self.current_index + 1, len(self.good_idx), self.current_ccd
                    )
                self.ccd_data = True
            except:
                self.current_ccd += 1

    def show_pg_results(
        self,
        pg_num,
        ccd_num,
        suffix="current",
        plot_stamps="coadd",
        cutoff=0.0,
        lh_lim=10.0,
        index=None,
        use_widget=False,
    ):
        self.current_pg = pg_num
        self.current_ccd = ccd_num
        self.suffix = suffix
        self.cutoff = cutoff
        self.lh_lim = lh_lim

        res_per_ccd = []
        all_coadd_stamps = []
        good_coadd_stamps = []
        all_stamp_probs = []
        found_stamps = []
        found_results = {}
        num_except = 0
        num_stamps = 0

        self._load_stamps()
        num_stamps += len(self.stamps)

        self._run_filter()

        res_per_ccd.append(len(self.good_idx))
        if len(self.good_idx) < 1:
            print("Failed CCN Filtering. Continuing...")
            return []

        if np.count_nonzero(self.results) != 0:
            if plot_stamps == "coadd":
                stamps_fig = self.stamper.plot_stamps(
                    self.results[self.good_idx],
                    np.array(self.lc_list)[self.good_idx],
                    np.array(self.lc_index)[self.good_idx],
                    np.copy(self.stamps[self.good_idx]),
                    0.00,
                )
                # stamps_fig = stamper.plot_stamps(results, lc_list, lc_index, stamps, 0.03)
            elif plot_stamps == "all":
                print("Plotting {} results".format(len(self.good_idx)))
                final_num_plotted = 0
                if index is None:
                    if use_widget is False:
                        for i in self.good_idx:
                            stamps_fig = self.stamper.plot_all_stamps(
                                self.results[i],
                                np.array(self.lc_list)[i],
                                np.array(self.lc_index)[i],
                                np.copy(self.stamps[i]),
                                np.copy(self.all_stamps[i]),
                                stamp_index=i,
                                compare_SNR=False,
                            )
                            if stamps_fig is not None:
                                final_num_plotted += 1
                        print("{} stamps have consistent SNR. Plotting...".format(final_num_plotted))
                    else:
                        self.current_index = 0
                        button1 = widgets.Button(description="Next", layout=widgets.Layout(width="100px"))
                        button2 = widgets.Button(description="Previous", layout=widgets.Layout(width="100px"))
                        button3 = widgets.Button(description="Save", layout=widgets.Layout(width="100px"))
                        button4 = widgets.Button(description="Next Ccd", layout=widgets.Layout(width="100px"))

                        self.counter = widgets.Text(disabled=True, layout=widgets.Layout(width="100px"))

                        self.out = widgets.Output()

                        buttons = widgets.VBox(children=[button1, button2, button3, self.counter, button4])
                        all_widgets = widgets.HBox(children=[buttons, self.out])
                        display(all_widgets)

                        if len(self.good_idx) > 0:
                            self.plot_init()

                        button1.on_click(self.plot_next)
                        button2.on_click(self.plot_prev)
                        button3.on_click(self.save_current_fig)
                        button4.on_click(self._next_ccd)

                else:
                    stamps_fig = self.stamper.plot_all_stamps(
                        self.results[index],
                        np.array(self.lc_list)[index],
                        np.array(self.lc_index)[index],
                        np.copy(self.stamps[index]),
                        np.copy(self.all_stamps[index]),
                        stamp_index=index,
                    )
            elif plot_stamps == "none":
                print("Returning {} results without plotting".format(len(self.good_idx)))
        return self.stamps[self.good_idx]


class CNNFilter:
    def __init__(self, model_path):
        self.cnn_model = tf.keras.models.load_model(model_path)

    def cnn_filter(self, imgs, cutoff=0.5):
        good_idx = np.linspace(0, len(imgs) - 1, len(imgs))
        true_false = []

        keras_stamps = []
        sigmaG_coeff = 0.7413
        for img in imgs:
            per25, per50, per75 = np.percentile(img, [25, 50, 75])
            sigmaG = sigmaG_coeff * (per75 - per25)
            img[img < (per50 - 2 * sigmaG)] = per50 - 2 * sigmaG
            img -= np.min(img)
            if np.sum(img) != 0:
                img /= np.sum(img)
            img = img.reshape(21, 21)
            img[np.isnan(img)] = 0
            keras_stamps.append(img)
        keras_stamps = np.reshape(keras_stamps, [-1, 21, 21, 1])
        cnn_results = self.cnn_model.predict(keras_stamps)
        good_idx = good_idx[cnn_results[:, 1] > cutoff]

        return good_idx.astype(int)

    def no_filter(self, imgs):
        good_idx = np.linspace(0, len(imgs) - 1, len(imgs))
        return good_idx.astype(int)

    def occlusion_test(self, input_stamp, kernel_size=5):
        i = 0
        j = 0
        heatmap = []
        sigmaG_coeff = 0.7413
        for j in range(22 - kernel_size):
            for i in range(22 - kernel_size):
                img = np.copy(input_stamp)
                img[i : i + kernel_size, j : j + kernel_size] = 0
                per25, per50, per75 = np.percentile(img, [25, 50, 75])
                sigmaG = sigmaG_coeff * (per75 - per25)
                img[img < (per50 - 2 * sigmaG)] = per50 - 2 * sigmaG
                img -= np.min(img)
                img /= np.sum(img)
                keras_stamps = np.reshape(img, [-1, 21, 21, 1])
                probs = np.concatenate(self.cnn_model.predict(keras_stamps))
                heatmap.append(probs[1])
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(input_stamp)
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(np.array(heatmap).reshape(22 - kernel_size, 22 - kernel_size))
        fig.colorbar(im, ax=ax2)
