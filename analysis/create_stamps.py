import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from astropy.io import fits


class create_stamps(object):

    def __init__(self):

        return

    def load_lightcurves(self, lc_filename):

        lc = []
        with open(lc_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                lc.append(np.array(row, dtype=np.float))

        return lc

    def load_times(self, time_filename):

        times = []
        with open(time_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                times.append(np.array(row, dtype=np.float))

        return times

    def load_stamps(self, stamp_filename):

        stamps = np.genfromtxt(stamp_filename)
        if len(np.shape(stamps)) < 2:
            stamps = np.array([stamps])
        stamp_normalized = stamps/np.sum(stamps, axis=1).reshape(len(stamps), 1)

        return stamp_normalized

    def stamp_filter(self, stamps, center_thresh):

        keep_stamps = np.where(np.max(stamps, axis=1) > center_thresh)[0]
        print('Center filtering keeps %i out of %i stamps.' % (len(keep_stamps),
                                                               len(stamps)))
        return keep_stamps

    def load_results(self, res_filename):

        results = np.genfromtxt(res_filename, usecols=(1,3,5,7,9,11,13),
                                names=['lh', 'flux', 'x', 'y', 'vx', 'vy', 'num_obs'])
        return results

    def plot_stamps(self, results, lc, stamps, center_thresh, fig=None):

        keep_idx = self.stamp_filter(stamps, center_thresh)

        if fig is None:
            fig = plt.figure(figsize=(12, len(keep_idx)*2))

        i = 0
        for stamp_idx in keep_idx:
            fig.add_subplot(len(keep_idx),2,(i*2)+1)
            plt.imshow(stamps[stamp_idx].reshape(21,21))
            fig.add_subplot(len(keep_idx),2,(i*2)+2)
            plt.plot(lc[stamp_idx])
            res_line = results[stamp_idx]
            plt.title('Pixel (x,y) = (%i, %i), Vel. (x,y) = (%f, %f), Lh = %f' %
                      (res_line['x'], res_line['y'], res_line['vx'],
                       res_line['vy'], res_line['lh']))
            i+=1
        plt.tight_layout()

        return fig

    def calc_mag(self, image_files, lc, idx_list):

        flux_vals = []
        
        for filenum, lc_val in zip(idx_list, lc):
            hdulist = fits.open(image_files[int(filenum)])
            j_flux = lc_val/hdulist[0].header['FLUXMAG0']
            flux_vals.append(j_flux)

        return -2.5*np.log10(np.mean(flux_vals))
