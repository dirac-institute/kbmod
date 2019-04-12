import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from astropy.io import fits


class create_stamps(object):

    def __init__(self):

        return

    def load_lightcurves(self, lc_filename, lc_index_filename):

        lc = []
        lc_index = []
        with open(lc_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                lc.append(np.array(row, dtype=np.float))
        with open(lc_index_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                lc_index.append(np.array(row, dtype=np.int))

        return lc, lc_index

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

    def plot_all_stamps(self, results, lc, lc_index, coadd_stamp, stamps):
        """Plot the coadded and individual stamps of the candidate object
           along with its lightcurve.
        """
        # Set the rows and columns for the stamp subplots.
        # These will affect the size of the lightcurve subplot.
        numCols=5
        # Find the number of subplots to make. Add one for the coadd.
        numPlots = len(stamps)
        # Compute number of rows for the plot
        numRows = numPlots // numCols
        # Add a row if numCols doesn't divide evenly into numPlots
        if (numPlots % numCols):
            numRows+=1
        # Add a row if numRows=1. Avoids an error caused by ax being 1D.
        if (numRows==1):
            numRows+=1
        # Plot the coadded stamp and the lightcurve
        fig1,ax1 = plt.subplots(1,2,figsize=[2.5*numCols,2.75])
        x_values = np.linspace(1,len(lc),len(lc))
        ax1[0].imshow(coadd_stamp.reshape(21,21))
        ax1[1].plot(x_values,lc,'b')
        ax1[1].plot(x_values[lc==0],lc[lc==0],'g',lw=4)
        ax1[1].plot(x_values[lc_index],lc[lc_index],'r.',ms=15)
        ax1[1].xaxis.set_ticks(x_values)
        res_line = results
        ax1[1].set_title('Pixel (x,y) = (%i, %i), Vel. (x,y) = (%f, %f), Lh = %f' %
                  (res_line['x'], res_line['y'], res_line['vx'], 
                       res_line['vy'], res_line['lh']))
        plt.tight_layout()
        
        # Generate the stamp plots, setting the size with figsize
        fig2,ax2 = plt.subplots(nrows=numRows,ncols=numCols,
                              figsize=[2.5*numCols,2.75*numRows])

        # Turn off all axes. They will be turned back on for proper plots.
        for row in ax2:
            for column in row:
                column.axis('off')

        axi=0
        axj=0
        for j,stamp in enumerate(stamps):

            im=ax2[axi,axj].imshow(stamp)
            ax2[axi,axj].set_title('visit='+str(j+1))
            ax2[axi,axj].axis('on')
            # If KBMOD says the index is valid, highlight in red
            if (lc_index==j).any():
                for axis in ['top','bottom','left','right']:
                    ax2[axi,axj].spines[axis].set_linewidth(4)
                    ax2[axi,axj].spines[axis].set_color('r')
                ax2[axi,axj].tick_params(axis='x', colors='red')
                ax2[axi,axj].tick_params(axis='y', colors='red')
            # Compute the axis indexes for the next iteration
            if axj<numCols-1:
                axj+=1
            else:
                axj=0
                axi+=1
        return(fig1,fig2) 
           
    def plot_stamps(self, results, lc, lc_index, stamps, center_thresh, fig=None):
        keep_idx = self.stamp_filter(stamps, center_thresh)

        if fig is None:
            fig = plt.figure(figsize=(12, len(lc_index)*2))
        for i,stamp_idx in enumerate(keep_idx):
            current_lc = lc[stamp_idx]
            current_lc_index = lc_index[stamp_idx]
            x_values = np.linspace(1,len(current_lc),len(current_lc))
            fig.add_subplot(len(keep_idx),2,(i*2)+1)
            plt.imshow(stamps[stamp_idx].reshape(21,21))
            fig.add_subplot(len(keep_idx),2,(i*2)+2)
            plt.plot(x_values,current_lc,'b')
            plt.plot(x_values[current_lc==0],current_lc[current_lc==0],'g',lw=4)
            plt.plot(x_values[current_lc_index],current_lc[current_lc_index],'r.',ms=15)
            plt.xticks(x_values)
            res_line = results[stamp_idx]
            plt.title('Pixel (x,y) = (%i, %i), Vel. (x,y) = (%f, %f), Lh = %f, index = %i' %
                      (res_line['x'], res_line['y'], res_line['vx'],
                       res_line['vy'], res_line['lh'], stamp_idx))
        plt.tight_layout()

        return fig

    def calc_mag(self, image_files, lc, idx_list):

        flux_vals = []
        
        for filenum, lc_val in zip(idx_list, lc):
            hdulist = fits.open(image_files[int(filenum)])
            j_flux = lc_val/hdulist[0].header['FLUXMAG0']
            flux_vals.append(j_flux)

        return -2.5*np.log10(np.mean(flux_vals))
