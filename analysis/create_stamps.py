import os
import random
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

    def load_psi_phi(self, psi_filename, phi_filename, lc_index_filename):
        psi = []
        phi = []
        lc_index = []
        with open(psi_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                psi.append(np.array(row, dtype=np.float))
        with open(phi_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                phi.append(np.array(row, dtype=np.float))
        with open(lc_index_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                lc_index.append(np.array(row, dtype=np.int))
        return(psi, phi, lc_index)

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
        #stamp_normalized = stamps/np.sum(stamps, axis=1).reshape(len(stamps), 1)

        return stamps

    def stamp_filter(self, stamps, center_thresh, verbose=True):

        keep_stamps = np.where(np.max(stamps, axis=1) > center_thresh)[0]
        if verbose:
            print('Center filtering keeps %i out of %i stamps.'
                  % (len(keep_stamps), len(stamps)))
        return keep_stamps

    def load_results(self, res_filename):

        results = np.genfromtxt(res_filename, usecols=(1,3,5,7,9,11,13),
                                names=['lh', 'flux', 'x', 'y', 'vx', 'vy', 'num_obs'])
        return results

    def plot_all_stamps(
        self, results, lc, lc_index, coadd_stamp, stamps, sample=False):
        """Plot the coadded and individual stamps of the candidate object
           along with its lightcurve.
        """
        # Set the rows and columns for the stamp subplots.
        # These will affect the size of the lightcurve subplot.
        numCols=5
        # Find the number of subplots to make.
        numPlots = len(stamps)
        # Compute number of rows for the plot
        numRows = numPlots // numCols
        # Add a row if numCols doesn't divide evenly into numPlots
        if (numPlots % numCols):
            numRows+=1
        # Add a row if numRows=1. Avoids an error caused by ax being 1D.
        if (numRows==1):
            numRows+=1
        # Add a row for the lightcurve subplots
        numRows+=1
        if sample:
            numRows=4
        # Plot the coadded stamp and the lightcurve
        # Generate the stamp plots, setting the size with figsize
        fig,ax = plt.subplots(nrows=numRows,ncols=numCols,
                              figsize=[3.5*numCols,3.5*numRows])
        # In the first row, we only want the coadd and the lightcurve.
        # Delete all other axes.
        for i in range(numCols):
            if i>1:
                fig.delaxes(ax[0,i])
        # Plot coadd and lightcurve
        x_values = np.linspace(1,len(lc),len(lc))
        coadd_stamp = coadd_stamp.reshape(21,21)
        ax[0,0].imshow(coadd_stamp)
        ax[0,1] = plt.subplot2grid((numRows,numCols), (0,1),colspan=4,rowspan=1)
        ax[0,1].plot(x_values,lc,'b')
        ax[0,1].plot(x_values[lc==0],lc[lc==0],'g',lw=4)
        ax[0,1].plot(x_values[lc_index],lc[lc_index],'r.',ms=15)
        ax[0,1].xaxis.set_ticks(x_values)
        res_line = results
        ax[0,1].set_title('Pixel (x,y) = (%i, %i), Vel. (x,y) = (%f, %f), Lh = %f' %
                  (res_line['x'], res_line['y'], res_line['vx'], 
                       res_line['vy'], res_line['lh']))
        plt.xticks(np.arange(min(x_values), max(x_values)+1, 5.0))

        # Turn off all axes. They will be turned back on for proper plots.
        for row in ax[1:]:
            for column in row:
                column.axis('off')
        size = 21
        sigma_x = 1.4
        sigma_y = 1.4

        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)

        x, y = np.meshgrid(x, y)
        gaussian_kernel = (1/(2*np.pi*sigma_x*sigma_y) 
            * np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))))
        sum_pipi = np.sum(gaussian_kernel**2)
        noise_kernel = np.zeros((21,21))
        x_mask = np.logical_or(x>5, x<-5)
        y_mask = np.logical_or(y>5, y<-5)
        mask = np.logical_or(x_mask,y_mask)
        noise_kernel[mask] = 1
        SNR = np.zeros(len(stamps)) 
        signal = np.zeros(len(stamps))
        noise = np.zeros(len(stamps))
        # Plot stamps of individual visits
        axi=1
        axj=0
        if sample:
            mask = np.array(random.sample(range(1,len(stamps)),15))
        else:
            mask = np.linspace(0,len(stamps),len(stamps)+1)
        for j,stamp in enumerate(stamps):
            signal[j] = np.sum(stamp*gaussian_kernel)
            noise[j] = np.var(stamp*noise_kernel)
            SNR[j] = signal[j]/np.sqrt(noise[j]*sum_pipi)
            if (mask == j).any():
                im = ax[axi,axj].imshow(stamp)
                ax[axi,axj].set_title(
                    'visit={0:d} | SNR={1:.2f}'.format(j+1,SNR[j]))
                ax[axi,axj].axis('on')
                # If KBMOD says the index is valid, highlight in red
                if (lc_index==j).any():
                    for axis in ['top','bottom','left','right']:
                        ax[axi,axj].spines[axis].set_linewidth(4)
                        ax[axi,axj].spines[axis].set_color('r')
                    ax[axi,axj].tick_params(axis='x', colors='red')
                    ax[axi,axj].tick_params(axis='y', colors='red')
                # Compute the axis indexes for the next iteration
                if axj<numCols-1:
                    axj += 1
                else:
                    axj = 0
                    axi += 1
        coadd_signal = np.sum(coadd_stamp*gaussian_kernel)
        coadd_noise = np.var(coadd_stamp*noise_kernel)
        coadd_SNR = coadd_signal/np.sqrt(coadd_noise*sum_pipi)
        Psi = np.sum(signal[lc_index]/noise[lc_index])
        Phi = np.sum(sum_pipi/noise[lc_index])
        summed_SNR = Psi/np.sqrt(Phi)
        ax[0,0].set_title(
            'Total SNR={:.2f}'.format(coadd_SNR))
        #ax[0,0].set_title(
        #    'Total SNR={:.2f}\nSummed SNR={:.2f}'.format(coadd_SNR,summed_SNR))
        for axis in ['top','bottom','left','right']:
            ax[0,0].spines[axis].set_linewidth(4)
            ax[0,0].spines[axis].set_color('r')
        ax[0,0].tick_params(axis='x', colors='red')
        ax[0,0].tick_params(axis='y', colors='red')
        return(fig) 

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

    def target_results(
        self, results, lc, lc_index, target_xy, stamps=None,
        center_thresh=None, target_vel=None, vel_tol=5, atol=10,
        title_info=None):
        keep_idx = np.linspace(0,len(lc)-1,len(lc)).astype(int)
        if stamps is not None:
            keep_idx = self.stamp_filter(stamps, center_thresh, verbose=False)
        recovered_idx = []
        # Count the number of objects within atol of target_xy
        count=0
        object_found=False
        for i,stamp_idx in enumerate(keep_idx):
            res_line = results[stamp_idx]
            if target_vel is not None:
                vel_truth = (
                    np.isclose(res_line['vx'], target_vel[0], atol=vel_tol) and
                    np.isclose(res_line['vy'], target_vel[1], atol=vel_tol))
            else:
                vel_truth = True

            if (np.isclose(res_line['x'],target_xy[0],atol=atol) 
                and np.isclose(res_line['y'],target_xy[1],atol=atol)
                and vel_truth):
                recovered_idx.append(stamp_idx)
                count+=1
        # Plot lightcurves of objects within atol of target_xy
        if count>0:
            object_found=True
        else:
            return(0,False,[])
        y_size = count

        fig = plt.figure(figsize=(12, 2*y_size))
        count=0
        for i,stamp_idx in enumerate(keep_idx):
            res_line = results[stamp_idx]
            if target_vel is not None:
                vel_truth = (
                    np.isclose(res_line['vx'], target_vel[0], atol=vel_tol) and
                    np.isclose(res_line['vy'], target_vel[1], atol=vel_tol))
            else:
                vel_truth = True

            if (np.isclose(res_line['x'],target_xy[0],atol=atol)
                and np.isclose(res_line['y'],target_xy[1],atol=atol)
                and vel_truth):
                current_lc = lc[stamp_idx]
                current_lc_index = lc_index[stamp_idx]
                x_values = np.linspace(1,len(current_lc),len(current_lc))
                if stamps is not None:
                    fig.add_subplot(y_size,2,(count*2)+1)
                    plt.imshow(stamps[stamp_idx].reshape(21,21))
                fig.add_subplot(y_size,2,(count*2)+2)
                plt.plot(x_values,current_lc,'b')
                plt.plot(x_values[current_lc==0],current_lc[current_lc==0],'g.',ms=15)
                plt.plot(x_values[current_lc_index],current_lc[current_lc_index],'r.',ms=15)
                plt.xticks(x_values)
                title = 'Pixel (x,y) = ({}, {}), Vel. (x,y) = ({}, {}), Lh = {}, index = {}' 
                if title_info is not None:
                    title = title_info+'\n'+title
                plt.title(title.format(
                    res_line['x'], res_line['y'], res_line['vx'],
                    res_line['vy'], res_line['lh'], stamp_idx))
                count+=1
        plt.tight_layout()
        return(fig, object_found, recovered_idx)

    def calc_mag(self, image_files, lc, idx_list):

        flux_vals = []
        
        for filenum, lc_val in zip(idx_list, lc):
            hdulist = fits.open(image_files[int(filenum)])
            j_flux = lc_val/hdulist[0].header['FLUXMAG0']
            flux_vals.append(j_flux)

        return -2.5*np.log10(np.mean(flux_vals))
