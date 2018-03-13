import os
import shutil
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import astropy.coordinates as astroCoords
import astropy.units as u
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from skimage import measure

def kalman_filter(obs, var):

    xhat = np.zeros(len(obs))
    P = np.zeros(len(obs))
    xhatminus = np.zeros(len(obs))
    Pminus = np.zeros(len(obs))
    K = np.zeros(len(obs))
    
    Q = 1.
    R = np.copy(var)
    
    xhat[0] = obs[0]
    P[0] = R[0]

    for k in range(1,len(obs)):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        K[k] = Pminus[k] / (Pminus[k] + R[k])
        xhat[k] = xhatminus[k] + K[k]*(obs[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    return xhat, P


def return_indices(psi_values, phi_values, val_on):

    flux_vals = psi_values/phi_values
    flux_idx = np.where(flux_vals != 0.)[0]
    if len(flux_idx) < 2:
        return ([], [-1], [])
    fluxes = flux_vals[flux_idx]
    inv_flux = np.array(phi_values[flux_idx])
    inv_flux[inv_flux < -999.] = 9999999.
    f_var = (1./inv_flux)
    
    ## 1st pass
    #f_var = #var_curve[flux_idx]#np.var(fluxes)*np.ones(len(fluxes))
    kalman_flux, kalman_error = kalman_filter(fluxes, f_var)
    if np.min(kalman_error) < 0.:
        return ([], [-1], [])
    deviations = np.abs(kalman_flux - fluxes) / kalman_error**.5
    
    #print(deviations, fluxes)
    # keep_idx = np.where(deviations < 500.)[0]
    keep_idx = np.where(deviations < 5.)[0]

    ## Second Pass (reverse order in case bright object is first datapoint)
    kalman_flux, kalman_error = kalman_filter(fluxes[::-1], f_var[::-1])
    if np.min(kalman_error) < 0.:
        return ([], [-1], [])
    deviations = np.abs(kalman_flux - fluxes[::-1]) / kalman_error**.5
    #print(fluxes, f_var, kalman_flux, kalman_error**.5, deviations)
    # keep_idx_back = np.where(deviations < 500.)[0]
    keep_idx_back = np.where(deviations < 5.)[0]

    if len(keep_idx) >= len(keep_idx_back):
        new_lh = np.sum(psi_values[flux_idx[keep_idx]])/np.sqrt(np.sum(phi_values[flux_idx[keep_idx]]))
        return (val_on, flux_idx[keep_idx], new_lh)
    else:
        reverse_idx = len(flux_idx)-1 - keep_idx_back
        new_lh = np.sum(psi_values[flux_idx[reverse_idx]])/np.sqrt(np.sum(phi_values[flux_idx[reverse_idx]]))
        return (val_on, flux_idx[reverse_idx], new_lh)

def stamp_filter_parallel(stamps):

    s = stamps - np.min(stamps)
    s /= np.sum(s)
    s = np.array(s, dtype=np.dtype('float64')).reshape(21, 21)
    mom = measure.moments_central(s, cr=10, cc=10)
    mom_list = [mom[2, 0], mom[0, 2], mom[1, 1], mom[1, 0], mom[0, 1]]
    peak_1, peak_2 = np.where(s == np.max(s))

    if len(peak_1) > 1:
        peak_1 = np.max(np.abs(peak_1-10.))

    if len(peak_2) > 1:
        peak_2 = np.max(np.abs(peak_2-10.))

    if ((mom_list[0] < 35.5) & (mom_list[1] < 35.5) & 
            (np.abs(mom_list[2]) < 1.) & 
            (np.abs(mom_list[3]) < .25) & (np.abs(mom_list[4]) < .25) & 
            (np.abs(peak_1 - 10.) < 2.) & (np.abs(peak_2 - 10.) < 2.)):
        keep_stamps = 1
    else:
        keep_stamps = 0

    return keep_stamps


class analysis_utils(object):

    def __init__(self):

        return

    def cluster_results(self, results, x_size, y_size, 
                        v_lim, ang_lim, dbscan_args=None):

        default_dbscan_args = dict(eps=0.03, min_samples=-1, n_jobs=-1)

        if dbscan_args is not None:
            default_dbscan_args.update(dbscan_args)
        dbscan_args = default_dbscan_args  

        x_arr = []
        y_arr = []
        vel_arr = []
        ang_arr = []

        for line in results:
            x_arr.append(line.x)
            y_arr.append(line.y)
            vel_arr.append(np.sqrt(line.x_v**2. + line.y_v**2.))
            ang_arr.append(np.arctan(line.y_v/line.x_v))

        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        vel_arr = np.array(vel_arr)
        ang_arr = np.array(ang_arr)

        scaled_x = x_arr/x_size
        scaled_y = y_arr/y_size
        scaled_vel = (vel_arr - v_lim[0])/(v_lim[1] - v_lim[0])
        scaled_ang = (ang_arr - ang_lim[0])/(ang_lim[1] - ang_lim[0])

        db_cluster = DBSCAN(**dbscan_args)

        db_cluster.fit(np.array([scaled_x, scaled_y,
                                scaled_vel, scaled_ang], dtype=np.float).T)

        top_vals = []
        for cluster_num in np.unique(db_cluster.labels_):
            cluster_vals = np.where(db_cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])

        del(db_cluster)

        return top_vals

    def calc_ecliptic_angle(self, test_wcs, angle_to_ecliptic=0.):

        wcs = [test_wcs]
        pixel_coords = [[],[]]
        pixel_start = [[1000, 2000]]
        angle = np.float(angle_to_ecliptic)
        vel_array = np.array([[6.*np.cos(angle), 6.*np.sin(angle)]])
        time_array = [0.0, 1.0, 2.0]

        vel_par_arr = vel_array[:, 0]
        vel_perp_arr = vel_array[:, 1]

        if type(vel_par_arr) is not np.ndarray:
            vel_par_arr = [vel_par_arr]
        if type(vel_perp_arr) is not np.ndarray:
            vel_perp_arr = [vel_perp_arr]
        for start_loc, vel_par, vel_perp in zip(pixel_start, 
                                                vel_par_arr, vel_perp_arr):

            start_coord = astroCoords.SkyCoord.from_pixel(start_loc[0],
                                                          start_loc[1],
                                                          wcs[0])
            eclip_coord = start_coord.geocentrictrueecliptic
            eclip_l = []
            eclip_b = []
            for time_step in time_array:
                eclip_l.append(eclip_coord.lon + vel_par*time_step*u.arcsec)
                eclip_b.append(eclip_coord.lat + vel_perp*time_step*u.arcsec)
            eclip_vector = astroCoords.SkyCoord(eclip_l, eclip_b,
                                                frame='geocentrictrueecliptic')
            pixel_coords_set = astroCoords.SkyCoord.to_pixel(eclip_vector, wcs[0])
            pixel_coords[0].append(pixel_coords_set[0])
            pixel_coords[1].append(pixel_coords_set[1])

        pixel_coords = np.array(pixel_coords)


        x_dist = pixel_coords[0, 0, -1] - pixel_coords[0, 0, 0]
        y_dist = pixel_coords[1, 0, -1] - pixel_coords[1, 0, 0]

        eclip_angle = np.arctan(y_dist/x_dist)

        return eclip_angle
