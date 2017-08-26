import os
import sys
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from skimage import measure
from sklearn.cluster import DBSCAN
    
def file_and_header_length(filename):

    i = 0
    header_length = 0
    with open(filename, 'r') as f:
        for line in f:
            i+=1
            if line.startswith('#'):
                header_length += 1
                
    return i, header_length

def load_chunk(filename, chunk_start, chunk_size):
    results_arr = np.genfromtxt(filename, names=['t0_x', 't0_y',
                                                 'v_x', 'v_y',
                                                 'likelihood', 'est_flux'],
                                skip_header=chunk_start,
                                max_rows=chunk_size)
    return results_arr

def calcCenters(startLocArr, velArr, timeArr):

    startLocArr = np.array([startLocArr['t0_x'], startLocArr['t0_y']])
    velArr = np.array([velArr['v_x'], velArr['v_y']])
    centerArr = []
    for time in timeArr:
        centerArr.append(startLocArr + (velArr*time))
    return np.array(centerArr)

def createPostageStamp(imageArray, objectStartArr, velArr,
                       timeArr, stamp_width):

    """
    Create postage stamp image coadds of potential objects traveling along
    a trajectory.
    Parameters
    ----------
    imageArray: numpy array, required
    The masked input images.
    objectStartArr: numpy array, required
    An array with the starting location of the object in pixels.
    velArr: numpy array, required
    The x,y velocity in pixels/hr. of the object trajectory.
    timeArr: numpy array, required
    The time in hours of each image starting from 0 at the first image.
    stamp_width: numpy array or list, [2], required
    The row, column dimensions of the desired output image.
    Returns
    -------
    stampImage: numpy array
    The coadded postage stamp.
    singleImagesArray: numpy array
    The postage stamps that were added together to create the coadd.
    """

    singleImagesArray = []
    stampWidth = np.array(stamp_width, dtype=int)
    #print stampWidth
    stampImage = np.zeros(stampWidth)
    
    #if len(np.shape(imageArray)) < 3:
    #    imageArray = [imageArray]
    
    measureCoords = calcCenters(np.array(objectStartArr), np.array(velArr), timeArr)
    
    if len(np.shape(measureCoords)) < 2:
        measureCoords = [measureCoords]
    off_edge = []
    for centerCoords in measureCoords:
        if (centerCoords[0] + stampWidth[0]/2 + 1) > np.shape(imageArray[0].science())[1]:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        elif (centerCoords[0] - stampWidth[0]/2) < 0:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        elif (centerCoords[1] + stampWidth[1]/2 + 1) > np.shape(imageArray[0].science())[0]:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        elif (centerCoords[1] - stampWidth[1]/2) < 0:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        else:
            off_edge.append(False)
            
    i=0
    for image in imageArray:
        if off_edge[i] is False:
            xmin = int(np.rint(measureCoords[i,1]-stampWidth[0]/2))
            xmax = int(xmin + stampWidth[0])
            ymin = int(np.rint(measureCoords[i,0]-stampWidth[1]/2))
            ymax = int(ymin + stampWidth[1])
            #print xmin, xmax, ymin, ymax
            single_stamp = image.science()[xmin:xmax, ymin:ymax]
            single_stamp[np.isnan(single_stamp)] = 0.
            single_stamp[np.isinf(single_stamp)] = 0.
            stampImage += single_stamp
            singleImagesArray.append(single_stamp)
        else:
            single_stamp = np.zeros((stampWidth))
            singleImagesArray.append(single_stamp)
            
        i+=1
    return stampImage, singleImagesArray

def clusterResults(results, dbscan_args=None):
    
    """
    Use scikit-learn algorithm of density-based spatial clustering of
    applications with noise (DBSCAN)
    (http://scikit-learn.org/stable/modules/generated/
    sklearn.cluster.DBSCAN.html)
    to cluster the results of the likelihood image search using starting
    location, total velocity and slope of trajectory.
    Parameters
    ----------
    results: numpy recarray, required
    The results output from findObjects in searchImage.
    dbscan_args: dict, optional
    Additional arguments for the DBSCAN instance. See options in link
    above.
    Returns
    -------
    db_cluster: DBSCAN instance
    DBSCAN instance with clustering completed. To get cluster labels use
    db_cluster.labels_
    top_vals: list of integers
    The indices in the results array where the most likely object in each
    cluster is located.
    """

    default_dbscan_args = dict(eps=0.03, min_samples=1, n_jobs=-1)

    if dbscan_args is not None:
        default_dbscan_args.update(dbscan_args)
    dbscan_args = default_dbscan_args

    slope_arr = []
    intercept_arr = []
    t0x_arr = []
    t0y_arr = []
    vel_total_arr = []
    vx_arr = []
    vel_x_arr = []
    vel_y_arr = []

    for target_num in range(len(results)):
        
        t0x = results['t0_x'][target_num]
        t0x_arr.append(t0x)
        t0y = results['t0_y'][target_num]
        t0y_arr.append(t0y)
        v0x = results['v_x'][target_num]
        vel_x_arr.append(v0x)
        v0y = results['v_y'][target_num]
        vel_y_arr.append(v0y)
        
    db_cluster = DBSCAN(**dbscan_args)
        
    scaled_t0x = np.array(t0x_arr) #- np.min(t0x_arr)
    if np.max(scaled_t0x) > 0.:
        scaled_t0x = scaled_t0x/4200.#np.max(scaled_t0x)
    scaled_t0y = np.array(t0y_arr) #- np.min(t0y_arr)
    if np.max(scaled_t0y) > 0.:
        scaled_t0y = scaled_t0y/4200.#np.max(scaled_t0y)
    scaled_vx = np.array(vel_x_arr)# - np.min(vel_x_arr)
    if np.max(scaled_vx) > 0.:
        scaled_vx /= np.max(scaled_vx)
    scaled_vy = np.array(vel_y_arr)# - np.min(vel_y_arr)
    if np.max(scaled_vy) > 0.:
        scaled_vy /= np.max(scaled_vy)
        
    db_cluster.fit(np.array([scaled_t0x, scaled_t0y,
                             scaled_vx, scaled_vy], dtype=np.float).T)
    
    top_vals = []
    for cluster_num in np.unique(db_cluster.labels_):
        cluster_vals = np.where(db_cluster.labels_ == cluster_num)[0]
        top_vals.append(cluster_vals[0])
        
    return db_cluster, top_vals

#http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
def kalman_filter(obs, var):

    xhat = np.zeros(len(obs))
    P = np.zeros(len(obs))
    xhatminus = np.zeros(len(obs))
    Pminus = np.zeros(len(obs))
    K = np.zeros(len(obs))
    
    Q = 0
    R = np.copy(var)
    R[R==0.] = 100.
    
    xhat[0] = obs[0]
    P[0] = R[0]

    for k in range(1,len(obs)):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        K[k] = Pminus[k] / (Pminus[k] + R[k])
        xhat[k] = xhatminus[k] + K[k]*(obs[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
        
    return xhat, P

def calc_nu(psi_vals, phi_vals):

    not_0 = np.where(psi_vals > -9000.)[0]

    if len(not_0) > 0:
        return np.sum(psi_vals[not_0])/np.sqrt(np.sum(phi_vals[not_0]))
    else:
        return 0.

def create_filter_curve(ps_lc, ph_lc, lc_on):
    num_stamps = len(ps_lc[0])
    lc_list = []
    
    for ps_lc_line, ph_lc_line in zip(ps_lc, ph_lc):
        
        filter_sums = []
        #for ps_lc_line, ph_lc_line in zip(ps_lc_list, ph_lc_list):
    
        fluxes = ps_lc_line/ph_lc_line
        #use_f = np.where(ps_lc_line != 0.)
        fluxes = fluxes#[use_f]
        f_var = np.var(fluxes)*np.ones(len(fluxes))
        kalman_flux, kalman_error = kalman_filter(fluxes, f_var)
        deviations = np.abs(kalman_flux - fluxes) / kalman_error**.5
        #print deviations
        keep_idx = np.where(deviations < 1.)[0]
        
        kalman_flux, kalman_error = kalman_filter(fluxes[::-1], f_var[::-1])
        deviations = np.abs(kalman_flux - fluxes[::-1]) / kalman_error**.5
        #print deviations
        keep_idx_back = np.where(deviations < 1.)[0]
        
        if len(keep_idx) >= len(keep_idx_back):
            single_stamps = fluxes[keep_idx]
            #print keep_idx
        else:
            keep_idx = num_stamps-1 - keep_idx_back
            single_stamps = fluxes[keep_idx]
            #print keep_idx
        filter_sums = calc_nu(ps_lc_line[keep_idx], ph_lc_line[keep_idx])
        lc_list.append(filter_sums)
        
        #return filter_stamps
    return (lc_on, lc_list)

def return_filter_curve(ps_lc_line, ph_lc_line):
    num_stamps = len(ps_lc_line)
    #filter_sums = []
    filter_stamps = []
    #for ps_lc_line, ph_lc_line in zip(ps_lc_list, ph_lc_list):
    
    fluxes = ps_lc_line/ph_lc_line
    #use_f = np.where(ps_lc_line != 0.)
    fluxes = fluxes#[use_f]
    f_var = np.var(fluxes)*np.ones(len(fluxes))
    kalman_flux, kalman_error = kalman_filter(fluxes, f_var)
    deviations = np.abs(kalman_flux - fluxes) / kalman_error**.5
    #print deviations
    keep_idx = np.where(deviations < 1.)[0]
    
    kalman_flux, kalman_error = kalman_filter(fluxes[::-1], f_var[::-1])
    deviations = np.abs(kalman_flux - fluxes[::-1]) / kalman_error**.5
    #print deviations
    keep_idx_back = np.where(deviations < 1.)[0]
    
    if len(keep_idx) >= len(keep_idx_back):
        single_stamps = fluxes[keep_idx]
        filter_stamps.append(single_stamps)
        kalman_flux, kalman_error = kalman_filter(fluxes, f_var)
        kf_b, kf_e = kalman_filter(fluxes[::-1], f_var[::-1])
        #print keep_idx
    else:
        keep_idx = num_stamps-1 - keep_idx_back
        single_stamps = fluxes[keep_idx]
        filter_stamps.append(single_stamps)
        kalman_flux, kalman_error = kalman_filter(fluxes[::-1], f_var[::-1])
        kf_b, kf_e = kalman_filter(fluxes, f_var)
        #print keep_idx
        #filter_sums = calc_nu(ps_lc_line[keep_idx], ph_lc_line[keep_idx])
        
    return filter_stamps, keep_idx#, kalman_flux, kalman_error, kf_b, kf_e, fluxes

def get_likelihood_lcs(results_arr, psi, phi, image_times):
    ps_lc = np.zeros((len(image_times), len(results_arr)))
    ph_lc = np.zeros((len(image_times), len(results_arr)))
    print('Building Lightcurves')
    for idx, t_current in list(enumerate(image_times)):
        #print(idx)
        x0 = results_arr['t0_x'] + results_arr['v_x']*t_current
        y0 = results_arr['t0_y'] + results_arr['v_y']*t_current
        x0_0 = x0
        y0_0 = y0
        x0_0 = np.array(x0_0, dtype=np.int)
        y0_0 = np.array(y0_0, dtype=np.int)
        x0_0[np.where(((x0_0 > 4199) | (x0_0 < 0)))] = 4199
        y0_0[np.where(((y0_0 > 4199) | (y0_0 < 0)))] = 4199
        psi_on = psi[idx]
        phi_on = phi[idx]
        psi_on[4199, 4199] = 0.
        psi_on[np.isnan(psi_on)] = 0.
        psi_on[np.where(psi_on < -9000)] = 0.
        phi_on[np.where(phi_on < -9000)] = 999999.
        phi_on[np.where(phi_on == 0.)] = 999999.
        ps_lc[idx] = psi_on[y0_0, x0_0]
        #print(p_o)
        ph_lc[idx] = phi_on[y0_0, x0_0]
    return ps_lc.T, ph_lc.T
