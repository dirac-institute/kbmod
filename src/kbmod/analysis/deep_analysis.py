import argparse
import os
import sys
import tempfile
from pathlib import Path

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import numpy as np

def get_image_file_info(data_dir):
    times = []
    wcs = None
    
    print("\nLoading image file data:")
    
    filenames = sorted(os.listdir(data_dir))
    for i, fname in enumerate(filenames):
        full_name = f"{data_dir}/{fname}"
        with fits.open(full_name) as f:
            # Extract the time.
            epoch = Time(f[0].header["DATE-AVG"], format="isot")
            times.append(epoch.mjd)
            
            # Extract the WCS.
            if i == 0:
                wcs = WCS(f[1].header)
            
            # Print out the details
            sky = wcs.pixel_to_world(0, 0)
            sky2 = wcs.pixel_to_world(2048, 4096)
            print("%f: (%f, %f) to (%f, %f)" % (epoch.mjd, sky.ra.deg, sky.dec.deg, sky2.ra.deg, sky2.dec.deg))
            
    return times, wcs

# Load a fits file that maps the expnum to mjd.
def get_expnum_to_image_index(times):
    print("\nLoading map of EXPNUM to image index.")

    num_times = len(times)
    best_mjd_for_index = [0.0] * num_times
    best_expnum_for_index = [-1] * num_times
    with fits.open("./deepb1.exposures.positions.fits") as hdu_list:
        for row in hdu_list[1].data:
            expnum = row["EXPNUM  "]
            mjd = row["mjd_mid "]
            
            for i in range(num_times):
                if mjd < times[i] and mjd > best_mjd_for_index[i]:
                    best_mjd_for_index[i] = mjd
                    best_expnum_for_index[i] = expnum

    # For each time in mjds find the closest in mjd_expnum.
    # This is linear right now, but we can use binary search if needed.
    expnum_to_index = {}
    for i in range(num_times):
        expnum_to_index[best_expnum_for_index[i]] = i

    return expnum_to_index


def load_deep_known_objects(wcs, ccd_name, expnum_map):
    ccdnums =  {'S29': 1, 'S30':  2, 'S31':  3, 'S25':  4, 'S26':  5, 'S27':  6, 'S28':  7, 'S20':  8, 'S21':  9, 'S22':  10, 
    'S23': 11, 'S24':  12, 'S14':  13, 'S15':  14, 'S16':  15, 'S17':  16, 'S18':  17, 'S19':  18, 'S8':  19, 'S9':  20, 
    'S10': 21, 'S11':  22, 'S12':  23, 'S13':  24, 'S1':  25, 'S2':  26, 'S3':  27, 'S4':  28, 'S5':  29, 'S6':  30, 
    'S7':  31, 'N1':  32, 'N2':  33, 'N3':  34, 'N4':  35, 'N5':  36, 'N6':  37, 'N7':  38, 'N8':  39, 'N9':  40, 
    'N10': 41, 'N11':  42, 'N12':  43, 'N13':  44, 'N14':  45, 'N15':  46, 'N16':  47, 'N17':  48, 'N18':  49, 
    'N19': 50, 'N20':  51, 'N21':  52, 'N22':  53, 'N23':  54, 'N24':  55, 'N25':  56, 'N26':  57, 'N27':  58, 'N28':  59, 'N29':  60, 'N30':  61, 'N31':  62}
    ccd_n = ccdnums[ccd_name]
    
    print("Finding object occurences in the data files.")
    
    num_times = len(expnum_map)
    objects = {}

    with fits.open("/epyc/users/smotherh/kbmod_paper_DEEP/Data/deep_b1.observations.fits") as hdu_list:
        data = hdu_list[1].data
        for i, row in enumerate(data):
            if row['CCDNUM  '] == ccd_n and row['EXPNUM  '] in expnum_map:
                obj_name = row['ORBITID ']
                time_idx = expnum_map[row['EXPNUM  ']]
                r = row['RA      ']
                d = row['DEC     ']
                
                if obj_name not in objects:
                    objects[obj_name] = [None] * num_times
                objects[obj_name][time_idx] = (r, d)
    return objects
                
if __name__ == "__main__":
    ccd_name = "S1"
    group_name = "B1e_20211006"
    data_dir = f"/epyc/projects3/smotherh/DEEP/warps/{group_name}/{ccd_name}"
    times, wcs = get_image_file_info(data_dir)
    expnum_map = get_expnum_to_image_index(times)

    objects = load_deep_known_objects(wcs, ccd_name, expnum_map)
    for name in objects:
        print(f"{name}: {objects[name]}")
    