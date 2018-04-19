import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from astropy.io import fits
from collections import OrderedDict

from create_stamps import create_stamps
from sklearn.neighbors import NearestNeighbors

def match_fakes(fake_df, res_df):
    fake_coords = np.array([fake_df['x'].values, fake_df['y'].values,
                            fake_df['x_final'].values, fake_df['y_final'].values], dtype=np.int).T
    res_coords = np.array([res_df['x'].values, res_df['y'].values,
                           res_df['x_final'].values, res_df['y_final'].values], dtype=np.int).T

    n_neigh = NearestNeighbors(n_neighbors=1)
    n_neigh.fit(fake_coords)
    dist, near = n_neigh.kneighbors(res_coords)
    return near.flatten(), dist.flatten()

def compare_fakes(image_dir, results_dir, results_suffix_list, time_file):

    keep_fake_df = None
    found_fake_df = None
    found_res_df = None
    
    for results_suffix in results_suffix_list:

        print(results_suffix)

        im_dir = os.path.join(image_dir, results_suffix)
        
        image_list = sorted(os.listdir(im_dir))
        image_list = [os.path.join(im_dir, im_name) for im_name in image_list]

        visit_list = np.array([image_name[-14:-8] for image_name in image_list], dtype=np.int)
        visit_nums, visit_times = np.genfromtxt(time_file, unpack=True)
        image_time_dict = OrderedDict()

        for visit_num, visit_time in zip(visit_nums, visit_times):
            image_time_dict[str(int(visit_num))] = visit_time
        image_mjd = np.array([image_time_dict[str(visit_id)] for visit_id in visit_list])
        image_mjd = image_mjd[np.where((image_mjd > 57070) & (image_mjd < 57073))]
        image_times = image_mjd - image_mjd[0]

        stamper = create_stamps()
        
        times_filename = os.path.join(results_dir, 'times_%s.txt' % results_suffix)
        times_list = stamper.load_times(times_filename)

        lc_filename = os.path.join(results_dir, 'lc_%s.txt' % results_suffix)
        lc_list = stamper.load_lightcurves(lc_filename)

        stamp_filename = os.path.join(results_dir, 'ps_%s.txt' % results_suffix)
        stamps = stamper.load_stamps(stamp_filename)

        result_filename = os.path.join(results_dir, 'results_%s.txt' % results_suffix)
        results = stamper.load_results(result_filename)

        fake_df = pd.read_csv(os.path.join(results_dir, 'results_fakes_%s.txt' % results_suffix),
                              delimiter=' ', names=['x','y','xv','yv','flux','mag'], skiprows=1)

        hdulist = fits.open(image_list[0])
        x_size = hdulist[1].header['NAXIS1']
        y_size = hdulist[1].header['NAXIS2']

        fake_coords = np.array([fake_df['x'], fake_df['y']], dtype=np.int).T
        fake_vel = np.array([fake_df['xv'], fake_df['yv']]).T
        fake_traj_coords = []
        for f_c, f_v in zip(fake_coords, fake_vel):
            fake_traj_coords.append(np.array([f_c[0] + f_v[0]*image_times, f_c[1] + f_v[1]*image_times]).T)
        fake_traj_coords = np.array(fake_traj_coords, dtype=np.int)

        keep_idx = []
        unmasked_obs = []
        final_pos = []
        for idx, fake_traj in enumerate(list(fake_traj_coords)):
            rejections = 0
            for fk_coord in fake_traj:
                if ((fk_coord[0] >= x_size) | (fk_coord[1] >= y_size)):
                    rejections += 1
                elif hdulist[2].data[fk_coord[1], fk_coord[0]] == 256:
                    rejections += 1
            if (len(image_times) - rejections) >= 6:
                keep_idx.append(idx)
            unmasked_obs.append(len(image_times) - rejections)
            final_pos.append(fake_traj[-1])

        fake_df['unmasked_obs'] = unmasked_obs
        final_pos = np.array(final_pos).T
        fake_df['x_final'] = final_pos[0]
        fake_df['y_final'] = final_pos[1]
        fake_df['patch'] = results_suffix[:-7]

        if keep_fake_df is None:
            if len(keep_idx) > 0:
                keep_fake_df = fake_df.iloc[keep_idx]
        else:
            if len(keep_idx) > 0:
                keep_fake_df = pd.concat([keep_fake_df, fake_df.iloc[keep_idx]])

        patch_keep_df = fake_df.iloc[keep_idx]

        print(len(keep_idx))
        if len(keep_idx) == 0:
            continue

        if results.size == 0:
            continue

        keep_results = stamper.stamp_filter(stamps, 0.03)

        if len(keep_results) == 0:
            continue
        
        if results.size > 1:
            res_df = pd.DataFrame(results).iloc[keep_results]
        else:
            res_df = pd.DataFrame()
            for key_name in list(results.dtype.fields.keys()):
                res_df[key_name] = [results[key_name]]

        mag_results = []
        for res_idx in keep_results:
            mag_results.append(stamper.calc_mag(image_list, lc_list[res_idx], times_list[res_idx]))
        res_df['mag'] = mag_results
        res_df['patch'] = results_suffix[:-7]

        res_coords = np.array([res_df['x'], res_df['y']], dtype=np.int).T
        res_vel = np.array([res_df['vx'], res_df['vy']]).T
        res_final_coords = []
        for r_c, r_v in zip(res_coords, res_vel):
            res_final_coords.append(np.array([r_c[0] + r_v[0]*image_times[-1], r_c[1] + r_v[1]*image_times[-1]]))
        res_final_coords = np.array(res_final_coords, dtype=np.int).T

        res_df['x_final'] = res_final_coords[0]
        res_df['y_final'] = res_final_coords[1]

        near, dist = match_fakes(patch_keep_df, res_df)
        match_idx = np.where(dist <= 10.)[0]
        unique_fakes, unique_idx = np.unique(near[match_idx], return_index=True)

        if found_fake_df is None:
            if len(match_idx) > 0:
                found_fake_df = keep_fake_df.iloc[near[match_idx[unique_idx]]]
        else:
            if len(match_idx) > 0:
                found_fake_df = pd.concat([found_fake_df, patch_keep_df.iloc[near[match_idx[unique_idx]]]])

        if found_res_df is None:
            if len(match_idx) > 0:
                found_res_df = res_df.iloc[match_idx[unique_idx]]
        else:
            if len(match_idx) > 0:
                found_res_df = pd.concat([found_res_df, res_df.iloc[match_idx[unique_idx]]])

        print(len(unique_idx))

    keep_fake_df.to_csv(os.path.join(results_dir, 'keep_fake.csv'), index=False)
    found_fake_df.to_csv(os.path.join(results_dir, 'found_fake.csv'), index=False)
    found_res_df.to_csv(os.path.join(results_dir, 'found_res.csv'), index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', dest='im_dir')
    parser.add_argument('--results_dir', dest='results_dir')
    parser.add_argument('--results_suffix',
                        dest='results_suffix')
    parser.add_argument('--time_file', dest='time_file')

    args = parser.parse_args()
    im_dir = args.im_dir
    results_dir = args.results_dir
    results_suffix = args.results_suffix
    time_file = args.time_file

    if results_suffix is None:
        results_suffix_list = os.listdir(im_dir)
    else:
        results_suffix_list = [results_suffix]

    compare_fakes(im_dir, results_dir, results_suffix_list, time_file)
