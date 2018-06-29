import os
import sys
import numpy as np
import pandas as pd
from astropy.time import Time

class mpc_submission(object):

    def __init__(self):
        
        return

    def write_mpc_sub(self, filename, obj_id, new):

        year = []
        month = []
        day = []
        ra_hour = []
        ra_min = []
        ra_sec = []
        dec_deg = []
        dec_min = []
        dec_sec = []
        obs_code = []
        
        mag_path = '/Users/Bryce/Desktop/HITS/flux_test'
        mag_filename = os.path.join(mag_path, '%s_mags.dat' % filename)
        
        obs_path = '/Users/Bryce/Documents/astro_work/kbmod/orbfit2_0'
        observation_filename = os.path.join(obs_path, '%s.ast' % filename)
        
        time_path = '/Users/Bryce/Desktop/HITS'
        time_filename = os.path.join(time_path, 'image_times.dat')
        
        time_file = pd.read_csv(time_filename, delimiter=' ', names=['visit_num', 'image_mjd'], skiprows=1)
        
        g_mag_file = pd.read_csv(mag_filename, names=['visit_num', 'g_mag'])
        g_mag_dict = dict((str(key), val) for key, val in zip(g_mag_file['visit_num'].values, g_mag_file['g_mag'].values))

        with open(observation_filename, 'r') as f:
            for line in f:
                year.append(line[15:19])
                month.append(line[20:22])
                day.append(line[23:31])
                ra_hour.append(line[32:34])
                ra_min.append(line[35:37])
                ra_sec.append(line[38:44])
                dec_deg.append(line[44:47])
                dec_min.append(line[48:50])
                dec_sec.append(line[51:56])
                obs_code.append(line[77:80])

        with open('%s.txt' % obj_id, 'w') as g:                

            #create header
            g.write('COD W84\n')
            g.write('CON J. B. Kalmbach, Box 351560 Seattle, WA 98195\n')
            g.write('CON [brycek@uw.edu]\n')
            g.write('OBS F. Forster, J. C. Maureira, L. Galbany, T. de Jaeger\n')
            g.write('OBS J. Martinez, G. Pignata, G. Medina, S. Gonzalez, C. Smith\n')
            g.write('MEA J. B. Kalmbach, P. Whidden, A. J. Connolly\n')
            g.write('TEL 4.0-m reflector + CCD\n')
            g.write('NET Gaia1\n')
            g.write('BND g\n')
            g.write('ACK %s\n' % obj_id)
            g.write('AC2 brycek@uw.edu\n')

            for idx in range(len(year)):
                
                time_obj = Time('%s-%s-%s' % (year[idx], month[idx], day[idx][:2]), format='iso')
                time_obj.mjd += np.float(day[idx][2:])

                for time_idx in range(len(time_file['image_mjd'].values)):
                    if np.abs(time_obj.mjd - np.float(time_file['image_mjd'].values[time_idx])) < 0.0001:
                        try:
                            g_mag = np.float(g_mag_dict['%i' % time_file.iloc[time_idx]['visit_num']])
                        except KeyError:
                            continue
                        
                if new is True and idx == 0:
                    g.write('     kbm%04i* C%s %s %s %s %s %s%s %s %s         %5.2fg      W84\n' % (int(obj_id), year[idx], month[idx], day[idx], ra_hour[idx], 
                                                                                                    ra_min[idx], ra_sec[idx], dec_deg[idx], dec_min[idx], 
                                                                                                    dec_sec[idx], g_mag))        
                elif new is True:
                    g.write('     kbm%04i  C%s %s %s %s %s %s%s %s %s         %5.2fg      W84\n' % (int(obj_id), year[idx], month[idx], day[idx], ra_hour[idx], 
                                                                                                    ra_min[idx], ra_sec[idx], dec_deg[idx], dec_min[idx], 
                                                                                                    dec_sec[idx], g_mag))        
                else:
                    g.write('%5s         C%s %s %s %s %s %s%s %s %s         %5.2fg      W84\n' % (obj_id, year[idx], month[idx], day[idx], ra_hour[idx], 
                                                                                                  ra_min[idx], ra_sec[idx], dec_deg[idx], dec_min[idx], 
                                                                                                  dec_sec[idx], g_mag))        

if __name__ == "__main__":

    mpc_sub = mpc_submission()
    
    orb_list = os.listdir('/Users/Bryce/Documents/astro_work/kbmod/orbfit2_0/')
    stamp_list = [stamp_name[:-4] for stamp_name in orb_list if ((stamp_name.startswith('stamp_')) and stamp_name.endswith('.ast'))]
    obj_id = 1
    for stamp_name in stamp_list:
        mpc_sub.write_mpc_sub(stamp_name, str(obj_id), True)
        obj_id += 1
