import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt

def makeArray(text):
    return np.fromstring(text,sep=' ')

if __name__ == "__main__":

    chip = sys.argv[1]

    chip_stamp_arrays = []
    chip_lc = []
    chip_field_times = []
    chip_id = []
    
    chip_df = pd.DataFrame(columns=['t0_x', 't0_y', 'theta_par', 'theta_perp', 'v_x',
                                    'v_y', 'likelihood', 'est_flux', 'field_num', 'times'])
    i = 0
    for field_num in range(1,57):

        #if field_num < 10:
        #    field_str = '0%i' % field_num
        #else:
        #    field_str = str(field_num)
        field_str = str(field_num)

        try:
            data = pd.read_csv('results/data/%s_%s_results.csv' % (chip, field_str))
        except IOError:
            continue

#        print field_str
#        print data['times']
#        data['times'] = data['times'].apply(makeArray)
#        print data['times']

        if len(data) > 0:
#            print data['times'][0]
            field_stamps = np.genfromtxt('results/stamps/%s_%s_stamps.dat' % (chip, field_str))

            if len(np.shape(field_stamps)) < 2:
                field_stamps = [field_stamps]
            for stamp in field_stamps:
                stamp_array = np.array(stamp).reshape(25,25)
                chip_stamp_arrays.append(stamp_array)

            light_curves = np.genfromtxt('results/light_curves/%s_%s_lightcurves.dat' % (chip, field_str))
            if len(np.shape(light_curves)) < 2:
                light_curves = [light_curves]
            for lc in light_curves:
                chip_lc.append(lc)
                i+=1

            for row in range(len(data)):
                
                chip_field_times.append((data['times'][row]))
                chip_id.append(field_str)

            chip_df = chip_df.append(data)
#    print chip_df['likelihood'][:10]        
#    chip_df['likelihood'] = chip_df['likelihood'].apply(pd.to_numeric)
#    print chip_df['likelihood'][:10]
    print len(chip_field_times), len(chip_df), len(chip_df[chip_df['likelihood'] > 6.]), i, chip_field_times[0]
    fig = plt.figure(figsize=(8, 3*len(chip_id)))
    chip_df.to_csv('results/%s_full_results.csv' % chip)
    for lc, stamp, plot_num, image_time_set, likely in zip(chip_lc, chip_stamp_arrays,
                                                           np.arange(len(chip_id)),
                                                           chip_field_times, chip_df['likelihood']):
        image_times = [x for x in image_time_set[1:-1].split()]
        fig.add_subplot(len(chip_id),2,2*plot_num + 1)
        plt.imshow(stamp, origin='lower', interpolation='None')
        plt.title(str(chip_id[plot_num]))
        fig.add_subplot(len(chip_id),2,2*plot_num + 2)
        plt.plot(image_times, lc)
        plt.xlabel('Time (days)')
        plt.ylabel('Flux')
        plt.tight_layout()
    plt.savefig(str(str(chip) + '_stamps.pdf')) 
