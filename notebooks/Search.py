
from __future__ import print_function
from astropy.io import fits
import numpy as np
from PIL import Image
import os
import subprocess
from sklearn.cluster import DBSCAN
from analyzeImage import analyzeImage

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def search_images(path, run_name, 
                  mask_threshold=None, mask_penalty=None, psf_sigma=None,
                  angles_to_search=None, min_angle=None, max_angle=None, 
                  velocities_to_search=None, min_velocity=None, max_velocity=None,
                  write_images=None):
        
    gpu_code_path = "../code/gpu/"
    #real_image_name = "chip_1"
    #real_image_path = gpu_code_path+"images/"+real_image_name
    psi_image_path = gpu_code_path+"output-images/psi"
    phi_image_path = gpu_code_path+"output-images/phi"
    
    
    if mask_threshold is None:
        mask_threshold=0.75
    if mask_penalty is None:
        mask_penalty=-0.02
    if psf_sigma is None: 
        psf_sigma=1.0
    if angles_to_search is None:
        angles_to_search=90
    if min_angle is None:
        min_angle=0.0
    if max_angle is None: 
        max_angle=6.283
    if velocities_to_search is None:
        velocities_to_search=25
    if min_velocity is None:
        min_velocity=200.0
    if max_velocity is None:
        max_velocity=500
    if write_images is None:
        write_images=0
    
    paramsText =  """Debug ................ : 1
    Image Count .......... : 15
    Generate Images ...... : 0
    Image Width .......... : 100
    Image Height ......... : 100
    PSF Sigma ............ : {psf}
    Object Brightness .... : 50.0
    Object Initial x ..... : 50.0
    Object Initial y ..... : 58.0
    Velocity x ........... : -1.33
    Velocity y ........... : -0.2
    Background Level ..... : 1024.0
    Background Sigma ..... : 4.0
    Mask Threshold ....... : {mask_t}
    Mask Penalty ......... : {mask_p}
    Angles to Search ..... : {angle_count}
    Minimum Angle ........ : {min_a}
    Maximum Angle ........ : {max_a}
    Velocities to Search . : {vel_count}
    Minimum Velocity ..... : {min_v}
    Maximum Velocity ..... : {max_v}
    Psi/Phi to file ...... : {write_imgs}
    Source Images Path ... : ../../{source}/
    Psi Images Path ...... : ../../{psi}/
    Phi Images Path....... : ../../{phi}/
    Results Path ......... : ../../../data/results/{name}.txt
    """.format( psf=psf_sigma,
              mask_t=mask_threshold, mask_p=mask_penalty, 
              angle_count=angles_to_search, min_a=min_angle, max_a=max_angle, 
              vel_count=velocities_to_search, min_v=min_velocity, max_v=max_velocity,source=path,
              psi=psi_image_path, phi=phi_image_path, write_imgs=write_images, name=run_name)
    
    paramsFile = open('../code/gpu/debug/parameters.config', 'w')
    paramsFile.write(paramsText)
    paramsFile.close()
    
    paramsFile = open('../data/results/{name}.config'.format(name=run_name), 'w')
    paramsFile.write(paramsText)
    paramsFile.close()

    popen = subprocess.Popen( "./clearImages.sh", stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
    popen.wait()
    output = popen.stderr.read()
    output += popen.stdout.read()
    print( output)

    for path in execute("./search.sh"):
        print(path, end="")


def create_stamps( results, images_path, count=1500 ):

 #   raw_results = np.genfromtxt(results_path, names=True)
 #   results = raw_results[0:120000]
    
    
    image_mjd = []
    for filename in sorted(os.listdir(images_path)):
        hdulist = fits.open(os.path.join(images_path, filename))
        image_mjd.append(hdulist[0].header['MJD'])

    image_mjd = np.array(image_mjd)
    image_times = image_mjd - image_mjd[0]
    #image_times*=24.
    
    hdulist = fits.open(os.path.join(images_path, os.listdir(images_path)[0]))
    num_images = len(os.listdir(images_path))
    image_shape = np.shape(hdulist[1].data)
    im_array = np.zeros((num_images, image_shape[0], image_shape[1]))

    for idx, filename in list(enumerate(sorted(os.listdir(images_path)))):

        image_file = os.path.join(images_path, filename)
        hdulist = fits.open(image_file)
        im_array[idx] = hdulist[1].data#*mask


    create_files = True
    maxshow = count if create_files else 75
    f_results = results#results #filtered_results
    imgs = im_array
    for imNum in range(min(len(f_results), maxshow)):
        cr = f_results[imNum]
        if create_files:
            arr = ai.createPostageStamp(imgs,
                list(f_results[['t0_x', 't0_y']][imNum]),
                np.array(list(f_results[['v_x', 'v_y']][imNum])),
                image_times, [25., 25.])[0]
            arr -= arr.min()
            arr *= (255.0/arr.max())
            im = Image.fromarray(arr.astype(np.uint8))
            im.save("../data/stamps/c"+images_path[-1]+
                "p"+str(int(cr[0]))+"_"+str(int(cr[1]))+
                "v"+str(int(cr[4]))+"_"+str(int(cr[5]))+".tif")
        else:
            plt.imshow(ai.createPostageStamp(imgs,
                list(f_results[['t0_x', 't0_y']][imNum]),
                np.array(list(f_results[['v_x', 'v_y']][imNum])),
                image_times, [25., 25.])[0],
                origin='lower',
                #cmap=plt.cm.Greys_r,
                interpolation='None')
            plt.title(str('#' + str(imNum+1) + ' [x,y] = '
                    + str(list(f_results[['t0_x', 't0_y']][imNum])))
                      + ' v = ' + str(list(f_results[['v_x', 'v_y']][imNum])))
            plt.show()


def result_quality( results ):
    pass


