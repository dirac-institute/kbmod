from kbmodpy import kbmod as kb
from trajectoryFiltering import *
import numpy as np
import random as rd
import math
import os

class Bunch(object):
   def __init__(self, adict):
      self.__dict__.update(adict)

def run_search(par, run_number):

   par = Bunch(par)

   files = os.listdir(par.path)

   files.sort()
   files = [par.path+f for f in files]
   files = files[:par.max_images]

   images = [kb.layered_image(f) for f in files]

   p = kb.psf(par.psf_width)
   angle_range = par.angle_range
   velocity_range = par.velocity_range

   results_key = []
   for _ in range(par.object_count):
      traj = kb.trajectory()
      traj.x = int(rd.uniform(*par.x_range))
      traj.y = int(rd.uniform(*par.y_range))
      ang = rd.uniform(*angle_range)
      vel = rd.uniform(*velocity_range)
      traj.x_v = vel*math.cos(ang)
      traj.y_v = vel*math.sin(ang)
      traj.flux = rd.uniform(*par.flux_range)
      results_key.append(traj)

   results_key.extend(par.real_results)

   for t in results_key:
      add_trajectory(images, t, p)

   stack = kb.image_stack(images)

   stack.apply_mask_flags(par.flags, par.flag_exceptions)
   stack.apply_master_mask(par.master_flags, 2)

   search = kb.stack_search(stack, p)

   search_angle_r = (angle_range[0]/par.search_margin, 
                     angle_range[1]*par.search_margin)
   search_velocity_r = (velocity_range[0]/par.search_margin,
                        velocity_range[1]*par.search_margin)
   search.gpu(par.angle_steps, par.velocity_steps, 
      *search_angle_r, *search_velocity_r, par.min_observations)

   #results = search.get_results(0, par.results_count)
   search.save_results(par.results_file_path+'run{0:03d}.txt'.format(run_number+1), 0.03)

   if par.save_science:
      images = stack.sciences()
      for i in range(len(images)):
         np.save(par.img_save_path+'R{0:03d}'.format(run_number+1)+'SCI{0:03d}.npy'.format(i+1), images[i])

   return results_key
