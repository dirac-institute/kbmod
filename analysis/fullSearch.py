from kbmodpy import kbmod as kb
from trajectoryFiltering import *
import numpy as np
import random as rd
import math
import os

class Bunch(object):
   def __init__(self, adict):
      self.__dict__.update(adict)

def run_search(par):

   par = Bunch(par)

   files = os.listdir(par.path)

   files.sort()
   files = [par.path+f for f in files]
   files = files[:par.max_images]
   files 

   images = [kb.layered_image(f) for f in files]

   p = kb.psf(par.psf_width)
   angle_range = tuple( math.atan(yv/xv) \
      for xv,yv in zip(par.xv_range, par.yv_range))
   velocity_range = tuple( math.sqrt(xv**2+yv**2) \
      for xv,yv in zip(par.xv_range, par.yv_range))

   results_key = []
   for _ in range(par.object_count):
      traj = kb.trajectory()
      traj.x = int(rd.uniform(*par.x_range))
      traj.y = int(rd.uniform(*par.y_range))
      traj.x_v = rd.uniform(*par.xv_range)
      traj.y_v = rd.uniform(*par.yv_range)
      traj.flux = rd.uniform(*par.flux_range)
      results_key.append(traj)

   results_key.extend(par.real_results)

   for t in results_key:
      add_trajectory(images, t, p)

   stack = kb.image_stack(images)

   stack.apply_mask_flags(par.flags, par.flag_exceptions)
   stack.apply_master_mask(par.master_flags, 2)

   search = kb.stack_search(stack, p)

   search.gpu(par.angle_steps, par.velocity_steps, 
      *angle_range, *velocity_range, par.min_observations)

   results = search.get_results(0, par.results_count)

   results_clustered = [ results[i] for i in 
      cluster_trajectories(results,
      dbscan_args=dict(eps=par.cluster_eps, min_samples=1))[1] ]

   results_matched, results_unmatched = \
      match_trajectories(results_clustered,
      results_key, par.match_v, par.match_coord)
  
   results_to_plot = results_unmatched

   images = [i.science() for i in stack.get_images()]
   stamps = [create_postage_stamp(images, t, stack.get_times(), [21,21])[0] \
      for t in results_to_plot]

   return results_matched, results_unmatched, stamps
