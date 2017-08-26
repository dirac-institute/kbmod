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

   results = search.get_results(0, par.results_count)

   results_clustered = [ results[i] for i in 
      cluster_trajectories(results,
      dbscan_args=dict(eps=par.cluster_eps, 
          n_jobs=-1, min_samples=1))[1] ]

   matched_results = []
   bad_results = []
   
   for r in results_clustered:
      if any(compare_trajectory_once(r, test, par.match_v, par.match_coord)
      for test in results_key):
         matched_results.append(r)
      else:
         bad_results.append(r)

   images = [i.science() for i in stack.get_images()]

   bad_stamps = [create_postage_stamp(images, t, stack.get_times(), 
   [par.stamp_dim, par.stamp_dim])[0] for t in bad_results]

   matched_stamps = [create_postage_stamp(images, t, stack.get_times(),
   [par.stamp_dim, par.stamp_dim])[0] for t in matched_results]

   return matched_stamps, bad_stamps
