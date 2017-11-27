import kbmod
import numpy
import re

# layered image functions

def science_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_science(), copy=copy_data  )

def mask_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_mask(), copy=copy_data  )

def variance_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_variance(), copy=copy_data  )

def pool_science(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_science_pooled(), copy=copy_data  )

def pool_variance(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_variance_pooled(), copy=copy_data  )

kbmod.layered_image.science = science_to_numpy
kbmod.layered_image.mask = mask_to_numpy
kbmod.layered_image.variance = variance_to_numpy
kbmod.layered_image.pool_science = pool_science
kbmod.layered_image.pool_variance = pool_variance

# stack functions

def master_mask_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_master_mask(), copy=copy_data  )

def sciences_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return [ numpy.array( img, copy=copy_data  ) for img in self.get_sciences()]

def masks_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return [ numpy.array( img, copy=copy_data  ) for img in self.get_masks()]

def variances_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return [ numpy.array( img, copy=copy_data  ) for img in self.get_variances()]

kbmod.image_stack.master_mask = master_mask_to_numpy
kbmod.image_stack.sciences = sciences_to_numpy
kbmod.image_stack.masks = masks_to_numpy
kbmod.image_stack.variances = variances_to_numpy

# search functions

def psi_images_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return [ numpy.array( img, copy=copy_data ) for img in self.get_psi_images()]

def phi_images_to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return [numpy.array( img, copy=copy_data ) for img in self.get_phi_images()]

def lightcurve(self, t):
   psi = self.psi_stamps(t, 0)
   phi = self.phi_stamps(t, 0)
   psi = numpy.concatenate(psi)
   phi = numpy.concatenate(phi)
   return (psi,phi)   

kbmod.stack_search.get_psi = psi_images_to_numpy
kbmod.stack_search.get_phi = phi_images_to_numpy
kbmod.stack_search.lightcurve = lightcurve

# trajectory utilities

def compare_trajectory(a, b, v_thresh, pix_thresh):
    # compare flux too?
    if (b.obs_count == 0 and 
    abs(a.x-b.x)<=pix_thresh and 
    abs(a.y-b.y)<=pix_thresh and 
    abs(a.x_v-b.x_v)<v_thresh and 
    abs(a.y_v-b.y_v)<v_thresh):
        b.obs_count += 1
        return True
    else:
        return False

def match_trajectories(results_list, test_list, v_thresh, pix_thresh):
    matches = []
    unmatched = []
    for r in results_list:
        if any(compare_trajectory(r, test, v_thresh, pix_thresh)
        for test in test_list):
            matches.append(r)
    for t in test_list:
        if (t.obs_count == 0):
            unmatched.append(t)
        t.obs_count = 0
    return matches, unmatched

def score_results(results, test, v_thresh, pix_thresh):
    score = 0
    for t in range(len(test)):
        for i in range(len(results)):
            if (compare_trajectory(results[i], test[t], v_thresh, pix_thresh)):
                score += i
                test[t].obs_count = 0
                break
            if (i==len(results)-1):
                score += i
    return score/len(test)

def save_trajectories(t_list, path):
    if (len(t_list) == 0):
        return
    if (type(t_list[0]) == kbmod.traj_region):
        t_list = region_to_grid(t_list)
    with open(path, 'w+') as f:
        for t in t_list:
            f.write(str(t)+'\n')

def load_trajectories(path):
    t_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            t = kbmod.trajectory()
            t.lh = float(nums[0])
            t.flux = float(nums[1])
            t.x = int(float(nums[2]))
            t.y = int(float(nums[3]))
            t.x_v = float(nums[4])
            t.y_v = float(nums[5])
            t.obs_count = int(float(nums[6]))
            t_list.append(t)
    return t_list
    
def grid_to_region(t_list, duration):
    r_list = []
    for t in t_list:
        r = kbmod.traj_region()
        r.ix = t.x
        r.iy = t.y
        r.fx = t.x+t.x_v*duration
        r.fy = t.y+t.y_v*duration
        r.depth = 0
        r.obs_count = t.obs_count
        r.likelihood = t.lh
        r.flux = t.flux
        r_list.append(r)
    return r_list

def region_to_grid(r_list, duration):
    t_list = []
    for r in r_list:
        t = kbmod.trajectory()
        t.x = int(r.ix)
        t.y = int(r.iy)
        t.x_v = (r.fx-r.ix)/duration
        t.y_v = (r.fy-r.iy)/duration
        t.lh = r.likelihood
        t.flux = r.flux
        t.obs_count = r.obs_count
        t_list.append(t)
    return t_list

kbmod.save_trajectories = save_trajectories
kbmod.load_trajectories = load_trajectories
kbmod.grid_to_region = grid_to_region
kbmod.region_to_grid = region_to_grid
kbmod.match_trajectories = match_trajectories
kbmod.score_results = score_results

# constants
kbmod.__version__ = "0.3.4"
kbmod.pool_max = 1
kbmod.pool_min = 0
kbmod.no_data = -9999.0
