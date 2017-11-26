import kbmod
import numpy

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
    abs(a.x_v/b.x_v-1)<v_thresh and 
    abs(a.y_v/b.y_v-1)<v_thresh):
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

def save_trajectories(t_list, path):
    pass

def load_trajectories(path):
    pass

def grid_to_region(t_list, duration):
    pass

def region_to_grid(t_list, duration):
    pass

kbmod.match_trajectories = match_trajectories

# constants
kbmod.__version__ = "0.3.3"
kbmod.pool_max = 1
kbmod.pool_min = 0
kbmod.no_data = -9999.0
