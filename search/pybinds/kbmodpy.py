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

kbmod.layered_image.science = science_to_numpy
kbmod.layered_image.mask = mask_to_numpy
kbmod.layered_image.variance = variance_to_numpy

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

kbmod.stack_search.get_psi = psi_images_to_numpy
