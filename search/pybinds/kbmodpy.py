import kbmod
import numpy

def to_numpy(self, copy_data=False):
   if copy_data == None:
      copy_data = False
   return numpy.array( self.get_science(), copy=copy_data  )

kbmod.layered_image.array = to_numpy
