import unittest

import numpy as np

import kbmod.search as kb
from kbmod.fake_data_creator import *


class test_readme_example(unittest.TestCase):
    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_make_and_copy(self):
        # Create a point spread function
        psf = kb.PSF(1.5)

        # Create fake data with ten 512x512 pixel images.
        ds = FakeDataSet(512, 512, 10)
        imgs = ds.stack.get_images()

        # Get the timestamp of the first image.
        t0 = imgs[0].get_obstime()

        # Specify an artificial object
        flux = 275.0
        position = (10.7, 15.3)
        velocity = (2, 0)

        # Inject object into images
        for im in imgs:
            dt = im.get_obstime() - t0
            add_fake_object(
                im,
                position[0] + dt * velocity[0],
                position[1] + dt * velocity[1],
                flux,
                psf,
            )

        # Create a new image stack with the inserted object.
        stack = kb.ImageStack(imgs)

        # Recover the object by searching a set of trajectories.
        search = kb.stack_search(stack)
        search.search(
            5,  # Number of search angles to try (-0.1, -0.05, 0.0, 0.05, 0.1)
            5,  # Number of search velocities to try (0, 1, 2, 3, 4)
            -0.1,  # The minimum search angle to test
            0.1,  # The maximum search angle to test
            0,  # The minimum search velocity to test
            4,  # The maximum search velocity to test
            7,  # The minimum number of observations
        )

        # Get the top 10 results.
        results = search.get_results(0, 10)


if __name__ == "__main__":
    unittest.main()
