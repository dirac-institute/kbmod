import unittest

from kbmod.core.psf import PSF
from kbmod.trajectory_generator import KBMODV1Search
import kbmod.search as kb
from kbmod.fake_data.fake_data_creator import *


class test_readme_example(unittest.TestCase):
    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_make_and_copy(self):
        # Create a point spread function
        psf = PSF.make_gaussian_kernel(1.5)

        # Create fake data with ten 512x512 pixel images.
        fake_times = create_fake_times(10, t0=57130.2)
        ds = FakeDataSet(512, 512, fake_times)
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

        # Generate a set of trajectories to test from each pixel.
        gen = KBMODV1Search(
            5,  # Number of search velocities to try (0, 0.8, 1.6, 2.4, 3.2)
            0,  # The minimum search velocity to test (inclusive)
            4,  # The maximum search velocity to test (exclusive)
            5,  # Number of search angles to try (-0.1, -0.06, -0.02, 0.02, 0.6)
            -0.1,  # The minimum search angle to test (inclusive)
            0.1,  # The maximum search angle to test (exclusive)
        )
        candidates = [trj for trj in gen]

        # Do the actual search.
        search = kb.StackSearch(stack)
        search.set_min_obs(7)
        search.search_all(candidates)

        # Get the top 10 results.
        results = search.get_results(0, 10)


if __name__ == "__main__":
    unittest.main()
