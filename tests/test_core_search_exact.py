"""Test that the core search can perfectly recover an object with linear motion. This test turns off
all filtering and just checks the GPU search code."""

import unittest

from kbmod.configuration import SearchConfiguration
from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import add_fake_object, make_fake_layered_image
from kbmod.run_search import SearchRunner
from kbmod.search import *
from kbmod.trajectory_generator import VelocityGridSearch


@unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
class test_search_exact(unittest.TestCase):
    def test_core_search_exact(self):
        # image properties
        img_count = 20
        dim_y = 200
        dim_x = 300
        noise_level = 1.0
        variance = noise_level**2
        p = PSF.make_gaussian_kernel(1.0)

        # object properties -- The object is moving in a straight line
        object_flux = 250.0
        start_x = 70
        start_y = 45
        xvel = 40.0
        yvel = -10.0
        trj = Trajectory(
            x=start_x,
            y=start_y,
            vx=xvel,
            vy=yvel,
        )

        # create image set with single moving object
        imlist = []
        for i in range(img_count):
            time = i / img_count
            im = make_fake_layered_image(dim_x, dim_y, noise_level, variance, time, p, seed=i)
            add_fake_object(
                im,
                trj.get_x_pos(time),
                trj.get_y_pos(time),
                object_flux,
                p,
            )
            imlist.append(im)
        stack = ImageStack(imlist)

        # Turn off all filtering and use a custom trajectory generator that
        # tests 1681 velocities per pixel and includes the true velocity.
        config = SearchConfiguration()
        config.set("do_clustering", False)
        config.set("do_mask", False)
        config.set("lh_level", 0.0)
        config.set("num_obs", 1)
        config.set("sigmaG_lims", [5, 95])
        gen = VelocityGridSearch(41, -80.0, 80.0, 41, -20.0, 20.0)

        # Run the search.
        runner = SearchRunner()
        results = runner.do_core_search(config, stack, gen)
        self.assertGreater(len(results), 0)

        # Check that the best result is the true one.
        self.assertEqual(results["x"][0], start_x)
        self.assertEqual(results["y"][0], start_y)
        self.assertEqual(results["vx"][0], xvel)
        self.assertEqual(results["vy"][0], yvel)


if __name__ == "__main__":
    unittest.main()
