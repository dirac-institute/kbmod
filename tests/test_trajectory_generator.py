import numpy as np
import unittest

from astropy.wcs import WCS

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.trajectory_generator import (
    KBMODV1Search,
    KBMODV1SearchConfig,
    EclipticCenteredSearch,
    SingleVelocitySearch,
    RandomVelocitySearch,
    VelocityGridSearch,
    create_trajectory_generator,
)
from kbmod.work_unit import WorkUnit


class test_trajectory_generator(unittest.TestCase):
    def test_SingleVelocitySearch(self):
        gen = SingleVelocitySearch(10.0, 5.0)
        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 1)
        self.assertEqual(trjs[0].vx, 10.0)
        self.assertEqual(trjs[0].vy, 5.0)

    def test_VelocityGridSearch(self):
        gen = VelocityGridSearch(3, 0.0, 2.0, 3, -0.25, 0.25)
        expected_x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
        expected_y = [-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25]

        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 9)
        for i in range(9):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test that we get the correct results if we dump to a table.
        tbl = gen.to_table()
        self.assertEqual(len(tbl), 9)
        for i in range(9):
            self.assertAlmostEqual(tbl["vx"][i], expected_x[i], delta=0.001)
            self.assertAlmostEqual(tbl["vy"][i], expected_y[i], delta=0.001)

        # Test invalid number of steps or ranges.
        self.assertRaises(ValueError, VelocityGridSearch, 3, 0.0, 2.0, 0, -0.25, 0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 0, 0.0, 2.0, 3, -0.25, 0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 3, 0.0, 2.0, 3, 0.25, -0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 3, 2.0, 0.0, 3, -0.25, 0.25)

    def test_KBMODV1Search(self):
        # Note that KBMOD v1's legacy search will never include the upper bound of angle or velocity.
        gen = KBMODV1Search(3, 0.0, 3.0, 2, -0.25, 0.25)
        expected_x = [0.0, 0.9689, 1.9378, 0.0, 1.0, 2.0]
        expected_y = [0.0, -0.247, -0.4948, 0.0, 0.0, 0.0]

        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 6)
        for i in range(6):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test that we get the correct results if we dump to a table.
        tbl = gen.to_table()
        self.assertEqual(len(tbl), 6)
        for i in range(6):
            self.assertAlmostEqual(tbl["vx"][i], expected_x[i], delta=0.001)
            self.assertAlmostEqual(tbl["vy"][i], expected_y[i], delta=0.001)

        # Test invalid number of steps and ranges.
        self.assertRaises(ValueError, KBMODV1Search, 3, 0.0, 3.0, 0, -0.25, 0.25)
        self.assertRaises(ValueError, KBMODV1Search, 0, 0.0, 3.0, 2, -0.25, 0.25)
        self.assertRaises(ValueError, KBMODV1Search, 3, 0.0, 3.0, 2, 0.25, -0.25)
        self.assertRaises(ValueError, KBMODV1Search, 3, 3.5, 3.0, 2, -0.25, 0.25)

    def test_EclipticCenteredSearch(self):
        gen = EclipticCenteredSearch(
            [0.0, 2.0, 3], [-45.0, 45.0, 3], angle_units="degrees", given_ecliptic=0.0
        )
        expected_x = [0.0, 0.707107, 1.41421, 0.0, 1.0, 2.0, 0.0, 0.707107, 1.41421]
        expected_y = [0.0, -0.707107, -1.41421, 0.0, 0.0, 0.0, 0.0, 0.707107, 1.41421]

        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 9)
        for i in range(9):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test that we get the correct results if we dump to a table.
        tbl = gen.to_table()
        self.assertEqual(len(tbl), 9)
        for i in range(9):
            self.assertAlmostEqual(tbl["vx"][i], expected_x[i], delta=0.001)
            self.assertAlmostEqual(tbl["vy"][i], expected_y[i], delta=0.001)

        # Test invalid number of steps and ranges.
        self.assertRaises(ValueError, EclipticCenteredSearch, [0.0, 3.0, 3], [-0.25, 0.25, 0])
        self.assertRaises(ValueError, EclipticCenteredSearch, [0.0, 3.0, 0], [-0.25, 0.25, 2])
        self.assertRaises(ValueError, EclipticCenteredSearch, [3.5, 3.0, 3], [-0.25, 0.25, 2])

    def test_KBMODV1SearchConfig(self):
        # Note that KBMOD v1's search will never include the upper bound of angle or velocity.
        gen = KBMODV1SearchConfig([0.0, 3.0, 3], [0.25, 0.25, 2], average_angle=0.0)
        expected_x = [0.0, 0.9689, 1.9378, 0.0, 1.0, 2.0]
        expected_y = [0.0, -0.247, -0.4948, 0.0, 0.0, 0.0]

        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 6)
        for i in range(6):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

    def test_RandomVelocitySearch(self):
        gen = RandomVelocitySearch(0.0, 2.0, -0.25, 0.25)

        # Try at least 1000 iterations and make sure it is still generating.
        for itr in range(1000):
            trj = next(gen)
            self.assertGreaterEqual(trj.x, 0.0)
            self.assertLessEqual(trj.x, 2.0)
            self.assertGreaterEqual(trj.y, -0.25)
            self.assertLessEqual(trj.y, 0.25)

        # Generate a single additional candidate.
        gen2 = RandomVelocitySearch(0.0, 2.0, -0.25, 0.25, max_samples=1)
        self.assertEqual(len([trj for trj in gen2]), 1)

        # No more samples to generate
        self.assertEqual(len([trj for trj in gen2]), 0)

        # Generate a twenty more candidates.
        gen2.reset_sample_count(20)
        self.assertEqual(len([trj for trj in gen2]), 20)

    def test_create_trajectory_generator(self):
        config1 = {
            "name": "VelocityGridSearch",
            "vx_steps": 10,
            "min_vx": 0,
            "max_vx": 5,
            "vy_steps": 20,
            "min_vy": -5,
            "max_vy": 15,
        }
        gen1 = create_trajectory_generator(config1)
        self.assertTrue(type(gen1) is VelocityGridSearch)
        self.assertEqual(gen1.vx_steps, 10)
        self.assertEqual(gen1.min_vx, 0)
        self.assertEqual(gen1.max_vx, 5)
        self.assertEqual(gen1.vy_steps, 20)
        self.assertEqual(gen1.min_vy, -5)
        self.assertEqual(gen1.max_vy, 15)

        config2 = {"name": "SingleVelocitySearch", "vx": 1, "vy": 2}
        gen2 = create_trajectory_generator(config2)
        self.assertTrue(type(gen2) is SingleVelocitySearch)
        self.assertEqual(gen2.vx, 1)
        self.assertEqual(gen2.vy, 2)

        # Create a fake work unit with one image and a WCS with a non-zero ecliptic angle.
        fake_wcs = WCS(naxis=2)
        fake_wcs.wcs.crpix = [0.0, 0.0]
        fake_wcs.wcs.cdelt = np.array([-0.1, 0.1])
        fake_wcs.wcs.crval = [0, -90]
        fake_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        fake_wcs.wcs.crota = np.array([0.0, 0.0])

        fake_data = FakeDataSet(10, 10, [0.0])
        base_config = SearchConfiguration()
        fake_work = WorkUnit(im_stack=fake_data.stack, config=base_config, wcs=fake_wcs)
        fake_ecliptic = fake_work.compute_ecliptic_angle()
        self.assertGreater(fake_ecliptic, 1.0)

        # Test we can create a trajectory generator with optional keyword parameters.
        config3 = {
            "name": "EclipticCenteredSearch",
            "angles": [0.0, 45.0, 2],
            "velocities": [0.0, 1.0, 2],
            "angle_units": "degrees",
            "given_ecliptic": None,
        }

        # Without a given ecliptic, we use the WCS.
        gen3 = create_trajectory_generator(config3, work_unit=fake_work)
        self.assertTrue(type(gen3) is EclipticCenteredSearch)
        self.assertAlmostEqual(gen3.ecliptic_angle, fake_ecliptic)
        self.assertAlmostEqual(gen3.min_ang, fake_ecliptic)
        self.assertAlmostEqual(gen3.max_ang, fake_ecliptic + np.pi / 4.0)

        # The given_ecliptic has priority over the fake WCS.
        config3["given_ecliptic"] = 0.0
        gen4 = create_trajectory_generator(config3, work_unit=fake_work)
        self.assertAlmostEqual(gen4.ecliptic_angle, 0.0)
        self.assertEqual(gen4.min_ang, 0.0)
        self.assertEqual(gen4.max_ang, np.pi / 4.0)

        # Fail with no name or a bad name.
        self.assertRaises(KeyError, create_trajectory_generator, {})
        self.assertRaises(KeyError, create_trajectory_generator, {"name": "Invalid_generator"})

    def test_create_trajectory_generator_config(self):
        config = SearchConfiguration()
        generator_config = {
            "name": "KBMODV1SearchConfig",
            "ang_arr": [0.5, 0.5, 30],
            "average_angle": 0.0,
            "v_arr": [0.0, 10.0, 100],
        }
        config.set("generator_config", generator_config)

        # We process the legacy configuration correctly.
        gen1 = create_trajectory_generator(config)
        self.assertTrue(type(gen1) is KBMODV1SearchConfig)
        self.assertEqual(gen1.vel_steps, 100)
        self.assertEqual(gen1.min_vel, 0.0)
        self.assertEqual(gen1.max_vel, 10.0)
        self.assertEqual(gen1.ang_steps, 30)
        self.assertEqual(gen1.min_ang, -0.5)
        self.assertEqual(gen1.max_ang, 0.5)

        # Fail if no generator configuration is provided.
        config.set("generator_config", None)
        self.assertRaises(ValueError, create_trajectory_generator, config)


if __name__ == "__main__":
    unittest.main()
