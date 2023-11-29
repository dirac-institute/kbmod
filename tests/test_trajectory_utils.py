import unittest

from kbmod.trajectory_utils import *
from kbmod.search import *


class test_trajectory_utils(unittest.TestCase):
    def test_make_trajectory(self):
        trj = make_trajectory(x=1, y=2, vx=3.0, vy=4.0, flux=5.0, lh=6.0, obs_count=7)
        self.assertEqual(trj.x, 1)
        self.assertEqual(trj.y, 2)
        self.assertEqual(trj.vx, 3.0)
        self.assertEqual(trj.vy, 4.0)
        self.assertEqual(trj.flux, 5.0)
        self.assertEqual(trj.lh, 6.0)
        self.assertEqual(trj.obs_count, 7)

    def test_trajectory_from_np_object(self):
        np_obj = np.array(
            [(300.0, 750.0, 106.0, 44.0, 9.52, -0.5, 10.0)],
            dtype=[
                ("lh", "<f8"),
                ("flux", "<f8"),
                ("x", "<f8"),
                ("y", "<f8"),
                ("vx", "<f8"),
                ("vy", "<f8"),
                ("num_obs", "<f8"),
            ],
        )

        trj = trajectory_from_np_object(np_obj)
        self.assertEqual(trj.x, 106)
        self.assertEqual(trj.y, 44)
        self.assertAlmostEqual(trj.vx, 9.52, delta=1e-5)
        self.assertAlmostEqual(trj.vy, -0.5, delta=1e-5)
        self.assertEqual(trj.flux, 750.0)
        self.assertEqual(trj.lh, 300.0)
        self.assertEqual(trj.obs_count, 10)

    def test_trajectory_from_dict(self):
        trj_dict = {
            "x": 1,
            "y": 2,
            "vx": 3.0,
            "vy": 4.0,
            "flux": 5.0,
            "lh": 6.0,
            "obs_count": 7,
        }
        trj = trajectory_from_dict(trj_dict)

        self.assertEqual(trj.x, 1)
        self.assertEqual(trj.y, 2)
        self.assertEqual(trj.vx, 3.0)
        self.assertEqual(trj.vy, 4.0)
        self.assertEqual(trj.flux, 5.0)
        self.assertEqual(trj.lh, 6.0)
        self.assertEqual(trj.obs_count, 7)

    def test_trajectory_yaml(self):
        """Test serializing and then deserializing the Trajectory to a YAML."""
        org_trj = make_trajectory(x=1, y=2, vx=3.0, vy=4.0, flux=5.0, lh=6.0, obs_count=7)
        yaml_str = trajectory_to_yaml(org_trj)
        self.assertGreater(len(yaml_str), 0)

        new_trj = trajectory_from_yaml(yaml_str)
        self.assertEqual(new_trj.x, 1)
        self.assertEqual(new_trj.y, 2)
        self.assertEqual(new_trj.vx, 3.0)
        self.assertEqual(new_trj.vy, 4.0)
        self.assertEqual(new_trj.flux, 5.0)
        self.assertEqual(new_trj.lh, 6.0)
        self.assertEqual(new_trj.obs_count, 7)


if __name__ == "__main__":
    unittest.main()
