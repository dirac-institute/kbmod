import os
import tempfile
import unittest

from kbmod.file_utils import *
from kbmod.search import *


class test_file_utils(unittest.TestCase):
    def test_load_times(self):
        times = FileUtils.load_time_dictionary("./data/fake_times.dat")
        self.assertEqual(len(times), 3)
        self.assertTrue("000003" in times)
        self.assertTrue("000005" in times)
        self.assertTrue("010006" in times)
        self.assertEqual(times["000003"], 57162.0)
        self.assertEqual(times["000005"], 57172.0)
        self.assertEqual(times["010006"], 100000.0)

    def test_load_psfs(self):
        psfs = FileUtils.load_psf_dictionary("./data/fake_psfs.dat")
        self.assertEqual(len(psfs), 2)
        self.assertTrue("000002" in psfs)
        self.assertTrue("000012" in psfs)
        self.assertEqual(psfs["000002"], 1.3)
        self.assertEqual(psfs["000012"], 1.5)

    def test_load_results(self):
        np_results = FileUtils.load_results_file("./data/fake_results.txt")
        self.assertEqual(len(np_results), 2)
        self.assertEqual(np_results[0]["x"], 106)
        self.assertEqual(np_results[0]["y"], 44)
        self.assertAlmostEqual(np_results[0]["vx"], 9.52)
        self.assertAlmostEqual(np_results[0]["vy"], -0.5)
        self.assertAlmostEqual(np_results[0]["lh"], 300.0)
        self.assertAlmostEqual(np_results[0]["flux"], 750.0)
        self.assertEqual(np_results[0]["num_obs"], 10)
        self.assertEqual(np_results[1]["x"], 55)
        self.assertEqual(np_results[1]["y"], 60)
        self.assertAlmostEqual(np_results[1]["vx"], 10.5)
        self.assertAlmostEqual(np_results[1]["vy"], -1.7)
        self.assertAlmostEqual(np_results[1]["lh"], 250.0)
        self.assertAlmostEqual(np_results[1]["flux"], 500.0)
        self.assertEqual(np_results[1]["num_obs"], 9)

    def test_load_results_trajectories(self):
        trj_results = FileUtils.load_results_file_as_trajectories("./data/fake_results.txt")
        self.assertEqual(len(trj_results), 2)

        self.assertTrue(isinstance(trj_results[0], trajectory))
        self.assertEqual(trj_results[0].x, 106)
        self.assertEqual(trj_results[0].y, 44)
        self.assertAlmostEqual(trj_results[0].x_v, 9.52, delta=1e-6)
        self.assertAlmostEqual(trj_results[0].y_v, -0.5, delta=1e-6)
        self.assertAlmostEqual(trj_results[0].lh, 300.0, delta=1e-6)
        self.assertAlmostEqual(trj_results[0].flux, 750.0, delta=1e-6)
        self.assertEqual(trj_results[0].obs_count, 10)

        self.assertTrue(isinstance(trj_results[1], trajectory))
        self.assertEqual(trj_results[1].x, 55)
        self.assertEqual(trj_results[1].y, 60)
        self.assertAlmostEqual(trj_results[1].x_v, 10.5, delta=1e-6)
        self.assertAlmostEqual(trj_results[1].y_v, -1.7, delta=1e-6)
        self.assertAlmostEqual(trj_results[1].lh, 250.0, delta=1e-6)
        self.assertAlmostEqual(trj_results[1].flux, 500.0, delta=1e-6)
        self.assertEqual(trj_results[1].obs_count, 9)


if __name__ == "__main__":
    unittest.main()
