import tempfile
import unittest

import numpy as np

from kbmod.analysis_utils import PostProcess
from kbmod.search import *


class test_kernels_wrappers(unittest.TestCase):
    def test_sigmag_filtered_indices_same(self):
        # With everything the same, nothing should be filtered.
        values = [1.0 for _ in range(20)]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), 20)

    def test_sigmag_filtered_indices_no_outliers(self):
        # Try with a median of 1.0 and a percentile range of 3.0 (2.0 - -1.0).
        # It should filter any values outside [-3.45, 5.45]
        values = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.1]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), len(values))

    def test_sigmag_filtered_indices_one_outlier(self):
        # Try with a median of 1.0 and a percentile range of 3.0 (2.0 - -1.0).
        # It should filter any values outside [-3.45, 5.45]
        values = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 5.46]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), len(values) - 1)

        # The only missing value should be index=8.
        for i in range(8):
            self.assertTrue(i in inds)
        self.assertFalse(8 in inds)

        # All points pass if we use a larger width.
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 3.0)
        self.assertEqual(len(inds), len(values))

    def test_sigmag_filtered_indices_other_bounds(self):
        # Do the filtering of test_sigmag_filtered_indices_one_outlier
        # with wider bounds [-1.8944, 3.8944].
        values = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.85]
        inds = sigmag_filtered_indices(values, 0.15, 0.85, 0.4824, 2.0)

        # Nothing is filtered this time.
        self.assertEqual(len(inds), len(values))
        for i in range(9):
            self.assertTrue(i in inds)

        # Move one of the points to be an outlier.
        values = [-1.9, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.85]
        inds = sigmag_filtered_indices(values, 0.15, 0.85, 0.4824, 2.0)

        # The first entry is filtered this time.
        self.assertEqual(len(inds), len(values) - 1)
        self.assertFalse(0 in inds)
        for i in range(1, 9):
            self.assertTrue(i in inds)

    def test_sigmag_filtered_indices_two_outliers(self):
        # Try with a median of 0.0 and a percentile range of 1.1 (1.0 - -0.1).
        # It should filter any values outside [-1.631, 1.631].
        values = [1.6, 0.0, 1.0, 0.0, -1.5, 0.5, 1000.1, 0.0, 0.0, -5.2, -0.1]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        for idx in inds:
            self.assertGreaterEqual(values[idx], -1.631)
            self.assertLessEqual(values[idx], 1.631)
        self.assertEqual(len(inds), len(values) - 2)

        # One more point passes if we use a larger width.
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 20.0)
        self.assertEqual(len(inds), len(values) - 1)

    def test_sigmag_filtered_indices_three_outliers(self):
        # Try with a median of 5.0 and a percentile range of 4.0 (7.0-3.0).
        # It should filter any values outside [-0.93, 10.93].
        values = [5.0]
        for i in range(12):
            values.append(3.0)
        values.append(10.95)
        values.append(-1.50)
        for i in range(12):
            values.append(7.0)
        values.append(-0.95)
        values.append(7.0)

        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), len(values) - 3)

        for i in range(29):
            valid = i != 13 and i != 14 and i != 27
            self.assertEqual(i in inds, valid)

    def test_calculate_likelihood_psiphi(self):
        # make sure that the calculate_likelihood_psi_phi works.
        psi_values = [1.0 for _ in range(20)]
        phi_values = [1.0 for _ in range(20)]

        lh = calculate_likelihood_psi_phi(psi_values, phi_values)

        self.assertEqual(lh, 4.47213595499958)

    def test_calculate_likelihood_psiphi_zero_or_negative_phi(self):
        # make sure that the calculate_likelihood_psi_phi works
        # properly when phi values are less than or equal to zero.
        psi_values = [1.0 for _ in range(20)]
        phi_values = [-1.0 for _ in range(20)]

        # test negatives
        lh = calculate_likelihood_psi_phi(psi_values, phi_values)
        self.assertEqual(lh, 0.0)

        # test zero
        lh = calculate_likelihood_psi_phi([1.0], [0.0])
        self.assertEqual(lh, 0.0)


if __name__ == "__main__":
    unittest.main()
