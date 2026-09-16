"""Regressions and an independent reference for validity-aware clustering."""

import unittest
from unittest.mock import patch

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.filters.clustering_filters import (
    ClusterPredictionFilter,
    GreedyValidEndpointFilter,
    NNSweepFilter,
    apply_clustering,
)
from kbmod.results import Results
from kbmod.run_search import SearchRunner
from kbmod.search import Trajectory
from kbmod.trajectory_generator import VelocityGridSearch


def make_results(coords, masks, likelihood=None):
    coords = np.asarray(coords)
    n = len(coords)
    data = {c: coords[:, i] for i, c in enumerate(("x", "y", "vx", "vy"))}
    data.update(likelihood=np.zeros(n), flux=np.zeros(n), obs_count=np.zeros(n, dtype=int))
    result = Results(data)
    result.update_obs_valid(np.asarray(masks, dtype=bool), drop_empty_rows=False)
    result.table["likelihood"] = np.arange(n, 0, -1) if likelihood is None else likelihood
    result.table["id"] = np.arange(n)
    return result


class TestValidEndpointClustering(unittest.TestCase):
    def test_invalid_endpoints_and_masked_likelihood(self):
        # Same pixels during the valid interval, but very different extrapolations.
        results = make_results([[100, 0, 0, 0], [0, 0, 2, 0]], [[0, 1, 1, 0]] * 2)
        times = np.array([0, 49, 51, 100])
        results.add_psi_phi_data([[1000, 5, 5, 1000], [0, 10, 10, 0]], np.ones((2, 4)), results["obs_valid"])
        self.assertGreater(results["likelihood"][1], results["likelihood"][0])
        self.assertEqual(ClusterPredictionFilter(3, [0, 100]).keep_indices(results), [0, 1])
        self.assertEqual(NNSweepFilter(3, [0, 100]).keep_indices(results), [0, 1])
        self.assertEqual(GreedyValidEndpointFilter(3, times).keep_indices(results), [1])
        # keep_indices does not mutate the caller or recompute ranking from invalid flux.
        self.assertEqual(len(results), 2)
        np.testing.assert_array_equal(results["obs_valid"], [[0, 1, 1, 0]] * 2)

    def test_same_trajectory_different_valid_endpoints(self):
        results = make_results([[5, 6, 100, -50]] * 2, [[1, 1, 1, 0], [0, 1, 1, 1]])
        self.assertEqual(GreedyValidEndpointFilter(1, [0, 1, 2, 3]).keep_indices(results), [0])

    def test_disjoint_single_and_empty_support(self):
        results = make_results(
            [[0, 0, 1, 1]] * 4,
            [[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]],
        )
        self.assertEqual(GreedyValidEndpointFilter(100, [0, 1, 2, 3]).keep_indices(results), [0, 1, 2, 3])

    def test_duplicate_times_do_not_establish_motion(self):
        results = make_results([[0, 0, 0, 0]] * 2, [[1, 1, 0]] * 2)
        self.assertEqual(GreedyValidEndpointFilter(1, [0, 0, 1]).keep_indices(results), [0, 1])

    def test_overlap_fraction_applies_to_both_rows(self):
        results = make_results([[0, 0, 0, 0]] * 2, [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
        self.assertEqual(GreedyValidEndpointFilter(1, np.arange(5)).keep_indices(results), [0, 1])
        self.assertEqual(
            GreedyValidEndpointFilter(1, np.arange(5), min_shared_fraction=0.4).keep_indices(results), [0]
        )
        self.assertEqual(
            GreedyValidEndpointFilter(1, np.arange(5), min_shared_fraction=0, min_shared_obs=3).keep_indices(
                results
            ),
            [0, 1],
        )

    def test_distance_boundary(self):
        results = make_results([[0, 0, 0, 0], [0, 0, 3, 4]], [[1, 1]] * 2)
        self.assertEqual(GreedyValidEndpointFilter(5, [0, 1]).keep_indices(results), [0])
        self.assertEqual(GreedyValidEndpointFilter(4.999, [0, 1]).keep_indices(results), [0, 1])

    def test_greedy_no_chaining_and_stable_ties(self):
        results = make_results([[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]], [[1, 1]] * 3, [30, 20, 10])
        filt = GreedyValidEndpointFilter(3, [0, 1])
        self.assertEqual(filt.keep_indices(results), [0, 2])
        results.table = results.table[[2, 0, 1]]
        self.assertEqual(filt.keep_indices(results), [0, 1])
        results.table["likelihood"] = [10, 10, 10]
        self.assertEqual(filt.keep_indices(results), [0, 1])

    def test_all_valid_matches_sorted_nn_on_simple_groups(self):
        results = make_results(
            [[0, 0, 1, 1], [1, 0, 1, 1], [100, 100, 1, 1], [100, 101, 1, 1]], [[1, 1, 1]] * 4
        )
        expected = NNSweepFilter(2, [0, 10]).keep_indices(results)
        self.assertEqual(GreedyValidEndpointFilter(2, [0, 5, 10]).keep_indices(results), expected)

    def test_unsorted_times_preserve_mask_alignment_and_reference_epoch(self):
        # The trajectory origin is t=50, not the earliest timestamp t=0.
        results = make_results([[100, 0, 0, 0], [100, 0, 2, 0]], [[0, 0, 1, 1, 0]] * 2)
        params = {"cluster_type": "greedy_valid_start_end", "cluster_eps": 3, "times": [50, 100, 51, 49, 0]}
        apply_clustering(results, params)
        self.assertEqual(list(results["id"]), [0])
        self.assertEqual(params["times"], [50, 100, 51, 49, 0])
        self.assertNotIn("pred_times", params)

    def test_empty_singleton_missing_and_malformed_masks(self):
        filt = GreedyValidEndpointFilter(1, [0, 1])
        self.assertEqual(filt.keep_indices(Results()), [])
        results = make_results([[0, 0, 0, 0]], [[1, 1]])
        self.assertEqual(filt.keep_indices(results), [0])
        for masks in ([[True]], [[1, 1]], [True], [[True, False, True]]):
            results.table["obs_valid"] = masks
            with self.assertRaises(ValueError):
                filt.keep_indices(results)
        results.table.remove_column("obs_valid")
        with self.assertRaisesRegex(ValueError, "requires an obs_valid"):
            filt.keep_indices(results)

    def test_invalid_parameters_and_nonfinite_data(self):
        for eps in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                GreedyValidEndpointFilter(eps, [0, 1])
        for times in [[], [[0, 1]], [0, np.nan], [0, np.inf]]:
            with self.assertRaises(ValueError):
                GreedyValidEndpointFilter(1, times)
            with self.assertRaises(ValueError):
                apply_clustering(
                    make_results([[0, 0, 0, 0]], [[1, 1]]),
                    {"cluster_type": "greedy_valid_start_end", "cluster_eps": 1, "times": times},
                )
        for kwargs in [
            {"min_shared_obs": 1},
            {"min_shared_obs": 2.5},
            {"min_shared_obs": True},
            {"min_shared_fraction": -0.1},
            {"min_shared_fraction": 1.1},
            {"min_shared_fraction": np.nan},
            {"batch_size": 0},
            {"batch_size": 1.5},
            {"batch_size": True},
        ]:
            with self.assertRaises(ValueError):
                GreedyValidEndpointFilter(1, [0, 1], **kwargs)
        for col in ("vx", "vy", "likelihood"):
            results = make_results([[0, 0, 0, 0]], [[1, 1]])
            results.table[col] = [np.nan]
            with self.assertRaises(ValueError):
                GreedyValidEndpointFilter(1, [0, 1]).keep_indices(results)

    def test_randomized_against_direct_position_reference(self):
        rng = np.random.default_rng(314159)
        for _ in range(20):
            n, t = 30, 7
            times = rng.uniform(-2, 2, t)
            coords = rng.integers(-2, 3, (n, 4))
            valid = rng.random((n, t)) < 0.7
            lh = rng.permutation(n)
            results = make_results(coords, valid, lh)
            x = coords[:, 0, None] + coords[:, 2, None] * times
            y = coords[:, 1, None] + coords[:, 3, None] * times
            expected = []
            for i in np.argsort(-lh):
                duplicate = False
                for j in expected:
                    shared = np.flatnonzero(valid[i] & valid[j])
                    if len(shared) < 2 or len(shared) < 0.5 * max(valid[i].sum(), valid[j].sum()):
                        continue
                    a, b = shared[np.argmin(times[shared])], shared[np.argmax(times[shared])]
                    if times[a] == times[b]:
                        continue
                    dist = np.linalg.norm(
                        [x[i, a] - x[j, a], y[i, a] - y[j, a], x[i, b] - x[j, b], y[i, b] - y[j, b]]
                    )
                    if dist <= 3:
                        duplicate = True
                        break
                if not duplicate:
                    expected.append(i)
            for batch in [1, 4, 1024]:
                actual = GreedyValidEndpointFilter(3, times, batch_size=batch).keep_indices(results)
                self.assertEqual(actual, sorted(expected))

    def test_search_runner_forwards_overlap_options(self):
        data = FakeDataSet(10, 10, [60000, 60001, 60002, 60003, 60004])
        for min_obs, fraction, expected in [(2, 0.4, 1), (3, 0.4, 2), (2, 0.5, 2)]:
            results = make_results([[5, 5, 0, 0]] * 2, [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
            config = SearchConfiguration(
                {
                    "cluster_type": "greedy_valid_start_end",
                    "cluster_eps": 1,
                    "cluster_min_shared_obs": min_obs,
                    "cluster_min_shared_fraction": fraction,
                    "stamp_type": None,
                    "near_dup_thresh": 0,
                    "max_masked_pixels": 1.0,
                }
            )
            runner = SearchRunner()
            with patch.object(runner, "do_core_search", return_value=results):
                actual = runner.run_search(config, data.stack_py)
            self.assertEqual(len(actual), expected)

    def test_cpu_search_with_validity_aware_clustering(self):
        times = 59000 + np.arange(20) / 20
        data = FakeDataSet(50, 60, times, psf_val=0.5, use_seed=314159)
        data.insert_object(Trajectory(x=17, y=12, vx=21.0, vy=16.0, flux=250.0))
        config = SearchConfiguration(
            {
                "cpu_only": True,
                "cluster_type": "greedy_valid_start_end",
                "cluster_eps": 3,
                "near_dup_thresh": 0,
                "candidate_dup_px": 0,
                "stamp_type": None,
                "x_pixel_bounds": [16, 19],
                "y_pixel_bounds": [11, 14],
                "results_per_pixel": 25,
                "track_filtered": True,
            }
        )
        result = SearchRunner().run_search(
            config, data.stack_py, VelocityGridSearch(5, 15.0, 27.0, 5, 10.0, 22.0)
        )
        self.assertGreater(len(result), 0)
        self.assertIn("obs_valid", result.colnames)
        self.assertTrue(any(key.startswith("GreedyValidEndpointFilter") for key in result.filtered_stats))
        # The brightest surviving trajectory recovers the deterministic injected track.
        best = result.table[np.argmax(result["likelihood"])]
        for col, value in {"x": 17, "y": 12, "vx": 21, "vy": 16}.items():
            self.assertAlmostEqual(best[col], value)


if __name__ == "__main__":
    unittest.main()
