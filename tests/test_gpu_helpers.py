"""Test some of the functions needed for the GPU."""

import unittest

from kbmod.search import HAS_GPU, print_cuda_stats, validate_gpu


class test_gpu_helpers(unittest.TestCase):
    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_validate_gpu_true(self):
        self.assertTrue(validate_gpu(0))

        # Add a memory test that will fail on all known GPUs (1 exobyte of memory).
        self.assertFalse(validate_gpu(1152921504606846976))

    @unittest.skipIf(HAS_GPU, "Skipping test (GPU detected)")
    def test_validate_gpu_false(self):
        # We should always fail if there is no GPU.
        self.assertFalse(validate_gpu(0))

    def test_print_cuda_stats(self):
        # We should be able to call print_cuda_stats even if there is no GPU.
        print_cuda_stats()


if __name__ == "__main__":
    unittest.main()
