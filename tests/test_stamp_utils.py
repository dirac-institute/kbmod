from itertools import product
import numpy as np
import unittest

from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.results import Results
from kbmod.search import (
    ImageStack,
    LayeredImage,
    PSF,
    StampType,
    Trajectory,
    KB_NO_DATA,
)
from kbmod.stamp_utils import extract_stamp_np, StampMaker


class test_stamp_utils(unittest.TestCase):
    # Tests the basic cutout of a stamp from an numpy array of pixel data.
    def test_extract_stamp_np(self):
        """Tests the basic stamp creation."""
        width = 10
        height = 12
        img_data = np.arange(0, width * height, dtype=np.single).reshape(height, width)

        # Test a stamp at the corner of the image. But entirely within the image.
        stamp = extract_stamp_np(img_data, 2, 2, 2)
        self.assertEqual(stamp.shape, (5, 5))
        self.assertTrue(np.allclose(stamp, img_data[0:5, 0:5], equal_nan=True))

        # Test a stamp that is not at the corner.
        stamp2 = extract_stamp_np(img_data, 8, 5, 1)
        self.assertEqual(stamp2.shape, (3, 3))
        self.assertTrue(np.allclose(stamp2, img_data[4:7, 7:10], equal_nan=True))

        # Test a stamp that goes out of bounds.
        stamp3 = extract_stamp_np(img_data, 0, 11, 1)
        expected = np.array([[np.nan, 100.0, 101.0], [np.nan, 110.0, 111.0], [np.nan, np.nan, np.nan]])
        self.assertTrue(np.allclose(stamp3, expected, equal_nan=True))

        # Test a stamp that is completely out of bounds.
        stamp4 = extract_stamp_np(img_data, 20, 20, 1)
        expected = np.full((3, 3), np.nan)
        self.assertTrue(np.allclose(stamp4, expected, equal_nan=True))

        # Test a stamp that is completely out of bounds along a second direction.
        stamp5 = extract_stamp_np(img_data, -5, -5, 1)
        expected = np.full((3, 3), np.nan)
        self.assertTrue(np.allclose(stamp5, expected, equal_nan=True))

        # Test a stamp that overlaps at a single corner pixel.
        stamp6 = extract_stamp_np(img_data, -1, -1, 1)
        expected = np.full((3, 3), np.nan)
        expected[2][2] = 0.0
        self.assertTrue(np.allclose(stamp6, expected, equal_nan=True))

    def test_extract_stamp_np_masked(self):
        """Tests the basic stamp creation with masked pixels."""
        width = 10
        height = 12

        img_data = 10.0 * np.ones((height, width), dtype=np.single)
        img_data[5, 6] = 0.1
        img_data[5, 7] = KB_NO_DATA
        img_data[3, 1] = 100.0
        img_data[4, 4] = KB_NO_DATA
        img_data[5, 5] = 100.0

        # Test basic stamp creation.
        stamp = extract_stamp_np(img_data, 7, 5, 1)
        self.assertEqual(stamp.shape, (3, 3))
        self.assertTrue(np.allclose(stamp, img_data[4:7, 6:9], equal_nan=True))

    def test_stamp_maker_basic(self):
        # Create a quick fake data set with known values.
        num_times = 5
        width = 10
        height = 15
        psf = PSF(1e-12)

        imgs = []
        for i in range(num_times):
            # Each science layer is the time stamp plus a very small offset for
            # where the pixel is in the image.
            sci_data = np.arange(float(i), float(i) + 0.01 * width * height, 0.01, dtype=np.single).reshape(
                height, width
            )
            img = LayeredImage(
                sci_data,
                np.full((height, width), 0.01).astype(np.float32),  # var
                np.zeros((height, width)).astype(np.float32),  # mask
                psf,
                i,  # time
            )
            imgs.append(img)
        stack = ImageStack(imgs)

        # Create a moving trajectory and its expected center values.
        trj = Trajectory(2, 2, 1, 0)
        expected = [0.22, 1.23, 2.24, 3.25, 4.26]

        for mk_copy, use_var in product([True, False], [True, False]):
            with self.subTest(mk_copy=mk_copy, use_var=use_var):
                stamper = StampMaker(stack, 5, mk_copy=mk_copy)

                stamp_array = stamper.create_stamp_stack(trj, use_var=use_var)
                self.assertEqual(stamp_array.shape[0], num_times)
                self.assertEqual(stamp_array.shape[1], 11)
                self.assertEqual(stamp_array.shape[2], 11)

                # Check the center pixel in the stamp
                for i in range(num_times):
                    if use_var:
                        self.assertAlmostEqual(stamp_array[i][5][5], 0.01, delta=0.0001)
                    else:
                        self.assertAlmostEqual(stamp_array[i][5][5], expected[i], delta=0.0001)

        # Test that if we use a mask array, we only produce a subset of the stack.
        stamper = StampMaker(stack, 3)
        stamp_array = stamper.create_stamp_stack(trj, mask=np.array([True, False, True, True, False]))
        self.assertEqual(stamp_array.shape[0], 5)
        self.assertEqual(stamp_array.shape[1], 7)
        self.assertEqual(stamp_array.shape[2], 7)
        self.assertAlmostEqual(stamp_array[0][3][3], expected[0], delta=0.0001)
        self.assertTrue(np.all(np.isnan(stamp_array[1])))
        self.assertAlmostEqual(stamp_array[2][3][3], expected[2], delta=0.0001)
        self.assertAlmostEqual(stamp_array[3][3][3], expected[3], delta=0.0001)
        self.assertTrue(np.all(np.isnan(stamp_array[4])))

    def test_make_coadds_simple(self):
        psf = PSF(1e-12)

        # Create an image set with three 3x3 images.
        sci1 = np.array([[0, np.nan, np.nan], [0, np.nan, 0.5], [0, 1, 0.5]]).astype(np.float32)
        sci2 = np.array([[1, np.nan, 0.5], [1, 2, 0.5], [1, 2, 0.5]]).astype(np.float32)
        sci3 = np.array([[2, 3, 0.5], [2, 3, 0.5], [2, 3, 0.5]]).astype(np.float32)
        var1 = np.full((3, 3), 0.1).astype(np.float32)
        var2 = np.full((3, 3), 0.2).astype(np.float32)
        var3 = np.full((3, 3), 0.5).astype(np.float32)
        msk = np.zeros((3, 3)).astype(np.float32)

        imlist = [
            LayeredImage(sci1, var1, msk, psf, 0.0),
            LayeredImage(sci2, var2, msk, psf, 1.0),
            LayeredImage(sci3, var3, msk, psf, 2.0),
        ]
        stack = ImageStack(imlist)

        # One Trajectory right in the image's middle.
        trj = Trajectory(x=1, y=1, vx=0.0, vy=0.0)

        # Build a stamp maker with radius=1.
        stamper = StampMaker(stack, 1)

        # Compute and check the coadds
        coadds = stamper.make_coadds(trj, ["sum", "mean", "median", "weighted"])

        expected_sum = np.array([[3.0, 3.0, 1.0], [3.0, 5.0, 1.5], [3.0, 6.0, 1.5]]).astype(np.float32)
        self.assertTrue(np.allclose(coadds["sum"], expected_sum, atol=1e-5))

        expected_mean = np.array([[1.0, 3.0, 0.5], [1.0, 2.5, 0.5], [1.0, 2.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(coadds["mean"], expected_mean, atol=1e-5))

        expected_median = np.array([[1.0, 3.0, 0.5], [1.0, 2.5, 0.5], [1.0, 2.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(coadds["median"], expected_median, atol=1e-5))

        expected_weighted = np.array(
            [
                [0.5294117647058824, 3.0, 0.5],
                [0.5294117647058824, 2.2857142857142856, 0.5],
                [0.5294117647058824, 1.5294117647058822, 0.5],
            ]
        )
        self.assertTrue(np.allclose(coadds["weighted"], expected_weighted, atol=1e-5))

        # Compute and check the coadds when we mask out the third image. Note that there are NO valid values
        # of pixel (0, 2), so we use 0.0.
        coadds = stamper.make_coadds(
            trj, ["sum", "mean", "median", "weighted"], mask=np.array([True, True, False])
        )

        expected_sum = np.array([[1.0, 0.0, 0.5], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0]]).astype(np.float32)
        self.assertTrue(np.allclose(coadds["sum"], expected_sum, atol=1e-5))

        expected_mean = np.array([[0.5, 0.0, 0.5], [0.5, 2.0, 0.5], [0.5, 1.5, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(coadds["mean"], expected_mean, atol=1e-5))

        expected_median = np.array([[0.5, 0.0, 0.5], [0.5, 2.0, 0.5], [0.5, 1.5, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(coadds["median"], expected_median, atol=1e-5))

        expected_weighted = np.array(
            [
                [0.3333333333333333, 0.0, 0.5],
                [0.3333333333333333, 2.0, 0.5],
                [0.3333333333333333, 1.3333333333333333, 0.5],
            ]
        )
        self.assertTrue(np.allclose(coadds["weighted"], expected_weighted, atol=1e-5))

    def test_append_coadds(self):
        psf = PSF(1e-12)

        # Create an image set with three 3x3 images.
        sci1 = np.array([[0, np.nan, np.nan], [0, np.nan, 0.5], [0, 1, 0.5]]).astype(np.float32)
        sci2 = np.array([[1, np.nan, 0.5], [1, 2, 0.5], [1, 2, 0.5]]).astype(np.float32)
        sci3 = np.array([[2, 3, 0.5], [2, 3, 0.5], [2, 3, 0.5]]).astype(np.float32)
        var1 = np.full((3, 3), 0.1).astype(np.float32)
        var2 = np.full((3, 3), 0.2).astype(np.float32)
        var3 = np.full((3, 3), 0.5).astype(np.float32)
        msk = np.zeros((3, 3)).astype(np.float32)

        imlist = [
            LayeredImage(sci1, var1, msk, psf, 0.0),
            LayeredImage(sci2, var2, msk, psf, 1.0),
            LayeredImage(sci3, var3, msk, psf, 2.0),
        ]
        stack = ImageStack(imlist)

        # A few trajectories with the first right in the image's middle.
        trjs = [
            Trajectory(x=1, y=1, vx=0.0, vy=0.0),
            Trajectory(x=0, y=0, vx=0.0, vy=0.0),
            Trajectory(x=0, y=0, vx=0.5, vy=1.0),
        ]
        results = Results.from_trajectories(trjs)

        # Build a stamp maker with radius=1.
        stamper = StampMaker(stack, 1)

        # Compute and check the coadds
        stamper.append_coadds(results, ["sum", "mean", "median", "weighted"], use_masks=False)

        expected_sum = np.array([[3.0, 3.0, 1.0], [3.0, 5.0, 1.5], [3.0, 6.0, 1.5]]).astype(np.float32)
        self.assertTrue(np.allclose(results["coadd_sum"][0], expected_sum, atol=1e-5))

        expected_mean = np.array([[1.0, 3.0, 0.5], [1.0, 2.5, 0.5], [1.0, 2.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(results["coadd_mean"][0], expected_mean, atol=1e-5))

        expected_median = np.array([[1.0, 3.0, 0.5], [1.0, 2.5, 0.5], [1.0, 2.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(results["coadd_median"][0], expected_median, atol=1e-5))

        expected_weighted = np.array(
            [
                [0.5294117647058824, 3.0, 0.5],
                [0.5294117647058824, 2.2857142857142856, 0.5],
                [0.5294117647058824, 1.5294117647058822, 0.5],
            ]
        )
        self.assertTrue(np.allclose(results["coadd_weighted"][0], expected_weighted, atol=1e-5))

        # Compute and check the coadds when we mask out the third image. Note that there are
        # NO valid values of pixel (0, 2), so we use 0.0.
        img_valid = [[True, True, False], [True, True, False], [False, False, False]]
        results2 = Results.from_trajectories(trjs)
        results2.update_obs_valid(img_valid, drop_empty_rows=False)
        stamper.append_coadds(results2, ["sum", "mean", "median", "weighted"], use_masks=True)

        expected_sum = np.array([[1.0, 0.0, 0.5], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0]]).astype(np.float32)
        self.assertTrue(np.allclose(results2["coadd_sum"][0], expected_sum, atol=1e-5))

        expected_mean = np.array([[0.5, 0.0, 0.5], [0.5, 2.0, 0.5], [0.5, 1.5, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(results2["coadd_mean"][0], expected_mean, atol=1e-5))

        expected_median = np.array([[0.5, 0.0, 0.5], [0.5, 2.0, 0.5], [0.5, 1.5, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(results2["coadd_median"][0], expected_median, atol=1e-5))

        expected_weighted = np.array(
            [
                [0.3333333333333333, 0.0, 0.5],
                [0.3333333333333333, 2.0, 0.5],
                [0.3333333333333333, 1.3333333333333333, 0.5],
            ]
        )
        self.assertTrue(np.allclose(results2["coadd_weighted"][0], expected_weighted, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
