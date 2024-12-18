import unittest

import numpy as np

from kbmod.search import (
    Index,
    Point,
    Rectangle,
    centered_range,
    anchored_block,
)


class test_index(unittest.TestCase):
    """Test Index class instantiates, sets and prints correctly."""

    def test_init(self):
        """Test Index instantiates correctly."""
        idx = Index(1, 2)
        self.assertEqual(idx.i, 1)
        self.assertEqual(idx.j, 2)

    def test_eq(self):
        """Test Index sets attrs and compares correctly."""
        idx = Index(1, 2)
        idx2 = Index(1, 2)
        self.assertEqual(idx, idx2)

        idx2.i = 10
        self.assertNotEqual(idx, idx2)
        self.assertEqual(idx2.i, 10)

    def test_to_yaml(self):
        """Test Index produces a correct YAML-like record."""
        self.assertEqual("{i: 1, j: 2}", Index(1, 2).to_yaml())


class test_point(unittest.TestCase):
    """Test Point class instantiates, sets and prints correctly."""

    def test_init(self):
        """Test Point instantiates correctly."""
        p = Point(1, 2)
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)

    def test_eq(self):
        """Test Point sets attrs and compares correctly."""
        p = Point(1, 2)
        p1 = Point(1, 2)
        self.assertEqual(p, p1)
        p1.x = 10
        self.assertNotEqual(p, p1)
        self.assertEqual(p1.x, 10)

    def test_to_yaml(self):
        """Test Point produces a correct YAML-like record."""
        self.assertEqual("{x: 1.000000, y: 2.000000}", Point(1, 2).to_yaml())

    def test_to_index(self):
        """Test Point correctly casts itself as Index"""
        # Index spans <0, 1] inclusively I guess
        # (doesn't really make sense though)
        p = Point(1, 1)
        self.assertEqual(Index(0, 0), p.to_index())

        p = Point(0.5, 0.5)
        self.assertEqual(Index(0, 0), p.to_index())

        p = Point(1.5, 1.5)
        self.assertEqual(Index(1, 1), p.to_index())

        # note Point and Index use different conventions
        # ij implies yx, versus Point, representing a Cartesian
        # coordinate uses xy.
        p = Point(1, 2)
        self.assertEqual(Index(1, 0), p.to_index())


class test_anchored_rect(unittest.TestCase):
    """Test Rectangle class instantiates, sets and prints correctly."""

    def test_init(self):
        """Test Rectangle instantiates and compares correctly."""
        rect = Rectangle(Index(0, 0), Index(1, 1), 2, 3)
        self.assertEqual(Index(0, 0), rect.corner)
        self.assertEqual(Index(1, 1), rect.anchor)
        self.assertEqual(2, rect.width)
        self.assertEqual(3, rect.height)

        rect2 = Rectangle((0, 0), (1, 1), 2, 3)
        self.assertEqual(rect2, rect)

        rect3 = Rectangle((0, 0), 2, 3)
        self.assertNotEqual(rect3, rect)
        self.assertEqual(rect3.corner, rect.corner)
        self.assertEqual(rect3.width, 2)
        self.assertEqual(rect3.height, 3)

        rect3.i = 10
        rect3.j = 20
        self.assertEqual(Index(10, 20), rect3.corner)

    def test_to_yaml(self):
        """Test Rectangle produces a correct YAML-like record."""
        self.assertEqual(
            "{corner: {i: 1, j: 2}, anchor: {i: 3, j: 4}, width: 5, height: 6}",
            Rectangle((1, 2), (3, 4), 5, 6).to_yaml(),
        )

        self.assertEqual(
            "{corner: {i: 1, j: 2}, anchor: {i: 0, j: 0}, width: 5, height: 6}",
            Rectangle((1, 2), 5, 6).to_yaml(),
        )


class test_indexing_functions(unittest.TestCase):
    """Tests various self-standing indexing functions."""

    def test_centered_range(self):
        """Test centered range."""
        # returns (start, end, length)
        # count inclusively, i.e. (start, end) = (0, 0) is [0] a single element
        self.assertEqual(centered_range(0, 0, 0), (0, 0, 0))
        self.assertEqual(centered_range(0, 0, 1), (0, 0, 1))
        self.assertEqual(centered_range(5, 0, 10), (5, 5, 1))

        # TODO: something is odd here, doublecheck
        # when no edges are hit (idx=5, radius=1) --> [4, 5, 6]
        # when edges are hit (idx=5, radius=3) --> [2, 3, 4, 5, 6, 7, 8]
        # when edges are hit (idx=5, radius=4) --> [1, 2, 3, 4, 5, 6, 7, 8]
        # i.e. start=4, end=6, length = 3
        self.assertEqual(centered_range(5, 1, 8), (4, 6, 3))
        self.assertEqual(centered_range(5, 2, 8), (3, 7, 5))
        self.assertEqual(centered_range(5, 3, 8), (2, 8, 6))
        self.assertEqual(centered_range(5, 4, 8), (1, 8, 7))
        self.assertEqual(centered_range(5, 5, 8), (0, 8, 8))

    def test_anchored_block(self):
        """Test cropping a 2r+1 rectangle around a given coordinate correctly
        calculates the anchor point."""
        # easier to reason about when used on an array. Clearer anchor tests
        # occur naturally as part of stamp creation. Hard to express clearly.
        img = np.arange(100).reshape(10, 10)

        # Middle of array, padding radius 1, no edge cases. Full array is
        # filled, so anchor 0, 0. Black so ugly...
        rect = anchored_block((5, 5), 1, img.shape)
        self.assertTrue(
            np.array_equal(
                img[4:7, 4:7],
                img[rect.i : rect.i + rect.height, rect.j : rect.j + rect.width],
            )
        )
        self.assertEqual(rect.anchor, Index(0, 0))

        # No left/up neighbors, anchor moves over 1 in both directions
        rect = anchored_block((0, 0), 1, img.shape)
        self.assertTrue(
            np.array_equal(
                img[0:2, 0:2],
                img[rect.i : rect.i + rect.height, rect.j : rect.j + rect.width],
            )
        )
        self.assertEqual(rect.anchor, Index(1, 1))

        # dimensions are not 2r+1=5, because 2 left and 1 top neighbors are
        # out of bounds. Anchor moves over by 1 in y and 2 in x.
        rect = anchored_block((1, 0), 2, img.shape)
        self.assertTrue(
            np.array_equal(
                img[0:4, 0:3],
                img[rect.i : rect.i + rect.height, rect.j : rect.j + rect.width],
            )
        )
        self.assertEqual(rect.anchor, Index(1, 2))

        # make sure we are clipping on maximums too, no idea what anchor should
        # do here.
        rect = anchored_block((0, 0), 100, img.shape)
        self.assertEqual(rect.width, 10)
        self.assertEqual(rect.height, 10)
        self.assertEqual(rect.anchor, Index(100, 100))

        # crop on bottom
        rect = anchored_block((10, 0), 2, img.shape)
        self.assertTrue(
            np.array_equal(
                img[8:, 0:3],
                img[rect.i : rect.i + rect.height, rect.j : rect.j + rect.width],
            )
        )
        self.assertEqual(rect.anchor, Index(0, 2))

        # crop on the right
        rect = anchored_block((0, 10), 2, img.shape)
        self.assertTrue(
            np.array_equal(
                img[0:3, 8:],
                img[rect.i : rect.i + rect.height, rect.j : rect.j + rect.width],
            )
        )
        self.assertEqual(rect.anchor, Index(2, 0))


if __name__ == "__main__":
    unittest.main()
