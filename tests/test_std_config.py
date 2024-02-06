import sys

import unittest

from kbmod import StandardizerConfig


class TestStandardizerConfig(unittest.TestCase):
    """Test StandardizerConfig."""

    def test_config(self):
        """Test Standardizer Config behaves as expected."""
        # Test init from dict
        expected = {"a": 1, "b": 2, "c": 3}
        conf = StandardizerConfig(expected)
        self.assertEqual(len(conf), 3)
        self.assertEqual(list(conf.keys()), ["a", "b", "c"])
        self.assertEqual(list(conf.values()), [1, 2, 3])
        self.assertTrue("a" in conf)
        self.assertFalse("noexist" in conf)

        # Test init from kwargs
        conf2 = StandardizerConfig(a=1, b=2, c=3)
        self.assertEqual(conf, conf2)

        # Test it behaves dict-like
        with self.assertRaises(KeyError):
            _ = conf2["noexist"]

        conf["a"] = 10
        self.assertEqual(conf["a"], 10)
        self.assertEqual(list(iter(conf)), ["a", "b", "c"])

        # Test .update method
        conf.update(conf2)
        self.assertEqual(conf, conf2)

        conf.update(expected)
        self.assertEqual(conf, expected)

        conf.update({"a": 11, "b": 12, "c": 13})
        self.assertDictEqual(conf.toDict(), {"a": 11, "b": 12, "c": 13})

        conf.update(a=1, b=2, c=3)
        self.assertEqual(conf, conf2)

        with self.assertRaises(TypeError):
            conf2.update([1, 2, 3])

    @unittest.skipIf(sys.version_info < (3, 9), "py<3.9 does not support or for dicts.")
    def test_or(self):
        expected = {"a": 1, "b": 2, "c": 3}
        conf = StandardizerConfig(expected)
        conf2 = StandardizerConfig(a=1, b=2, c=3)
        self.assertEqual(conf2 | conf, expected)


if __name__ == "__main__":
    unittest.main()
