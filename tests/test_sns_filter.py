import unittest
from kbmod.results import Results
from kbmod.filters.sns_filters import no_op_filter


class TestSnsFilter(unittest.TestCase):
    def test_no_op_returns_5(self):
        empty_result = Results()
        self.assertTrue(no_op_filter(empty_result) == 5)

    def test_always_passes(self):
        pass


if __name__ == "__main__":
    unittest.main()
