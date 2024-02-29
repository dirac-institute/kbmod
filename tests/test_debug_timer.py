import time
import unittest

from kbmod.search import DebugTimer


class test_debug_timer(unittest.TestCase):
    def test_create(self):
        my_timer = DebugTimer("hi", False)
        time1 = my_timer.read()

        # We use sleep (100ms) because we are only interested in
        # wall clock time having increased.
        time.sleep(0.1)
        time2 = my_timer.read()
        self.assertGreater(time2, time1)

        my_timer.stop()
        time3 = my_timer.read()
        time.sleep(0.1)
        time4 = my_timer.read()
        self.assertAlmostEqual(time3, time4)


if __name__ == "__main__":
    unittest.main()
