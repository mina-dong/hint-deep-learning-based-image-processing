"""Local tests use generated fixtures, NOT real image datasets or model benchmarks."""
import unittest
from cnnlab.common import ROOT
if __name__ == "__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/"tests")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
