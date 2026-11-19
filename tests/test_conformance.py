"""The adapter `trl_grpo` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "trl_grpo"


if __name__ == "__main__":
    unittest.main()
