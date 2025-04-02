"""
Tests for the pandas_flow module.
"""

import unittest
from ds_flow import pandas_flow


class TestPandasFlow(unittest.TestCase):
    """Test cases for pandas_flow module."""

    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(pandas_flow)


if __name__ == "__main__":
    unittest.main() 