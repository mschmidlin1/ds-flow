"""
Tests for the pandas_flow module.
"""

import unittest
from ds_flow import pandas_flow
from ds_flow.pandas_flow.hello_world import hello


class TestPandasFlow(unittest.TestCase):
    """Test cases for pandas_flow module."""

    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(pandas_flow)

    def test_hello(self):
        """Test the hello() function returns the expected greeting."""
        # Arrange & Act
        result = hello()
        
        # Assert
        self.assertEqual(result, "Hello, World!")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main() 