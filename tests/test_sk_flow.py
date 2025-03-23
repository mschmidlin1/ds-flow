"""
Tests for the sk_flow module.
"""

import unittest
from ds_flow import sk_flow


class TestSkFlow(unittest.TestCase):
    """Test cases for sk_flow module."""

    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(sk_flow)


if __name__ == "__main__":
    unittest.main() 