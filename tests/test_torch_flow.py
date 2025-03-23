"""
Tests for the torch_flow module.
"""

import unittest
from ds_flow import torch_flow


class TestTorchFlow(unittest.TestCase):
    """Test cases for torch_flow module."""

    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(torch_flow)


if __name__ == "__main__":
    unittest.main() 