"""
Tests for the torch_flow module.
"""

import unittest
from ds_flow import torch_flow
from ds_flow.torch_flow.torch_utils import conv2d_output_size, max_pool_output_size


class TestTorchFlow(unittest.TestCase):
    """Test cases for torch_flow module."""

    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(torch_flow)

    def test_conv2d_output_size(self):
        """Test the conv2d_output_size function with various parameters."""
        # Test default parameters
        self.assertEqual(conv2d_output_size(32), 30)
        
        # Test with padding to maintain input size
        self.assertEqual(conv2d_output_size(32, kernel_size=3, padding=1), 32)
        
        # Test with stride
        self.assertEqual(conv2d_output_size(32, stride=2), 15)
        
        # Test with dilation
        self.assertEqual(conv2d_output_size(32, dilation=2), 28)
        
        # Test with combination of parameters
        self.assertEqual(conv2d_output_size(64, kernel_size=5, stride=2, padding=2), 32)

        # Test invalid cases that would result in output size < 1
        with self.assertRaises(ValueError):
            # Small input with large kernel
            conv2d_output_size(2, kernel_size=5)
        
        with self.assertRaises(ValueError):
            # Small input with large dilation
            conv2d_output_size(3, dilation=3)
        
        self.assertEqual(conv2d_output_size(5, stride=6), 1)           

    def test_max_pool_output_size(self):
        """Test the max_pool_output_size function with various parameters."""
        # Test default parameters (pool_ksize=2, pool_stride=2)
        self.assertEqual(max_pool_output_size(32), 16)
        
        # Test with different pool size
        self.assertEqual(max_pool_output_size(32, pool_ksize=3), 15)
        
        # Test with padding
        self.assertEqual(max_pool_output_size(32, pool_padding=1), 17)
        
        # Test with different stride
        self.assertEqual(max_pool_output_size(32, pool_stride=1), 31)
        
        # Test with dilation
        self.assertEqual(max_pool_output_size(32, dilation=2), 15)
        
        # Test with combination of parameters
        self.assertEqual(max_pool_output_size(64, pool_ksize=3, pool_stride=2, pool_padding=1), 32)

        # Test invalid cases that would result in output size < 1
        with self.assertRaises(ValueError):
            # Small input with large pool size
            max_pool_output_size(2, pool_ksize=4)
        
        with self.assertRaises(ValueError):
            # Small input with large dilation
            max_pool_output_size(3, dilation=4)
        
        self.assertEqual(max_pool_output_size(4, pool_stride=5), 1)
        


if __name__ == "__main__":
    unittest.main() 