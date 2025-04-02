import unittest
import numpy as np
from ds_flow.np_flow import min_max_normalization

class TestNpFlow(unittest.TestCase):
    """Test cases for numpy_utils module."""

    def test_min_max_normalization_simple_array(self):
        """Test min_max_normalization with a simple array."""
        arr = np.array([1, 2, 3, 4, 5])
        result = min_max_normalization(arr)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_normalization_with_scalar(self):
        """Test min_max_normalization with a different scalar value."""
        arr = np.array([1, 2, 3, 4, 5])
        result = min_max_normalization(arr, scalar=2.0)
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_normalization_with_zeros(self):
        """Test min_max_normalization with array containing zeros."""
        arr = np.array([0, 1, 2, 3])
        result = min_max_normalization(arr)
        expected = np.array([0.0, 0.33333333, 0.66666667, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_normalization_with_negatives(self):
        """Test min_max_normalization with array containing negative values."""
        arr = np.array([-2, -1, 0, 1, 2])
        result = min_max_normalization(arr)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_normalization_with_nan(self):
        """Test min_max_normalization with array containing NaN values."""
        arr = np.array([1, 2, np.nan, 4, 5])
        result = min_max_normalization(arr)
        expected = np.array([0.0, 0.25, np.nan, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_normalization_2d_array(self):
        """Test min_max_normalization with a 2D array."""
        arr = np.array([[1, 2], [3, 4]])
        result = min_max_normalization(arr)
        expected = np.array([[0.0, 0.33333333], [0.66666667, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main() 