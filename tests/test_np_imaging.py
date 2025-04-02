import unittest
import numpy as np
from ds_flow.np_flow.np_imaging import log_of_img, sixteenbit_to_8bit, eightbit_to_sixteenbit

class TestNpImaging(unittest.TestCase):
    """Test cases for np_imaging module."""

    def test_log_of_img_basic(self):
        """Test log_of_img with basic uint8 input."""
        img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        result = log_of_img(img)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))
        # Check that the transformation preserves relative ordering
        self.assertTrue(result[0, 0] < result[0, 1] < result[1, 0] < result[1, 1])

    def test_log_of_img_different_dtype(self):
        """Test log_of_img with different output dtype."""
        img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        result = log_of_img(img, final_dtype=np.uint16)
        self.assertEqual(result.dtype, np.uint16)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 65535))

    def test_log_of_img_different_log_function(self):
        """Test log_of_img with different log function."""
        img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        result = log_of_img(img, log_func=np.log)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_log_of_img_different_buffer(self):
        """Test log_of_img with different buffer value."""
        img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        result = log_of_img(img, buffer=1.0)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_sixteenbit_to_8bit_basic(self):
        """Test sixteenbit_to_8bit with basic input."""
        img = np.array([[0, 256], [512, 65535]], dtype=np.uint16)
        expected = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        result = sixteenbit_to_8bit(img)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)

    def test_sixteenbit_to_8bit_edge_cases(self):
        """Test sixteenbit_to_8bit with edge cases."""
        img = np.array([[0, 255, 256, 65535]], dtype=np.uint16)
        expected = np.array([[0, 0, 1, 255]], dtype=np.uint8)
        result = sixteenbit_to_8bit(img)
        np.testing.assert_array_equal(result, expected)

    def test_eightbit_to_sixteenbit_basic(self):
        """Test eightbit_to_sixteenbit with basic input."""
        img = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        expected = np.array([[0, 256], [512, 65280]], dtype=np.uint16)
        result = eightbit_to_sixteenbit(img)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint16)

    def test_eightbit_to_sixteenbit_edge_cases(self):
        """Test eightbit_to_sixteenbit with edge cases."""
        img = np.array([0, 1, 255], dtype=np.uint8)
        expected = np.array([0, 256, 65280], dtype=np.uint16)
        result = eightbit_to_sixteenbit(img)
        np.testing.assert_array_equal(result, expected)

    def test_conversion_roundtrip(self):
        """Test that converting from 8-bit to 16-bit and back preserves the original values."""
        original = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        sixteen_bit = eightbit_to_sixteenbit(original)
        back_to_eight = sixteenbit_to_8bit(sixteen_bit)
        np.testing.assert_array_equal(original, back_to_eight)

if __name__ == "__main__":
    unittest.main() 