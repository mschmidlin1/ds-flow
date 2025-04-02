import unittest
import os
import tempfile
import shutil
from ds_flow.io_flow import (
    list_files_all_subdirectories,
    list_folders_all_subdirectories,
    list_files_and_folders_all_subdirectories,
)


class TestIOUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test directory structure:
        # test_dir/
        # ├── file1.txt
        # ├── file2.txt
        # ├── subdir1/
        # │   ├── file3.txt
        # │   └── nested/
        # │       └── file4.txt
        # └── subdir2/
        #     └── file5.txt
        
        # Create files
        self.files = [
            os.path.join(self.test_dir, "file1.txt"),
            os.path.join(self.test_dir, "file2.txt"),
            os.path.join(self.test_dir, "subdir1", "file3.txt"),
            os.path.join(self.test_dir, "subdir1", "nested", "file4.txt"),
            os.path.join(self.test_dir, "subdir2", "file5.txt"),
        ]
        
        # Create directories
        self.dirs = [
            os.path.join(self.test_dir, "subdir1"),
            os.path.join(self.test_dir, "subdir1", "nested"),
            os.path.join(self.test_dir, "subdir2"),
        ]
        
        # Create all directories
        for dir_path in self.dirs:
            os.makedirs(dir_path)
        
        # Create all files
        for file_path in self.files:
            with open(file_path, "w") as f:
                f.write("test content")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_list_files_all_subdirectories(self):
        # Get all files
        files = list_files_all_subdirectories(self.test_dir)
        
        # Sort both lists for comparison
        files.sort()
        expected_files = [os.path.abspath(f) for f in self.files]
        expected_files.sort()
        
        # Assert that all files are found
        self.assertEqual(files, expected_files)
        
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            list_files_all_subdirectories("/non/existent/path")

    def test_list_folders_all_subdirectories(self):
        # Get all folders
        folders = list_folders_all_subdirectories(self.test_dir)
        
        # Sort both lists for comparison
        folders.sort()
        expected_folders = [os.path.abspath(d) for d in self.dirs]
        expected_folders.sort()
        
        # Assert that all folders are found
        self.assertEqual(folders, expected_folders)
        
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            list_folders_all_subdirectories("/non/existent/path")

    def test_list_files_and_folders_all_subdirectories(self):
        # Get all files and folders
        paths = list_files_and_folders_all_subdirectories(self.test_dir)
        
        # Sort both lists for comparison
        paths.sort()
        expected_paths = [os.path.abspath(p) for p in self.files + self.dirs]
        expected_paths.sort()
        
        # Assert that all files and folders are found
        self.assertEqual(paths, expected_paths)
        
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            list_files_and_folders_all_subdirectories("/non/existent/path")


if __name__ == "__main__":
    unittest.main()
