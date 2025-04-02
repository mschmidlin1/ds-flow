import os
from typing import List


def list_files_all_subdirectories(dir_path: str) -> List[str]:
    """
    Recursively get all file paths in a directory and its subdirectories.

    Args:
        dir_path (str): The path to the directory to search.

    Returns:
        List[str]: A list of full file paths found in the directory and its subdirectories.

    Example:
        >>> list_files_all_subdirectories("./data")
        ['./data/file1.txt', './data/subdir/file2.txt']
    """
    all_files = []
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            all_files.extend(list_files_all_subdirectories(full_path))
        else:
            all_files.append(full_path)
    return all_files


def list_folders_all_subdirectories(dir_path: str) -> List[str]:
    """
    Recursively get all folder paths in a directory and its subdirectories.

    Args:
        dir_path (str): The path to the directory to search.

    Returns:
        List[str]: A list of full folder paths found in the directory and its subdirectories.

    Example:
        >>> list_folders_all_subdirectories("./data")
        ['./data/subdir1', './data/subdir1/nested']
    """
    folders = []
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            folders.append(full_path)
            folders.extend(list_folders_all_subdirectories(full_path))
    return folders


def list_files_and_folders_all_subdirectories(dir_path: str) -> List[str]:
    """
    Recursively get all file and folder paths in a directory and its subdirectories.

    Args:
        dir_path (str): The path to the directory to search.

    Returns:
        List[str]: A list of full file and folder paths found in the directory and its subdirectories.

    Example:
        >>> list_files_and_folders_all_subdirectories("./data")
        ['./data/file1.txt', './data/subdir1', './data/subdir1/file2.txt', './data/subdir1/nested']
    """
    paths = []
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        paths.append(full_path)
        if os.path.isdir(full_path):
            paths.extend(list_files_and_folders_all_subdirectories(full_path))
    return paths