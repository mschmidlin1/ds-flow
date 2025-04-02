import numpy as np


def min_max_normalization(arr: np.ndarray, scalar: float = 1.0) -> np.ndarray:
    """
    Performs min-max scaling on an array, normalizing values to a range of [0, scalar].

    This function scales the input array so that all values fall between 0 and the specified scalar,
    while preserving the relative differences between values. It handles NaN values by using np.nanmin
    and np.nanmax for the scaling.

    Args:
        arr (np.ndarray): Input array to normalize. Can contain NaN values.
        scalar (float, optional): The maximum value to scale to. Defaults to 1.0.

    Returns:
        np.ndarray: Normalized array with values scaled to [0, scalar] range.

    Example:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> normalized = min_max_normalization(arr)
        >>> print(normalized)
        [0.   0.25 0.5  0.75 1.  ]
    """
    return (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))*scalar