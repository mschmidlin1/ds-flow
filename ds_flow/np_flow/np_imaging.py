from ds_flow.np_flow.math_flow import min_max_normalization


import numpy as np


def log_of_img(
    img: np.ndarray,
    final_dtype: type = np.uint8,
    buffer: float = 2.0,
    log_func: callable = np.log10
) -> np.ndarray:
    """
    Apply logarithmic transformation to a grayscale image with normalization.

    This function takes a grayscale image of any bit depth and applies a logarithmic
    transformation to enhance details in darker regions while compressing brighter regions.
    The result is normalized and converted to the specified output data type.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image array of any bit depth.
    final_dtype : type, optional
        Output data type for the transformed image. Default is np.uint8.
    buffer : float, optional
        Small constant added to the input before taking the logarithm to avoid
        taking log of zero or negative values. Default is 2.0.
    log_func : callable, optional
        Logarithmic function to apply. Default is np.log10 (base-10 logarithm).

    Returns
    -------
    np.ndarray
        Logarithmically transformed image normalized to the range of final_dtype.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    >>> log_img = log_of_img(img)
    """

    max_val = np.iinfo(final_dtype).max
    log_img = log_func(img.astype(float)+buffer)
    log_img = min_max_normalization(log_img)
    log_img = log_img*max_val
    log_img = np.round(log_img)
    log_img = log_img.astype(final_dtype)
    return log_img


def sixteenbit_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit image to an 8-bit image by scaling down the pixel values.

    This function takes a 16-bit image (with pixel values ranging from 0 to 65535)
    and converts it to an 8-bit image (with pixel values ranging from 0 to 255)
    by dividing all pixel values by 256 and rounding down to the nearest integer.

    Args:
        img (np.ndarray): Input 16-bit image as a numpy array.

    Returns:
        np.ndarray: 8-bit image as a numpy array with dtype uint8.

    Note:
        - The input image should be a 16-bit image (dtype uint16)
        - The conversion is done by dividing by 256 and rounding down
        - The output will have dtype uint8
    """
    return (img / 256).astype(np.uint8)


def eightbit_to_sixteenbit(img: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit image to a 16-bit image by scaling up the pixel values.

    This function takes an 8-bit image (with pixel values ranging from 0 to 255)
    and converts it to a 16-bit image (with pixel values ranging from 0 to 65535)
    by multiplying all pixel values by 256.

    Args:
        img (np.ndarray): Input 8-bit image as a numpy array.

    Returns:
        np.ndarray: 16-bit image as a numpy array with dtype uint16.

    Note:
        - The input image should be an 8-bit image (dtype uint8)
        - The conversion is done by multiplying by 256
        - The output will have dtype uint16
    """
    img = img.astype(np.uint16)
    return img * 256