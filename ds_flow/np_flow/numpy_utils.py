


def min_max_normalization(arr):
    """Must take a numpy array. Returns the min-max normalized array."""
    return (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))




def log_of_img(img: np.array, final_dtype=np.uint8, buffer=2.0, log_func=np.log10):
    """
    Takes the log of a greyscale image. The image can be any bit depth.
    """
    
    max_val = np.iinfo(final_dtype).max
    log_img = log_func(img.astype(float)+buffer)
    log_img = min_max_normalization(log_img)
    log_img = log_img*max_val
    log_img = np.round(log_img)
    log_img = log_img.astype(final_dtype)
    return log_img


def sixteenbit_to_8bit(img: np.array):
    """
    Turns a 16 bit image in an 8 bit image.
    """
    return(img/256).astype(np.uint8)