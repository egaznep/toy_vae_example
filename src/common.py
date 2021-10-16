import numpy as np

def find_limits(x):
    """ Return limits of an array

    Args:
        x : Numpy array to have the min/max computed over

    Returns:
        min: Minimum value of the array
        max: Maximum value of the array
    """
    return np.min(x), np.max(x)

def merge_into_flat_one(*args):
    """  Merge many arrays of arbitrary shape in a list
    to a single, flat array.

    Returns:
        array: Flattened array containing all the data from
        all the arrays.
    """
    flat = []
    for arg in args:
        flat.append(np.asarray(arg).ravel())
    return np.concatenate(flat)