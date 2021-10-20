import numpy as np
import importlib

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

def get_constructor(start, type):
    """ Get constructor to the object described by string
    A constrained "eval" alternative for this specific purpose

    Args:
        start (string): Package to start looking for the object
        type (string): Type string, tokenized with '.' (python syntax) 

    Returns:
        callable: Constructor of the object.
    """
    tokens = type.split('.')
    N_tokens = len(tokens)

    for i, k in enumerate(tokens):
        if i+1 < N_tokens:
            start = start + '.' + k
            module = importlib.import_module(start)
        else:
            result = getattr(module, k)
    return result