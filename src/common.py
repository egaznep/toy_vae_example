import numpy as np
import importlib
import ctypes
import sys
from pathlib import Path

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

def magma_init():
    env_path = Path(sys.executable).parent.parent.resolve()
    magma_path = env_path / 'lib/libmagma.so'
    try:
        libmagma = ctypes.cdll.LoadLibrary(magma_path)
        libmagma.magma_init()
    except OSError as e:
        print('Path not found:', magma_path)
        print(repr(e))
