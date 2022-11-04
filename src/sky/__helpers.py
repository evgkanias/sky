import numpy as np

import os
import sys


RNG = np.random.RandomState(2021)
"""
The defaults random value generator.
"""
eps = np.finfo(float).eps
"""
The smallest non-zero positive.
"""
__root__ = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", ".."))
"""
The root directory
"""
__data__ = os.path.join(__root__, 'data')


def set_rng(seed):
    """
    Sets the default random state.

    Parameters
    ----------
    seed: int
    """
    global RNG
    RNG = np.random.RandomState(seed)


def add_noise(v=None, noise=0., shape=None, fill_value=0, rng=RNG):
    if shape is None and v is not None:
        shape = v.shape
    if shape is not None:
        size = np.sum(shape)
    elif v is not None:
        size = v.size
    else:
        size = None
    if isinstance(noise, np.ndarray):
        if size is None or noise.size == size:
            eta = np.array(noise, dtype=bool)
        else:
            eta = np.zeros(shape, dtype=bool)
            eta[:noise.size] = noise
    elif noise > 0:
        if shape is not None:
            eta = np.argsort(np.absolute(rng.randn(*shape)))[:int(noise * shape[0])]
        else:
            eta = rng.randn()
    else:
        eta = np.zeros_like(v, dtype=bool)

    if v is not None:
        v[eta] = fill_value

    return eta


def reset_data_directory(data_dir):
    """
    Sets up the default directory of the data.

    Parameters
    ----------
    data_dir : str
        the new directory path.
    """
    global __data__
    __data__ = data_dir


def print_error_and_exit(message: str):
    print(message)
    sys.exit(-1)

