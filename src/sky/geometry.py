from scipy.spatial.transform import Rotation as R

import numpy as np


def get_coordinates(ori):
    """

    Parameters
    ----------
    ori : R, np.ndarray

    Returns
    -------

    """

    if isinstance(ori, R):
        xyz = ori2xyz(ori)
    elif isinstance(ori, np.ndarray) and ori.shape[-1] > 2:
        xyz = ori[..., :3]
    else:
        return None

    return xyz


def ori2xyz(ori):
    return np.clip(ori.apply([0, 0, 1]), -1, 1)


def sph2xyz(elevation, azimuth):
    return np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ], dtype='float64')


def sph2ori(elevation, azimuth):
    return R.from_euler('ZY', np.vstack([azimuth, np.pi/2 - elevation]).T, degrees=False)


def xyz2elevation(xyz):
    return np.arcsin(xyz[..., 2])


def xyz2azimuth(xyz):
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    return (phi + np.pi) % (2 * np.pi) - np.pi


def angle_between(xyz_a, xyz_b):
    dot_product = np.dot(xyz_a, xyz_b)
    return np.arccos(dot_product)  # rad
