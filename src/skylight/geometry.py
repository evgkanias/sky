"""
Package that allows geometrical computations and transformations.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from scipy.spatial.transform import Rotation as R

import numpy as np


def get_coordinates(ori):
    """
    Transforms Rotation instances into a 3D unit vectors.
    If the array is already a vector, it keeps only the first 3 dimensions.

    Examples
    --------
    >>> r = R.from_euler('ZY', [0, 90], degrees=True)
    >>> np.round(get_coordinates(r), decimals=2)
    array([1., 0., 0.])
    >>> r = R.from_euler('ZY', [90, 90], degrees=True)
    >>> np.round(get_coordinates(r), decimals=2)
    array([0., 1., 0.])
    >>> r = R.from_euler('ZY', [0, 0], degrees=True)
    >>> np.round(get_coordinates(r), decimals=2)
    array([0., 0., 1.])
    >>> np.round(get_coordinates(np.array([0., 0., 1.])), decimals=2)
    array([0., 0., 1.])
    >>> np.round(get_coordinates(np.array([0., 0., 1., 1., 0.])), decimals=2)
    array([0., 0., 1.])
    >>> get_coordinates(np.array([0., 1.]))

    Parameters
    ----------
    ori : R, np.ndarray[float]
        an array of Rotation instances.

    Returns
    -------
    np.ndarray[float]
    """

    if isinstance(ori, R):
        xyz = ori2xyz(ori)
    elif isinstance(ori, np.ndarray) and ori.shape[-1] > 2:
        xyz = ori[..., :3]
    else:
        return None

    return xyz


def ori2xyz(ori):
    """
    Transforms Rotation instances to 3D unit vectors.

    Examples
    --------
    >>> r = R.from_euler('ZY', [0, 90], degrees=True)
    >>> np.round(ori2xyz(r), decimals=2)
    array([1., 0., 0.])
    >>> r = R.from_euler('ZY', [0, -90], degrees=True)
    >>> np.round(ori2xyz(r), decimals=2)
    array([-1.,  0.,  0.])
    >>> r = R.from_euler('ZY', [90, 90], degrees=True)
    >>> np.round(ori2xyz(r), decimals=2)
    array([0., 1., 0.])
    >>> r = R.from_euler('ZY', [-90, 90], degrees=True)
    >>> np.round(ori2xyz(r), decimals=2)
    array([ 0., -1.,  0.])
    >>> r = R.from_euler('ZY', [0, 0], degrees=True)
    >>> np.round(ori2xyz(r), decimals=2)
    array([0., 0., 1.])
    >>> r = R.from_euler('ZY', [0, 180], degrees=True)
    >>> np.round(ori2xyz(r), decimals=2)
    array([ 0.,  0., -1.])

    Parameters
    ----------
    ori : R
        an array of Rotation instances.

    Returns
    -------
    np.ndarray[float]
    """
    return np.clip(ori.apply([0, 0, 1]), -1, 1)


def sph2xyz(elevation, azimuth, degrees=False):
    """
    Transforms spherical coordinates (elevation above ground and azimuth) into a 3D unit vector.

    Examples
    --------
    >>> np.round(sph2xyz(0, 0, degrees=True), decimals=2)
    array([1., 0., 0.])
    >>> np.round(sph2xyz(0, 180, degrees=True), decimals=2)
    array([-1.,  0.,  0.])
    >>> np.round(sph2xyz(0, 90, degrees=True), decimals=2)
    array([0., 1., 0.])
    >>> np.round(sph2xyz(0, -90, degrees=True), decimals=2)
    array([ 0., -1.,  0.])
    >>> np.round(sph2xyz(90, 0, degrees=True), decimals=2)
    array([0., 0., 1.])
    >>> np.round(sph2xyz(-90, 0, degrees=True), decimals=2)
    array([ 0.,  0., -1.])

    Parameters
    ----------
    elevation : np.ndarray[float], float
        an array or scalar of the elevation(s) above the ground level.
    azimuth : np.ndarray[float], float
        an array or scalar of the azimuth(s).
    degrees : bool
        if True, the elevation and azimuth will be interpreted as degrees, otherwise as radians.

    Returns
    -------
    np.ndarray[float]
    """
    if degrees:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    return np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ], dtype='float64')


def sph2ori(elevation, azimuth, degrees=False):
    """
    Transforms spherical coordinates (elevation above ground and azimuth) into Rotation instances.

    Examples
    --------
    >>> np.round(sph2ori(0, 0, degrees=True).as_euler('ZYX', degrees=True), decimals=2)
    array([[ 0., 90.,  0.]])
    >>> np.round(sph2ori(0, 180, degrees=True).as_euler('ZYX', degrees=True), decimals=2)
    array([[180.,  90.,   0.]])
    >>> np.round(sph2ori(0, 90, degrees=True).as_euler('ZYX', degrees=True), decimals=2)
    array([[90., 90.,  0.]])
    >>> np.round(sph2ori(0, -90, degrees=True).as_euler('ZYX', degrees=True), decimals=2)
    array([[-90.,  90.,   0.]])
    >>> np.round(sph2ori(90, 0, degrees=True).as_euler('ZYX', degrees=True), decimals=2)
    array([[0., 0., 0.]])
    >>> np.round(sph2ori(-90, 0, degrees=True).as_euler('ZYX', degrees=True), decimals=2)
    array([[180.,   0., 180.]])

    Parameters
    ----------
    elevation : np.ndarray[float], float
        an array or scalar of the elevation(s) above the ground level.
    azimuth : np.ndarray[float], float
        an array or scalar of the azimuth(s).
    degrees : bool
        if True, the elevation and azimuth will be interpreted as degrees, otherwise as radians.

    Returns
    -------
    R
    """
    d90 = np.pi/2
    if degrees:
        d90 = np.rad2deg(d90)
    return R.from_euler('ZY', np.vstack([azimuth, d90 - elevation]).T, degrees=degrees)


def xyz2ori(xyz):
    """
    Transforms 3D init vectors into Rotation instances.

    Examples
    --------
    >>> np.round(ori2xyz(xyz2ori(np.array([1., 0., 0.]))), decimals=2)
    array([[1., 0., 0.]])
    >>> np.round(ori2xyz(xyz2ori(np.array([-1., 0., 0.]))), decimals=2)
    array([[-1.,  0.,  0.]])
    >>> np.round(ori2xyz(xyz2ori(np.array([0., 1., 0.]))), decimals=2)
    array([[0., 1., 0.]])
    >>> np.round(ori2xyz(xyz2ori(np.array([0., -1., 0.]))), decimals=2)
    array([[ 0., -1.,  0.]])
    >>> np.round(ori2xyz(xyz2ori(np.array([0., 0., 1.]))), decimals=2)
    array([[0., 0., 1.]])
    >>> np.round(ori2xyz(xyz2ori(np.array([0., 0., -1.]))), decimals=2)
    array([[ 0.,  0., -1.]])

    Parameters
    ----------
    xyz : np.ndarray[float]
        a 2D unit vector

    Returns
    -------
    R
    """
    azimuth = np.arctan2(xyz[..., 1], xyz[..., 0])
    zenith = np.arccos(xyz[..., 2])
    return R.from_euler('ZY', np.vstack([azimuth, zenith]).T, degrees=False)


def xyz2elevation(xyz):
    """
    Extracts the elevation above the ground (pitch) of 3D unit vectors.

    Examples
    --------
    >>> np.round(np.rad2deg(xyz2elevation(np.array([1., 0., 0.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(xyz2elevation(np.array([-1., 0., 0.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(xyz2elevation(np.array([0., 1., 0.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(xyz2elevation(np.array([0., -1., 0.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(xyz2elevation(np.array([0., 0., 1.]))), decimals=2)
    90.0
    >>> np.round(np.rad2deg(xyz2elevation(np.array([0., 0., -1.]))), decimals=2)
    -90.0

    Parameters
    ----------
    xyz : np.ndarray[float]
        array of 3D unit vectors.

    Returns
    -------
    np.ndarray[float]
        the elevations of the input vectors above the ground.
    """
    return np.arcsin(xyz[..., 2])


def xyz2azimuth(xyz):
    """
    Extracts the azimuth (yaw) of 3D unit vectors.

    Examples
    --------
    >>> np.round(np.rad2deg(xyz2azimuth(np.array([1., 0., 0.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(xyz2azimuth(np.array([-1., 0., 0.]))), decimals=2)
    -180.0
    >>> np.round(np.rad2deg(xyz2azimuth(np.array([0., 1., 0.]))), decimals=2)
    90.0
    >>> np.round(np.rad2deg(xyz2azimuth(np.array([0., -1., 0.]))), decimals=2)
    -90.0
    >>> np.round(np.rad2deg(xyz2azimuth(np.array([0., 0., 1.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(xyz2azimuth(np.array([0., 0., -1.]))), decimals=2)
    0.0

    Parameters
    ----------
    xyz : np.ndarray[float]
        array of 3D unit vectors.

    Returns
    -------
    np.ndarray[float]
        the azimuths of the input vectors.
    """
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    return (phi + np.pi) % (2 * np.pi) - np.pi


def angle_between(xyz_a, xyz_b):
    """
    Calculates the angle between two 3D unit vectors (in rad).

    Examples
    --------
    >>> np.round(np.rad2deg(angle_between(np.array([1., 0., 0.]), np.array([1., 0., 0.]))), decimals=2)
    0.0
    >>> np.round(np.rad2deg(angle_between(np.array([1., 0., 0.]), np.array([-1., 0., 0.]))), decimals=2)
    180.0
    >>> np.round(np.rad2deg(angle_between(np.array([1., 0., 0.]), np.array([0., 1., 0.]))), decimals=2)
    90.0
    >>> np.round(np.rad2deg(angle_between(np.array([1., 0., 0.]), np.array([0., 0., -1.]))), decimals=2)
    90.0

    Parameters
    ----------
    xyz_a : np.ndarray[float]
        array of 3D unit vectors.
    xyz_b : np.ndarray[float]
        another array of 3D unit vectors of the same size, or a single 3D unit vector.

    Returns
    -------
    np.ndarray[float]
        the angular distance (in rad) of the elementwise
    """
    dot_product = np.sum(xyz_a * xyz_b, axis=-1)
    return np.arccos(dot_product)  # rad
