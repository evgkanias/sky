"""
Package that allows rendering of the skylight radiance, transmittance, and polarisation to RGBA images.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias", "Petr Vévoda", "Alexander Wilkie"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from . import SkyBase, geometry as geo
from ._static import SPECTRUM_STEP, SPECTRUM_WAVELENGTHS, SPECTRUM_CHANNELS
from ._static import SPECTRAL_RESPONSE, SPECTRAL_RESPONSE_START, SPECTRAL_RESPONSE_STEP
from ._static import XYZ2RGB, MODES

from numba import vectorize

import numpy as np


def render_pixels(sky_model, albedo, altitude, visibility, resolution, mode):
    """
    An example of using Prague Sky Model for rendering a simple fisheye RGB image of the sky.

    Examples
    --------
    >>> import skylight
    >>> s = skylight.AnalyticalSky(theta_s=np.pi/6, phi_s=np.pi)
    >>> y = render_pixels(s, 0.5, 0.0, 59.4, 3, 0)
    >>> y.shape
    (58, 3, 3)
    >>> np.round(y[:3], decimals=2)  # doctest: +ELLIPSIS
    array([[[ 5.87,  0.13,  2.64],
            [ 9.07, -0.14, -0.07],
            [ 5.87,  0.13,  2.64]],
    ...
           [[ 4.91,  1.17,  2.83],
            [ 6.97,  0.19,  1.01],
            [ 4.91,  1.17,  2.83]],
    ...
           [[ 5.02,  2.16,  3.43],
            [ 6.59,  1.36,  2.05],
            [ 5.02,  2.16,  3.43]]])
    >>> p = render_pixels(s, 0.5, 0.0, 59.4, 3, 2)
    >>> np.round(p[:3], decimals=2)  # doctest: +ELLIPSIS
    array([[[-4.000e-02,  1.324e+01, -8.000e-02],
            [ 1.200e-01,  2.478e+01,  3.100e+00],
            [-4.000e-02,  1.324e+01, -8.000e-02]],
    ...
           [[-2.000e-02,  1.162e+01,  2.000e-02],
            [-1.500e-01,  1.904e+01,  5.090e+00],
            [-2.000e-02,  1.162e+01,  2.000e-02]],
    ...
           [[ 2.650e+00,  1.234e+01,  2.860e+00],
            [ 1.900e+00,  1.802e+01,  7.350e+00],
            [ 2.650e+00,  1.234e+01,  2.860e+00]]])
    >>> a = render_pixels(s, 0.5, 0.0, 59.4, 3, 4)
    >>> np.round(a[:3], decimals=2)  # doctest: +ELLIPSIS
    array([[[0.65, 0.81, 0.37],
            [0.  , 0.  , 0.  ],
            [0.65, 0.81, 0.37]],
    ...
           [[0.  , 1.  , 1.  ],
            [1.  , 1.  , 1.  ],
            [1.  , 0.  , 0.  ]],
    ...
           [[1.  , 0.  , 0.  ],
            [1.  , 1.  , 1.  ],
            [0.  , 1.  , 1.  ]]])

    Parameters
    ----------
    sky_model : SkyBase
        Reference to the sky model object.
    albedo : float
        Ground albedo, value in range [0, 1].
    altitude : float
        Altitude of view point in meters, value in range [0, 15000].
    visibility : float
        Horizontal visibility (meteorological range) at ground level in kilometers, value in range [20, 131.8].
    resolution : int
        Length of resulting square image size in pixels.
    mode : int, str
        Rendered quantity: (0) sky radiance, (1) sun radiance, (2) polarisation, (3) transmittance, or
        (4) angle of polarisation.

    Returns
    -------
    np.ndarray[float]
        The resulting images (index 0-2 = sRGB, index 3 - <# of channels in the dataset> = individual channels).
    """

    assert sky_model.is_initialised

    if isinstance(mode, str):
        mode = MODES.index(mode.lower())

    # We are viewing the skylight from 'altitude' meters above the origin
    viewpoint = np.array([0, 0, altitude], dtype='float64')

    # Create the output buffer and initialise to zero
    # one per wavelength (mono) plus 3 (RGB of visible light)
    out_result = np.zeros((SPECTRUM_CHANNELS + 3, resolution, resolution), dtype='float64')

    xs, ys = np.meshgrid(np.arange(resolution), np.arange(resolution))

    # For each pixel of the rendered image get the corresponding direction in fisheye projection
    views_dir = pixel2dir(xs, ys, resolution)
    pixel_map = ~np.all(np.isclose(views_dir, 0), axis=2)

    # Start of for-loop vectorisation

    # If the pixel lies outside the upper hemisphere, the direction will be zero. Such pixel is kept black
    views_arr = views_dir[pixel_map, :]
    views_ori = geo.xyz2ori(views_arr)

    si = sky_model(views_ori, SPECTRUM_WAVELENGTHS, albedo, altitude, visibility, mode)

    # Based on the selected model compute spectral skylight radiance, sun radiance, polarisation, or transmittance.
    if mode == 1:  # Sun radiance
        spectrum = si.sun_radiance
    elif mode == 2:  # Polarisation
        spectrum = si.degree_of_polarisation
    elif mode == 3:  # Transmittance
        spectrum = si.transmittance
    elif mode == 4:  # Angle of polarisation
        spectrum = si.angle_of_polarisation
    else:  # default: Sky radiance
        spectrum = si.sky_radiance

    assert spectrum is not None

    if mode == 4:
        rgb = angle2rgb(spectrum)
    else:
        rgb = spectrum2rgb(spectrum)

    out_result[:3, pixel_map] = rgb

    # Store the individual channels.
    out_result[3:, pixel_map] = np.float32(spectrum)

    out_result[:, ~pixel_map] = -1

    return out_result


def pixel2dir(x, y, resolution):
    """
    Transforms pixel positions into 3D unit vector direction.
    The position of the pixel is assumed to be in an image of size resolution x resolution.

    Examples
    --------
    >>> np.round(pixel2dir(2, 2, 5), decimals=2)
    array([[[0., 0., 1.]]])
    >>> np.round(pixel2dir(0, 0, 5), decimals=2)
    array([[[0., 0., 0.]]])
    >>> np.round(pixel2dir(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]), 5), decimals=2)
    array([[[ 0.  ,  0.  ,  0.  ],
            [-0.61, -0.61,  0.52],
            [ 0.  ,  0.  ,  1.  ],
            [ 0.61,  0.61,  0.52],
            [ 0.  ,  0.  ,  0.  ]]])
    >>> np.round(pixel2dir(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), 3), decimals=2)  # doctest: +ELLIPSIS
    array([[[-0.71, -0.71,  0.06],
            [ 0.  , -0.92,  0.38],
            [ 0.71, -0.71,  0.06]],
    ...
           [[-0.92,  0.  ,  0.38],
            [ 0.  ,  0.  ,  1.  ],
            [ 0.92,  0.  ,  0.38]],
    ...
           [[-0.71,  0.71,  0.06],
            [ 0.  ,  0.92,  0.38],
            [ 0.71,  0.71,  0.06]]])

    Parameters
    ----------
    x : np.ndarray[int], int
        the horizontal offset of the pixel(s).
    y : np.ndarray[int], int
        the vertical offset of the pixel(s).
    resolution : int
        the number of pixels per size of the image.

    Returns
    -------
    np.ndarray[float]
    """

    # ensure that x and y are 2D matrices
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    if not isinstance(y, np.ndarray):
        y = np.array([y])
    while x.ndim < 2:
        x = x[None, ...]
    while y.ndim < 2:
        y = y[None, ...]

    # For each pixel of the rendered image get the corresponding direction in fisheye projection
    views_x = _pixel2x(x, y, resolution)
    views_y = _pixel2y(x, y, resolution)
    views_z = _pixel2z(x, y, resolution)

    return np.transpose(np.concatenate([[views_x], [views_y], [views_z]]), axes=(1, 2, 0))


@vectorize(['float64(int32,int32,int32)'])
def _pixel2x(x, y, resolution):
    """
    Computes direction corresponding to given pixel coordinates in up-facing or side-facing fisheye projection.

    Parameters
    ----------
    x : int
    y : int
    resolution : int

    Returns
    -------
    float
    """

    # Make circular image area in center of image.
    radius = float(resolution) / 2

    scaled_x = (float(x) + 0.5 - radius) / radius
    scaled_y = (float(y) + 0.5 - radius) / radius
    denom = scaled_x * scaled_x + scaled_y * scaled_y + 1

    if denom > 2:
        return 0.
    else:
        return 2 * scaled_x / denom


@vectorize(['float32(int32,int32,int32)'])
def _pixel2y(x, y, resolution):
    """
    Computes direction corresponding to given pixel coordinates in up-facing or side-facing fisheye projection.

    Parameters
    ----------
    x : int
    y : int
    resolution : int

    Returns
    -------
    float
    """

    # Make circular image area in center of image.
    radius = float(resolution) / 2

    scaled_x = (float(x) + 0.5 - radius) / radius
    scaled_y = (float(y) + 0.5 - radius) / radius
    denom = scaled_x * scaled_x + scaled_y * scaled_y + 1

    if denom > 2:
        return 0.
    else:
        return 2 * scaled_y / denom


@vectorize(['float32(int32,int32,int32)'])
def _pixel2z(x, y, resolution):
    """
    Computes direction corresponding to given pixel coordinates in up-facing or side-facing fisheye projection.

    Parameters
    ----------
    x : int
    y : int
    resolution : int

    Returns
    -------
    float
    """

    # Make circular image area in center of image.
    radius = float(resolution) / 2

    scaled_x = (float(x) + 0.5 - radius) / radius
    scaled_y = (float(y) + 0.5 - radius) / radius
    denom = scaled_x * scaled_x + scaled_y * scaled_y + 1

    if denom > 2:
        return 0
    else:
        return -(denom - 2) / denom


def spectrum2rgb(spectrum):
    """
    Converts given spectrum to sRGB.

    >>> np.round(spectrum2rgb(0.001 * np.ones(SPECTRUM_WAVELENGTHS.shape)), decimals=4)
    array([0.1305, 0.1003, 0.0949])
    >>> np.round(spectrum2rgb(0.001 * np.arange(len(SPECTRUM_WAVELENGTHS)) / len(SPECTRUM_WAVELENGTHS)), decimals=4)
    array([0.0198, 0.0111, 0.0059])
    >>> np.round(spectrum2rgb(0.001 * np.arange(len(SPECTRUM_WAVELENGTHS))[::-1] / len(SPECTRUM_WAVELENGTHS)), decimals=4)
    array([0.1083, 0.0873, 0.0872])

    Parameters
    ----------
    spectrum : np.ndarray[float]
        the measures for each wavelength.

    Returns
    -------
    np.ndarray[float]
    """

    # Spectrum to XYZ
    response_idx = np.int32((SPECTRUM_WAVELENGTHS - SPECTRAL_RESPONSE_START) / SPECTRAL_RESPONSE_STEP)
    response_bool = np.all([0 <= response_idx, response_idx < len(SPECTRAL_RESPONSE)], axis=0)
    xyz = np.dot(SPECTRAL_RESPONSE[response_idx[response_bool]].T, spectrum[response_bool]) * SPECTRUM_STEP

    # XYZ to RGB
    return XYZ2RGB.T.dot(xyz)


def angle2rgb(angle, amin=-np.pi/2, amax=np.pi/2):
    """
    Converts given hue angles to sRGB.

    >>> np.round(angle2rgb(np.array([0.0])), decimals=4).T
    array([[1., 0., 0.]])
    >>> np.round(angle2rgb(np.array([np.pi / 6.0])), decimals=4).T
    array([[1., 1., 0.]])
    >>> np.round(angle2rgb(np.array([np.pi / 3.0])), decimals=4).T
    array([[0., 1., 0.]])
    >>> np.round(angle2rgb(np.array([np.pi / 2.0])), decimals=4).T
    array([[0., 1., 1.]])
    >>> np.round(angle2rgb(np.array([-np.pi / 3.0])), decimals=4).T
    array([[0., 0., 1.]])
    >>> np.round(angle2rgb(np.array([-np.pi / 6.0])), decimals=4).T
    array([[1., 0., 1.]])

    Parameters
    ----------
    angle : np.ndarray[float]
        the measures for each wavelength.
    amin : float
    amax : float

    Returns
    -------
    np.ndarray[float]
    """

    angle = ((angle - amin) % (amax - amin) + amin) / (amax - amin) * 360
    h = angle % 360
    z = 1 - abs((h / 60) % 2 - 1)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    c = np.all([0 <= h, h < 60], axis=0)
    r[c] = 1.
    g[c] = z[c]
    c = np.all([60 <= h, h < 120], axis=0)
    r[c] = z[c]
    g[c] = 1.
    c = np.all([120 <= h, h < 180], axis=0)
    g[c] = 1.
    b[c] = z[c]
    c = np.all([180 <= h, h < 240], axis=0)
    g[c] = z[c]
    b[c] = 1.
    c = np.all([240 <= h, h < 300], axis=0)
    r[c] = z[c]
    b[c] = 1.
    c = np.all([300 <= h, h < 360], axis=0)
    r[c] = 1.
    b[c] = z[c]

    return np.vstack([r, g, b])


def image2texture(image, exposure=None):
    """
    Transforms an image with arbitrary real values to a texture with values in [0, 255].

    Examples
    --------
    >>> image2texture(np.array([
    ...     [[0.5, 0.1, 1.0], [0.4, 0.2, 0.9], [0.6, 0.3, 1.1]],
    ...     [[0.1, 1.0, 0.0], [1.4, 0.5, 0.1], [0.8, 0.2, 0.1]],
    ...     [[1.0, 0.5, 0.9], [0.4, 0.7, 0.2], [0.3, 1.5, 0.6]],
    ... ]))  # doctest: +ELLIPSIS
    array([[[127,  25, 255, 255],
            [102,  51, 229, 255],
            [153,  76, 255, 255]],
    ...
           [[ 25, 255,   0, 255],
            [255, 127,  25, 255],
            [204,  51,  25, 255]],
    ...
           [[255, 127, 229, 255],
            [102, 178,  51, 255],
            [ 76, 255, 153, 255]]], dtype=uint8)

    Negative values will make the pixel transparent.
    >>> image2texture(np.array([
    ...     [[0.5, 0.1, 1.0], [0.4, 0.2, 0.9], [0.6, 0.3, 1.1]],
    ...     [[0.1, 1.0, 0.0], [1.4, -0.5, 0.1], [0.8, 0.2, 0.1]],
    ...     [[1.0, 0.5, 0.9], [0.4, -0.7, 0.2], [0.3, 1.5, 0.6]],
    ... ]))  # doctest: +ELLIPSIS
    array([[[127,  25, 255, 255],
            [102,  51, 229, 255],
            [153,  76, 255, 255]],
    ...
           [[ 25, 255,   0, 255],
            [255,   0,  25,   0],
            [204,  51,  25, 255]],
    ...
           [[255, 127, 229, 255],
            [102,   0,  51,   0],
            [ 76, 255, 153, 255]]], dtype=uint8)

    Parameters
    ----------
    image : np.ndarray[float]
    exposure : float, optional

    Returns
    -------
    np.ndarray[int]
    """
    texture = np.zeros((image.shape[0], image.shape[1], 4), dtype='uint8')
    img_tran = image < 0
    img_copy = image.copy()
    img_copy[img_tran] = 0.
    if exposure is None:
        no_gamma = np.clip(img_copy, 0, 1) * 255.
    else:
        no_gamma = apply_exposure(img_copy, exposure) * 255.
    if no_gamma.ndim < 3:
        no_gamma = no_gamma[..., None]
    if img_tran.ndim > 2:
        img_tran = np.any(img_tran, axis=2)

    texture[:, :, :3] = np.uint8(np.floor(no_gamma))
    texture[~img_tran, 3] = 255
    return texture


def apply_exposure(value, exposure):
    """
    Applies the exposure (responsiveness) of the sensor to the light.

    Examples
    --------
    >>> apply_exposure(1.0, 0.0)
    1.0
    >>> apply_exposure(1.0, 2.0)
    1.0
    >>> apply_exposure(1.0, 1.0)
    1.0
    >>> apply_exposure(1.0, -1.0)  # doctest: +ELLIPSIS
    0.72974...
    >>> apply_exposure(0.1, 0.0)  # doctest: +ELLIPSIS
    0.35111...
    >>> apply_exposure(0.1, -1.0)  # doctest: +ELLIPSIS
    0.25622...
    >>> apply_exposure(0.1, 1.0)  # doctest: +ELLIPSIS
    0.48115...

    Parameters
    ----------
    value : np.ndarray[float], float
        the value of a light property
    exposure : float
        the amount of exposure

    Returns
    -------
    np.ndarray[float]
    """
    exp_mult = np.float32(np.power(2.0, exposure))
    return np.clip(np.power(value * exp_mult, 1.0 / 2.2), 0, 1)
