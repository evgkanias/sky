from .prague import PragueSkyModel

from numba import vectorize, njit, prange

import numpy as np

# We use 55-channel spectrum for querying the model.
# The wavelengths samples are placed at centers of spectral bins used by the model.
SPECTRUM_STEP: int = 40
SPECTRUM_WAVELENGTHS = np.arange(300, 2461, SPECTRUM_STEP)
SPECTRUM_CHANNELS: int = len(SPECTRUM_WAVELENGTHS)

# Spectral response table used for converting spectrum to XYZ.
SPECTRAL_RESPONSE = np.array([
    [0.000129900000, 0.000003917000, 0.000606100000],
    [0.000232100000, 0.000006965000, 0.001086000000],
    [0.000414900000, 0.000012390000, 0.001946000000],
    [0.000741600000, 0.000022020000, 0.003486000000],
    [0.001368000000, 0.000039000000, 0.006450001000],
    [0.002236000000, 0.000064000000, 0.010549990000],
    [0.004243000000, 0.000120000000, 0.020050010000],
    [0.007650000000, 0.000217000000, 0.036210000000],
    [0.014310000000, 0.000396000000, 0.067850010000],
    [0.023190000000, 0.000640000000, 0.110200000000],
    [0.043510000000, 0.001210000000, 0.207400000000],
    [0.077630000000, 0.002180000000, 0.371300000000],
    [0.134380000000, 0.004000000000, 0.645600000000],
    [0.214770000000, 0.007300000000, 1.039050100000],
    [0.283900000000, 0.011600000000, 1.385600000000],
    [0.328500000000, 0.016840000000, 1.622960000000],
    [0.348280000000, 0.023000000000, 1.747060000000],
    [0.348060000000, 0.029800000000, 1.782600000000],
    [0.336200000000, 0.038000000000, 1.772110000000],
    [0.318700000000, 0.048000000000, 1.744100000000],
    [0.290800000000, 0.060000000000, 1.669200000000],
    [0.251100000000, 0.073900000000, 1.528100000000],
    [0.195360000000, 0.090980000000, 1.287640000000],
    [0.142100000000, 0.112600000000, 1.041900000000],
    [0.095640000000, 0.139020000000, 0.812950100000],
    [0.057950010000, 0.169300000000, 0.616200000000],
    [0.032010000000, 0.208020000000, 0.465180000000],
    [0.014700000000, 0.258600000000, 0.353300000000],
    [0.004900000000, 0.323000000000, 0.272000000000],
    [0.002400000000, 0.407300000000, 0.212300000000],
    [0.009300000000, 0.503000000000, 0.158200000000],
    [0.029100000000, 0.608200000000, 0.111700000000],
    [0.063270000000, 0.710000000000, 0.078249990000],
    [0.109600000000, 0.793200000000, 0.057250010000],
    [0.165500000000, 0.862000000000, 0.042160000000],
    [0.225749900000, 0.914850100000, 0.029840000000],
    [0.290400000000, 0.954000000000, 0.020300000000],
    [0.359700000000, 0.980300000000, 0.013400000000],
    [0.433449900000, 0.994950100000, 0.008749999000],
    [0.512050100000, 1.000000000000, 0.005749999000],
    [0.594500000000, 0.995000000000, 0.003900000000],
    [0.678400000000, 0.978600000000, 0.002749999000],
    [0.762100000000, 0.952000000000, 0.002100000000],
    [0.842500000000, 0.915400000000, 0.001800000000],
    [0.916300000000, 0.870000000000, 0.001650001000],
    [0.978600000000, 0.816300000000, 0.001400000000],
    [1.026300000000, 0.757000000000, 0.001100000000],
    [1.056700000000, 0.694900000000, 0.001000000000],
    [1.062200000000, 0.631000000000, 0.000800000000],
    [1.045600000000, 0.566800000000, 0.000600000000],
    [1.002600000000, 0.503000000000, 0.000340000000],
    [0.938400000000, 0.441200000000, 0.000240000000],
    [0.854449900000, 0.381000000000, 0.000190000000],
    [0.751400000000, 0.321000000000, 0.000100000000],
    [0.642400000000, 0.265000000000, 0.000049999990],
    [0.541900000000, 0.217000000000, 0.000030000000],
    [0.447900000000, 0.175000000000, 0.000020000000],
    [0.360800000000, 0.138200000000, 0.000010000000],
    [0.283500000000, 0.107000000000, 0.000000000000],
    [0.218700000000, 0.081600000000, 0.000000000000],
    [0.164900000000, 0.061000000000, 0.000000000000],
    [0.121200000000, 0.044580000000, 0.000000000000],
    [0.087400000000, 0.032000000000, 0.000000000000],
    [0.063600000000, 0.023200000000, 0.000000000000],
    [0.046770000000, 0.017000000000, 0.000000000000],
    [0.032900000000, 0.011920000000, 0.000000000000],
    [0.022700000000, 0.008210000000, 0.000000000000],
    [0.015840000000, 0.005723000000, 0.000000000000],
    [0.011359160000, 0.004102000000, 0.000000000000],
    [0.008110916000, 0.002929000000, 0.000000000000],
    [0.005790346000, 0.002091000000, 0.000000000000],
    [0.004109457000, 0.001484000000, 0.000000000000],
    [0.002899327000, 0.001047000000, 0.000000000000],
    [0.002049190000, 0.000740000000, 0.000000000000],
    [0.001439971000, 0.000520000000, 0.000000000000],
    [0.000999949300, 0.000361100000, 0.000000000000],
    [0.000690078600, 0.000249200000, 0.000000000000],
    [0.000476021300, 0.000171900000, 0.000000000000],
    [0.000332301100, 0.000120000000, 0.000000000000],
    [0.000234826100, 0.000084800000, 0.000000000000],
    [0.000166150500, 0.000060000000, 0.000000000000],
    [0.000117413000, 0.000042400000, 0.000000000000],
    [0.000083075270, 0.000030000000, 0.000000000000],
    [0.000058706520, 0.000021200000, 0.000000000000],
    [0.000041509940, 0.000014990000, 0.000000000000],
    [0.000029353260, 0.000010600000, 0.000000000000],
    [0.000020673830, 0.000007465700, 0.000000000000],
    [0.000014559770, 0.000005257800, 0.000000000000],
    [0.000010253980, 0.000003702900, 0.000000000000],
    [0.000007221456, 0.000002607800, 0.000000000000],
    [0.000005085868, 0.000001836600, 0.000000000000],
    [0.000003581652, 0.000001293400, 0.000000000000],
    [0.000002522525, 0.000000910930, 0.000000000000],
    [0.000001776509, 0.000000641530, 0.000000000000],
    [0.000001251141, 0.000000451810, 0.000000000000]
], dtype='float64')
SPECTRAL_RESPONSE_START = 360.0
SPECTRAL_RESPONSE_STEP = 5.0

MODES = [
    "sky radiance",
    "sun radiance",
    "polarisation",
    "transmittance"
]
"""
Rendered quantity
"""

XYZ2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
], dtype='float64').T


def render(sky_model, albedo, altitude, azimuth, elevation, visibility, resolution, mode):
    """
    An example of using Prague Sky Model for rendering a simple fisheye RGB image of the sky.

    Parameters
    ----------
    sky_model : PragueSkyModel
        Reference to the sky model object.
    albedo : float
        Ground albedo, value in range [0, 1].
    altitude : float
        Altitude of view point in meters, value in range [0, 15000].
    azimuth : float
        Sun azimuth at view point in degrees, value in range [0, 360].
    elevation : float
        Sun elevation at view point in degrees, value in range [-4.2, 90].
    visibility : float
        Horizontal visibility (meteorological range) at ground level in kilometers, value in range [20, 131.8].
    resolution : int
        Length of resulting square image size in pixels.
    mode : int, str
        Rendered quantity: (1) sky radiance, (2) sun radiance, (3) polarisation, or (4) transmittance.

    Returns
    -------
    np.ndarray[float]
        The resulting images (index 0-2 = sRGB, index 3 - <# of channels in the dataset> = individual channels).
    """

    assert sky_model.is_initialised

    if isinstance(mode, str):
        mode = MODES.index(mode.lower())

    # We are viewing the sky from 'altitude' meters above the origin
    viewpoint = np.array([0, 0, altitude])

    # Create the output buffer and initialise to zero
    # one per wavelength (mono) plus 3 (RGB of visible light)
    out_result = np.zeros((SPECTRUM_CHANNELS + 3, resolution, resolution), dtype='float64')

    xs, ys = np.meshgrid(np.arange(resolution), np.arange(resolution))

    # For each pixel of the rendered image get the corresponding direction in fisheye projection
    views_x = pixel2x(xs, ys, resolution)
    views_y = pixel2y(xs, ys, resolution)
    views_z = pixel2z(xs, ys, resolution)
    views_dir = np.transpose(np.concatenate([[views_x], [views_y], [views_z]]), axes=(1, 2, 0))
    pixel_map = ~np.all(np.isclose(views_dir, 0), axis=2)

    # Start of for-loop vectorisation

    # If the pixel lies outside the upper hemisphere, the direction will be zero. Such pixel is kept black
    views_arr = views_dir[pixel_map, :]

    # Get internal model parameters for the desired confirmation.
    params = sky_model.compute_parameters(viewpoint, views_arr, elevation, azimuth, visibility, albedo)

    # Based on the selected model compute spectral sky radiance, sun radiance, polarisation, or transmittance.
    if mode == 1:  # Sun radiance
        spectrum = sky_model.sun_radiance(params, SPECTRUM_WAVELENGTHS)
    elif mode == 2:  # Polarisation
        spectrum = np.abs(sky_model.polarisation(params, SPECTRUM_WAVELENGTHS))
    elif mode == 3:  # Transmittance
        spectrum = sky_model.transmittance(params, SPECTRUM_WAVELENGTHS, np.finfo(float).max)
    else:  # default: Sky radiance
        spectrum = sky_model.sky_radiance(params, SPECTRUM_WAVELENGTHS)

    rgb = spectrum2rgb(spectrum)
    out_result[:3, pixel_map] = rgb

    # Store the individual channels.
    out_result[3:, pixel_map] = np.float32(spectrum)

    # for x in range(resolution):
    #     for y in range(resolution):
    #
    #         # If the pixel lies outside the upper hemisphere, the direction will be zero. Such pixel is kept black
    #         if not pixel_map[x, y]:
    #             continue
    #
    #         # Get internal model parameters for the desired confirmation.
    #         params = sky_model.compute_parameters(viewpoint, views_dir[x, y], elevation, azimuth, visibility, albedo)
    #
    #         # Based on the selected model compute spectral sky radiance, sun radiance, polarisation, or transmittance
    #         spectrum = np.zeros(SPECTRUM_CHANNELS, dtype='float32')
    #         for wl in range(SPECTRUM_CHANNELS):
    #             spectrum[wl] = get_spectral_values(wl, params, sky_model, mode)
    #         # spectrum = get_spectral_values(np.arange(SPECTRUM_CHANNELS), params, sky_model, mode)
    #
    #         # Convert the spectral quantity to sRGB and store it at 0 in the result buffer.
    #         rgb = spectrum2rgb(spectrum)
    #         out_result[:3, x, y] = rgb
    #
    #         # Store the individual channels.
    #         for c in range(SPECTRUM_CHANNELS):
    #             out_result[c + 3, x, y] = float(spectrum[c])

    return out_result


@vectorize(['float64(int32,int32,int32)'])
def pixel2x(x, y, resolution):
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
def pixel2y(x, y, resolution):
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
def pixel2z(x, y, resolution):
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


def image2texture(image: np.ndarray, exposure: float):
    texture = np.zeros((image.shape[0], image.shape[1], 4), dtype='uint8')
    exp_mult = np.float32(np.power(2, exposure))
    no_gamma = np.clip(np.power(image * exp_mult, 1.0 / 2.2) * 255., 0, 255)
    if no_gamma.ndim < 3:
        no_gamma = no_gamma[..., None]

    texture[:, :, :3] = np.uint8(np.floor(no_gamma))
    texture[~np.all(np.isclose(texture[:, :, :3], 0), axis=2), 3] = 255
    return texture


# @vectorize(['np.ndarray[3, float](float32,float32,float32)'])
# def pixel2direction(x, y, resolution):
#     """
#     Computes direction corresponding to given pixel coordinates in up-facing or side-facing fisheye projection.
#
#     Parameters
#     ----------
#     x : float
#     y : float
#     resolution : float
#
#     Returns
#     -------
#     np.ndarray[float]
#     """
#
#     # Make circular image area in center of image.
#     radius = resolution / 2
#     dtype = np.dtype('float32')
#
#     scaled_x = (x + 0.5 - radius) / radius
#     scaled_y = (y + 0.5 - radius) / radius
#     denom = scaled_x * scaled_x + scaled_y * scaled_y + 1
#
#     d = np.zeros(3, dtype=dtype)
#     if denom <= 2:
#         d = np.array([
#             2 * scaled_x / denom,
#             2 * scaled_y / denom,
#             -(denom - 2) / denom
#         ], dtype=dtype)
#
#     return d


# def get_spectral_values(wl, params, sky_model, mode):
#     """
#     Based on the selected model compute spectral sky radiance, sun radiance, polarisation, or transmittance.
#
#     Parameters
#     ----------
#     wl : int
#     params : Parameters
#     sky_model : PragueSkyModel
#     mode : int
#
#     Returns
#     -------
#
#     """
#     if mode == 1:  # Sun radiance
#         return sky_model.sun_radiance(params, SPECTRUM_WAVELENGTHS[wl])
#     elif mode == 2:  # Polarisation
#         return np.abs(sky_model.polarisation(params, SPECTRUM_WAVELENGTHS[wl]))
#     elif mode == 3:  # Transmittance
#         return sky_model.transmittance(params, SPECTRUM_WAVELENGTHS[wl], np.finfo(float).max)
#     else:  # default: Sky radiance
#         return sky_model.sky_radiance(params, SPECTRUM_WAVELENGTHS[wl])
