from .__helpers import PSM_SUN_RADIUS, PSM_PLANET_RADIUS, PSM_PLANET_RADIUS_SQR, print_error_and_exit
from .__helpers import PSM_MIN_ALTITUDE, PSM_MAX_ALTITUDE, PSM_LIGHTCOLLECTION_VERTICAL_STEPSIZE, PSM_ARRAYSIZE

from collections import namedtuple

import numpy as np

SkyModelState = namedtuple("SkyModelState",
                           # Radiance metadata
                           "turbidities "
                           "albedos "
                           "altitudes "
                           "elevations "
                           "channels channel_start channel_width "
                           "tensor_components "
                           "sun_breaks sun_offset sun_stride "
                           "zenith_breaks zenith_offset zenith_stride "
                           "emph_breaks emph_offset "
                           "total_coefs_single_config "  # this is for one specific configuration
                           "total_coefs_all_configs total_configs "
                           
                           # Radiance data
                           "radiance_dataset "
                           
                           # Transmittance metadata
                           "trans_n_a trans_n_d trans_turbidities trans_altitudes trans_rank "
                           "transmission_altitudes transmission_turbities "
                           
                           # Transmittance data
                           "transmission_dataset_U transmission_dataset_V "
                           
                           # Polarisation metadata
                           "tensor_components_pol "
                           "sun_breaks_pol sun_offset_pol sun_stride_pol "
                           "zenith_breaks_pol zenith_offset_pol zenith_stride_pol "
                           "total_coefs_single_config_pol total_coefs_all_configs_pol "
                           
                           # Polarisation data
                           "polarisation_dataset")
"""
One blob of floats for each wavelength and task
"""

AltitudeAndElevation = namedtuple("AltitudeAndElevation",
                                  "solar_elevation_at_viewpoint altitude_of_viewpoint distance_to_view "
                                  "direction_to_zenith_n direction_to_sun_n")

SolarAngles = namedtuple("SolarAngles",
                         "solar_elevation_at_viewpoint altitude_of_viewpoint "
                         "theta gamma shadow zero")

trans_svd_rank = 12
safety_altitude = 0.0

sun_rad_star_wl = 310
sun_rad_increment_wl = 1
sun_rad_table = [
    9829.41, 10184., 10262.6, 10375.7, 10276., 10179.3, 10156.6, 10750.7, 11134., 11463.6, 11860.4, 12246.2, 12524.4,
    12780., 13187.4, 13632.4, 13985.9, 13658.3, 13377.4, 13358.3, 13239., 13119.8, 13096.2, 13184., 13243.5, 13018.4,
    12990.4, 13159.1, 13230.8, 13258.6, 13209.9, 13343.2, 13404.8, 13305.4, 13496.3, 13979.1, 14153.8, 14188.4, 14122.7,
    13825.4, 14033.3, 13914.1, 13837.4, 14117.2, 13982.3, 13864.5, 14118.4, 14545.7, 15029.3, 15615.3, 15923.5, 16134.8,
    16574.5, 16509., 16336.5, 16146.6, 15965.1, 15798.6, 15899.8, 16125.4, 15854.3, 15986.7, 15739.7, 15319.1, 15121.5,
    15220.2, 15041.2, 14917.7, 14487.8, 14011., 14165.7, 14189.5, 14540.7, 14797.5, 14641.5, 14761.6, 15153.7, 14791.8,
    14907.6, 15667.4, 16313.5, 16917., 17570.5, 18758.1, 20250.6, 21048.1, 21626.1, 22811.6, 23577.2, 23982.6, 24062.1,
    23917.9, 23914.1, 23923.2, 24052.6, 24228.6, 24360.8, 24629.6, 24774.8, 24648.3, 24666.5, 24938.6, 24926.3, 24693.1,
    24613.5, 24631.7, 24569.8, 24391.5, 24245.7, 24084.4, 23713.7, 22985.4, 22766.6, 22818.9, 22834.3, 22737.9, 22791.6,
    23086.3, 23377.7, 23461., 23935.5, 24661.7, 25086.9, 25520.1, 25824.3, 26198., 26350.2, 26375.4, 26731.2, 27250.4,
    27616., 28145.3, 28405.9, 28406.8, 28466.2, 28521.5, 28783.8, 29025.1, 29082.6, 29081.3, 29043.1, 28918.9, 28871.6,
    29049., 29152.5, 29163.2, 29143.4, 28962.7, 28847.9, 28854., 28808.7, 28624.1, 28544.2, 28461.4, 28411.1, 28478.,
    28469.8, 28513.3, 28586.5, 28628.6, 28751.5, 28948.9, 29051., 29049.6, 29061.7, 28945.7, 28672.8, 28241.5, 27903.2,
    27737., 27590.9, 27505.6, 27270.2, 27076.2, 26929.1, 27018.2, 27206.8, 27677.2, 27939.9, 27923.9, 27899.2, 27725.4,
    27608.4, 27599.4, 27614.6, 27432.4, 27460.4, 27392.4, 27272., 27299.1, 27266.8, 27386.5, 27595.9, 27586.9, 27504.8,
    27480.6, 27329.8, 26968.4, 26676.3, 26344.7, 26182.5, 26026.3, 25900.3, 25842.9, 25885.4, 25986.5, 26034.5, 26063.5,
    26216.9, 26511.4, 26672.7, 26828.5, 26901.8, 26861.5, 26865.4, 26774.2, 26855.8, 27087.1, 27181.3, 27183.1, 27059.8,
    26834.9, 26724.3, 26759.6, 26725.9, 26724.6, 26634.5, 26618.5, 26560.1, 26518.7, 26595.3, 26703.2, 26712.7, 26733.9,
    26744.3, 26764.4, 26753.2, 26692.7, 26682.7, 26588.1, 26478., 26433.7, 26380.7, 26372.9, 26343.3, 26274.7, 26162.3,
    26160.5, 26210., 26251.2, 26297.9, 26228.9, 26222.3, 26269.7, 26295.6, 26317.9, 26357.5, 26376.1, 26342.4, 26303.5,
    26276.7, 26349.2, 26390., 26371.6, 26346.7, 26327.6, 26274.2, 26247.3, 26228.7, 26152.1, 25910.3, 25833.2, 25746.5,
    25654.3, 25562., 25458.8, 25438., 25399.1, 25324.3, 25350., 25514., 25464.9, 25398.5, 25295.2, 25270.2, 25268.4,
    25240.6, 25184.9, 25149.6, 25123.9, 25080.3, 25027.9, 25012.3, 24977.9, 24852.6, 24756.4, 24663.5, 24483.6, 24398.6,
    24362.6, 24325.1, 24341.7, 24288.7, 24284.2, 24257.3, 24178.8, 24097.6, 24175.6, 24175.7, 24139.7, 24088.1, 23983.2,
    23902.7, 23822.4, 23796.2, 23796.9, 23814.5, 23765.5, 23703., 23642., 23592.6, 23552., 23514.6, 23473.5, 23431.,
    23389.3, 23340., 23275.1, 23187.3, 23069.5, 22967., 22925.3, 22908.9, 22882.5, 22825., 22715.4, 22535.5, 22267.1,
    22029.4, 21941.6, 21919.5, 21878.8, 21825.6, 21766., 21728.9, 21743.2, 21827.1, 21998.7, 22159.4, 22210., 22187.2,
    22127.2, 22056.2, 22000.2, 21945.9, 21880.2, 21817.1, 21770.3, 21724.3, 21663.2, 21603.3, 21560.4, 21519.8, 21466.2,
    21401.6, 21327.7, 21254.2, 21190.7, 21133.6, 21079.3, 21024., 20963.7, 20905.5, 20856.6, 20816.6, 20785.2, 20746.7,
    20685.3, 20617.8, 20561.1, 20500.4, 20421.2, 20333.4, 20247., 20175.3, 20131.4, 20103.2, 20078.5, 20046.8, 19997.2,
    19952.9, 19937.2, 19930.8, 19914.4, 19880.8, 19823., 19753.8, 19685.9, 19615.3, 19537.5, 19456.8, 19377.6, 19309.4,
    19261.9, 19228., 19200.5, 19179.5, 19164.8, 19153.1, 19140.6, 19129.2, 19120.6, 19104.5, 19070.6, 19023.9, 18969.3,
    18911.4, 18855., 18798.6, 18740.8, 18672.7, 18585.2, 18501., 18442.4, 18397.5, 18353.9, 18313.2, 18276.8, 18248.3,
    18231.2, 18224., 18225.4, 18220.1, 18192.6, 18155.1, 18119.8, 18081.6, 18035.6, 17987.4, 17942.8, 17901.7, 17864.2,
    17831.1, 17802.9, 17771.5, 17728.6, 17669.7, 17590.1, 17509.5, 17447.4, 17396., 17347.4, 17300.3, 17253.2, 17206.1,
    17159., 17127.6, 17127.6, 17133.6, 17120.4, 17097.2, 17073.3, 17043.7, 17003.4, 16966.3, 16946.3, 16930.9, 16907.7,
    16882.7, 16862., 16837.8, 16802.1, 16759.2, 16713.6, 16661.8, 16600.8, 16542.6, 16499.4, 16458.7, 16408., 16360.6,
    16329.5, 16307.4, 16286.7, 16264.9, 16239.6, 16207.8, 16166.8, 16118.2, 16064., 16011.2, 15966.9, 15931.9, 15906.9,
    15889.1, 15875.5, 15861.2, 15841.3, 15813.1, 15774.2, 15728.8, 15681.4, 15630., 15572.9, 15516.5, 15467.2, 15423.,
    15381.6, 15354.4, 15353., 15357.3, 15347.3, 15320.2, 15273.1, 15222., 15183.1, 15149.6, 15114.6, 15076.8, 15034.6,
    14992.9
]


def compute_altitude_and_elevation(viewpoint, ground_level_solar_elevation_at_origin,
                                   ground_level_solar_azimuth_at_origin):
    """
    Computes the canonical angles of the model from a normalised view vector and solar elevation.

    Parameters
    ----------
    viewpoint : np.ndarray
    ground_level_solar_elevation_at_origin : float
    ground_level_solar_azimuth_at_origin : float

    Returns
    -------
    solar_elevation_at_viewpoint : float
    altitude_of_viewpoint : float
    distance_to_view : float
    direction_to_zenith_n : np.ndarray
    direction_to_sun_n : np.ndarray
    """

    # Direction to zenith
    centre_of_earth = np.array([0, 0, -PSM_PLANET_RADIUS], dtype='float32')
    direction_to_zenith = viewpoint - centre_of_earth
    distance_to_view = np.linalg.norm(direction_to_zenith)
    direction_to_zenith_n = direction_to_zenith / distance_to_view

    # Altitude of viewpoint
    altitude_of_viewpoint = np.maximum(distance_to_view - PSM_PLANET_RADIUS, 0)

    # Direction to sun
    direction_to_sun_n = np.array([np.cos(ground_level_solar_azimuth_at_origin) *
                                   np.cos(ground_level_solar_elevation_at_origin),
                                   np.sin(ground_level_solar_azimuth_at_origin) *
                                   np.cos(ground_level_solar_elevation_at_origin),
                                   np.sin(ground_level_solar_elevation_at_origin)])

    # Solar elevation at viewpoint (more precisely, solar elevation at the point on the ground directly below viewpoint)
    dot_zenith_sun = np.dot(direction_to_zenith_n, direction_to_sun_n)

    solar_elevation_at_viewpoint = 0.5 * np.pi - np.arccos(dot_zenith_sun)

    return AltitudeAndElevation(solar_elevation_at_viewpoint, altitude_of_viewpoint, distance_to_view,
                                direction_to_zenith_n, direction_to_sun_n)


def compute_angles(viewpoint, view_direction, ground_level_solar_elevation_at_origin,
                   ground_level_solar_azimuth_at_origin):
    """
    Computes the canonical angles of the model from a normalised view vector and solar elevation.

    Parameters
    ----------
    viewpoint : np.ndarray
    view_direction : np.ndarray
    ground_level_solar_elevation_at_origin : float
    ground_level_solar_azimuth_at_origin : float

    Returns
    -------
    solar_elevation_at_viewpoint : float
    altitude_of_viewpoint : float
    theta : float
        zenith angle
    gamma : float
        sun angle
    shadow : float
        angle from the shadow point, which is further 90 degrees above the sun
    zero : float
        angle from the zero point, which lies at the horizon below the sun
    """

    # Shift viewpoint about safety altitude up
    centre_of_earth = np.array([0, 0, -PSM_PLANET_RADIUS])
    to_viewpoint = viewpoint - centre_of_earth
    to_viewpoint_n = to_viewpoint / np.linalg.norm(to_viewpoint)

    distance_to_view_tmp = np.linalg.norm(to_viewpoint) + safety_altitude
    to_shifted_viewpoint = distance_to_view_tmp * to_viewpoint_n
    shifted_viewpoint = to_shifted_viewpoint + centre_of_earth

    view_direction_n = view_direction / np.linalg.norm(view_direction)

    (solar_elevation_at_viewpoint, altitude_of_viewpoint, distance_to_view,
     direction_to_zenith_n, direction_to_sun_n) = compute_altitude_and_elevation(
        shifted_viewpoint, ground_level_solar_elevation_at_origin, ground_level_solar_azimuth_at_origin)

    # Altitude-corrected view direction
    if distance_to_view > PSM_PLANET_RADIUS:
        look_at = view_direction_n + shifted_viewpoint

        correction = np.sqrt(distance_to_view * distance_to_view - PSM_PLANET_RADIUS_SQR) / distance_to_view
        to_new_origin = (distance_to_view - correction) * direction_to_zenith_n
        new_origin = to_new_origin + centre_of_earth

        correct_view = look_at - new_origin
        correct_view_n = correct_view / np.linalg.norm(correct_view)
    else:
        correct_view_n = view_direction_n

    # Sun angle (gamma) - no correction
    dot_product_sun = np.dot(view_direction_n, direction_to_sun_n)
    gamma = np.arccos(dot_product_sun)

    # Shadow angle - requires correction
    effective_elevation = ground_level_solar_elevation_at_origin
    effective_azimuth = ground_level_solar_azimuth_at_origin
    shadow_angle = effective_elevation + np.pi * 0.5

    shadow_direction_n = np.array([
        np.cos(shadow_angle) * np.cos(effective_azimuth),
        np.con(shadow_angle) * np.sin(effective_azimuth),
        np.sin(shadow_angle)
    ])
    dot_product_shadow = np.dot(correct_view_n, shadow_direction_n)
    shadow = np.arccos(dot_product_shadow)

    # Zenith angle (theta) - corrected version stored in otherwise unused zero angle
    cos_theta_cor = np.dot(correct_view_n, direction_to_zenith_n)
    zero = np.arccos(cos_theta_cor)

    # Zenith angle (theta) - uncorrected version goes outside
    cos_theta = np.dot(view_direction_n, direction_to_zenith_n)
    theta = np.arccos(cos_theta)

    return SolarAngles(solar_elevation_at_viewpoint, altitude_of_viewpoint, theta, gamma, shadow, zero)


def lerp(start, end, factor):
    return (1 - factor) * start + factor * end


def find_segment(x, breaks):
    segment = 0
    for segment in range(len(breaks)):
        if breaks[segment+1] > x:
            break
    return segment


def eval_pp(x, segment, breaks, coefs):
    x0 = x - breaks[segment]
    sc = coefs[2 * segment:]  # segment coefs
    return sc[0] * x0 + sc[1]


def control_params_single_config(state: SkyModelState, dataset, total_coef_single_config,
                                 elevation, altitude, turbidity, albedo, wavelength):
    return dataset[total_coef_single_config * (
            wavelength +
            state.channels * elevation +
            state.channels * len(state.elevations) + altitude +
            state.channels * len(state.elevations) * len(state.altitudes) * albedo +
            state.channels * len(state.elevations) * len(state.altitudes) * len(state.albedos) * turbidity)]


def reconstruct(state: SkyModelState, gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment, control_params):
    res: float = 0
    for t in range(state.tensor_components):
        sun_val_t = eval_pp(
            gamma, gamma_segment, state.sun_breaks, control_params[state.sun_offset + t * state.sun_stride]
        )
        zenith_val_t = eval_pp(
            alpha, alpha_segment, state.zenith_breaks, control_params[state.zenith_offset + t * state.zenith_stride]
        )
        res += sun_val_t * zenith_val_t
        emph_val_t = eval_pp(
            zero, zero_segment, state.emph_breaks, control_params[state.emph_offset]
        )
        res *= emph_val_t

    return np.maximum(res, 0)


def map_parameter(param, values):
    if param < values[0]:
        mapped: float = 0
    elif param > values[-1]:
        mapped: float = len(values) - 1
    else:
        for v, val in enumerate(values):
            if abs(val - param) < 1e-6:
                mapped: float = v
                break
            elif param < val:
                mapped: float = v - (val - param) / (val - values[v - 1])
                break
    return mapped


def interpolate_elevation(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
                          gamma_segment, alpha_segment, zero_segment):
    elevation_low = float(int(elevation))
    factor = elevation - elevation_low

    control_params_low = control_params_single_config(
        state, state.radiance_dataset, state.total_coefs_single_config,
        elevation_low, altitude, turbidity, albedo, wavelength
    )
    res_low = reconstruct(
        state, gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment, control_params_low
    )
    if factor < 01e-6 or elevation_low >= len(state.elevations) - 1:
        return res_low

    control_params_high = control_params_single_config(
        state, state.radiance_dataset, state.total_coefs_single_config,
        elevation_low+1, altitude, turbidity, albedo, wavelength
    )
    res_high = reconstruct(
        state, gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment, control_params_high
    )
    return lerp(res_low, res_high, factor)


def interpolate_altitude(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
                         gamma_segment, alpha_segment, zero_segment):
    altitude_low = float(int(altitude))
    factor = altitude - altitude_low

    res_low = interpolate_elevation(
        state, elevation, altitude_low, turbidity, albedo, wavelength, gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )
    if factor < 01e-6 or altitude_low >= (len(state.altitudes) - 1):
        return res_low
    res_high = interpolate_elevation(
        state, elevation, altitude_low + 1, turbidity, albedo, wavelength, gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )
    return lerp(res_low, res_high, factor)


def interpolate_turbidity(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
                          gamma_segment, alpha_segment, zero_segment):
    turbidity_low = float(int(turbidity))
    factor = turbidity - turbidity_low

    res_low = interpolate_altitude(
        state, elevation, altitude, turbidity_low, albedo, wavelength, gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )
    if factor < 01e-6 or turbidity_low >= (len(state.turbidities) - 1):
        return res_low
    res_high = interpolate_altitude(
        state, elevation, altitude, turbidity_low + 1, albedo, wavelength, gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )
    return lerp(res_low, res_high, factor)


def interpolate_albedo(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
                       gamma_segment, alpha_segment, zero_segment):
    albedo_low = float(int(albedo))
    factor = albedo - albedo_low

    res_low = interpolate_turbidity(
        state, elevation, altitude, turbidity, albedo_low, wavelength, gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )
    if factor < 01e-6 or albedo_low >= (len(state.albedos) - 1):
        return res_low
    res_high = interpolate_turbidity(
        state, elevation, altitude, turbidity, albedo_low + 1, wavelength, gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )
    return lerp(res_low, res_high, factor)


def interpolate_wavelengths(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength,
                            gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment):
    # Don't interpolate, use the bin it belongs to
    return interpolate_albedo(
        state, elevation, altitude, turbidity, albedo, float(int(wavelength)), gamma, alpha, zero,
        gamma_segment, alpha_segment, zero_segment
    )


def sky_model_radiance(state, theta, gamma, shadow, zero, elevation, altitude, turbidity, albedo, wavelength):
    """

    Parameters
    ----------
    state : SkyModelState
    theta : float
    gamma : float
    shadow : float
    zero : float
    elevation : float
    altitude : float
    turbidity : float
    albedo : float
    wavelength : float

    Returns
    -------
    float
    """

    # translate parameter values to indices
    turbidity_control = map_parameter(turbidity, state.turbidities)
    albedo_control = map_parameter(albedo, state.albedos)
    altitude_control = map_parameter(altitude, state.altitudes)
    elevation_control = map_parameter(elevation, state.elevations)

    channel_control = (wavelength - state.channel_start) / state.channel_width

    if channel_control >= state.channels or channel_control < 0:
        return 0.

    # Get params corresponding to the indices, reconstruct result and interpolate

    alpha = shadow if elevation < 0 else zero
    gamma_segment = find_segment(gamma, state.sun_breaks)
    alpha_segment = find_segment(alpha, state.zenith_breaks)
    zero_segment = find_segment(zero, state.emph_breaks)

    res = interpolate_wavelengths(
        state, elevation_control, altitude_control, turbidity_control, albedo_control, channel_control,
        gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment
    )

    return res


def sky_model_solar_radiance(state, theta, gamma, shadow, zero, elevation, altitude, turbidity, albedo, wavelength):
    """
    This computes transmittance between a point at 'altitude' and infinity in
    the direction 'theta' at a wavelength 'wavelength'.

    Parameters
    ----------
    state : SkyModelState
    theta : float
    gamma : float
    shadow : float
    zero : float
    elevation : float
    altitude : float
    turbidity : float
    albedo : float
    wavelength : float

    Returns
    -------
    float
    """
    idx = (wavelength - sun_rad_star_wl) / sun_rad_increment_wl
    sun_radiance = 0

    if idx >= 0:
        low_idx = int(idx)
        idx_float = idx - float(low_idx)

        sun_radiance = sun_rad_table[low_idx] * (1 - idx_float) + sun_rad_table[low_idx + 1] * idx_float

    tau = sky_model_tau(state, theta, altitude, turbidity, wavelength, np.infty)

    return sun_radiance * tau


def sky_model_circle_bounds_2d(x_v, y_v, y_c, radius, d):
    pass


def sky_model_scale_ad():
    pass


def sky_model_to_ad(theta, distance, altitude, a, d):
    """

    Parameters
    ----------
    theta : float
    distance : float
    altitude : float
    a : float
    d : float

    Returns
    -------
    float
    """
    return None


def sky_model_transmittance_coefs_index():
    pass


def clamp_0_1():
    pass


def sky_model_calc_transmittance_svd_altitude():
    pass


def non_lin_lerp():
    pass


def sky_model_calc_transittance_svd():
    pass


def cbrt():
    pass


def sky_model_find_in_array():
    pass


def sky_model_tau(state, theta, altitude, turbidity, wavelength, distance):
    """
    This computes transmittance between a point at 'altitude' and infinity in
    the direction 'theta' at a wavelength 'wavelength'.

    Parameters
    ----------
    state : SkyModelState
    theta : float
    altitude : float
    turbidity : float
    wavelength : float
    distance : float

    Returns
    -------
    float
    """
    return None


def sky_model_polarisation(state, theta, gamma, shadow, zero, elevation, altitude, turbidity, albedo, wavelength):
    """

    Parameters
    ----------
    state : SkyModelState
    theta : float
    gamma : float
    shadow : float
    zero : float
    elevation : float
    altitude : float
    turbidity : float
    albedo : float
    wavelength : float

    Returns
    -------
    float
    """
    return None


def compute_pp_coefs(breaks, values, coefs, offset):
    """

    Parameters
    ----------
    breaks : list[float]
    values : list[float]
    coefs : list[float]
    offset : int

    Returns
    -------
    int
    """
    nb_breaks = len(breaks)
    for i in range(nb_breaks):
        coefs[offset + 2 * i] = (values[i+1] - values[i]) / (breaks[i+1] - breaks[i])
        coefs[offset + 2 * i + 1] = values[i]

    return 2 * nb_breaks - 2


