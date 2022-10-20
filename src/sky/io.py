from sky.__helpers import print_error_and_exit
from sky.prague_sky_model import SkyModelState

import numpy as np
import struct


def load_sky_model(filename):

    state = SkyModelState(
        # Radiance metadata
        turbidities=None, albedos=None, altitudes=None, elevations=None,
        channels=None, channel_start=None, channel_width=None,
        tensor_components=None,
        sun_breaks=None, sun_offset=None, sun_stride=None,
        zenith_breaks=None, zenith_offset=None, zenith_stride=None,
        emph_breaks=None, emph_offset=None,
        total_coefs_single_config=None, total_coefs_all_configs=None, total_configs=None,

        # Radiance data
        radiance_dataset=None,

        # Transmittance metadata
        trans_n_d=None, trans_n_a=None, trans_turbidities=None, trans_altitudes=None, trans_rank=None,
        transmission_altitudes=None, transmission_turbities=None,

        # Transmittance data
        transmission_dataset_U=None, transmission_dataset_V=None,

        # Polarisation metadata
        tensor_components_pol=None,
        sun_breaks_pol=None, sun_offset_pol=None, sun_stride_pol=None,
        zenith_breaks_pol=None, zenith_offset_pol=None, zenith_stride_pol=None,
        total_coefs_single_config_pol=None, total_coefs_all_configs_pol=None,

        # Polarisation data
        polarisation_dataset=None
    )

    with open(filename, "rb") as f:
        state = read_radiance(f, state)
        state = read_transmittance(f, state)
        try:
            state = read_polarisation(f, state)
        except Exception as e:
            print("No polarisation dataset available!")
            print(e)

    return state


def read_radiance(handle, state: SkyModelState):
    if isinstance(handle, str):
        handle = open(handle, "rb")

    # Read metadata

    # Structure of the metadata part of the data file:
    # turbidities(1 * int), turbidity_vals(turbidities * double),
    # albedos(1 * int), albedo_vals(albedos * double),
    # altitudes(1 * int), altitude_vals(altitudes * double),
    # elevations(1 * int), elevation_vals(elevations * double),
    # channels(1 * int), channel_start(1 * double), channel_width(1 * double),
    # tensor_components(1 * int),
    # sun_nbreaks(1 * int), sun_breaks(sun_nbreaks * double),

    # zenith_nbreaks(1 * int), zenith_breaks(zenith_nbreaks * double),
    # emph_nbreaks(1 * int), emph_breaks(emph_nbreaks * double)

    turbidities = read_double_list(handle)
    print(f"Turbidity: {len(turbidities)}")

    albedos = read_double_list(handle)
    print(f"Albedos: {len(albedos)}")

    altitudes = read_double_list(handle)
    print(f"Altitutes: {len(altitudes)}")

    elevations = read_double_list(handle)
    print(f"Elevations: {len(elevations)}")

    nb_channels = read_int(handle)
    channel_start = read_double(handle)
    channel_width = read_double(handle)
    tensor_components = read_int(handle)
    print(f"Number of channels: {nb_channels}, start: {channel_start:.2f}, width: {channel_width:.2f}")
    print(f"Tensor components: {tensor_components}")

    sun_breaks = read_double_list(handle)
    zenith_breaks = read_double_list(handle)
    emph_breaks = read_double_list(handle)

    # Calculate offsets and strides

    sun_offset = 0
    sun_stride = 2 * len(sun_breaks) - 2 + 2 * len(zenith_breaks) - 2
    print(f"Sun breaks: {len(sun_breaks)}, offset: {sun_offset}, stride: {sun_stride}")

    zenith_offset = sun_offset + 2 * len(sun_breaks) - 2
    zenith_stride = sun_stride
    print(f"Zenith breaks: {len(zenith_breaks)}, offset: {zenith_offset}, stride: {zenith_stride}")

    emph_offset = sun_offset + tensor_components * sun_stride
    print(f"Ephimeris breaks: {len(emph_breaks)}, offset: {emph_offset}")

    total_coefs_single_config = emph_offset + 2 * len(emph_breaks) - 2
    # this is for one specific configuration
    total_configs = (nb_channels * len(elevations) * len(altitudes) *
                           len(albedos) * len(turbidities))
    total_coefs_all_configs = total_coefs_single_config * total_configs
    print(f"Configurations: total configs = {total_configs}, "
          f"coefficients (single config) = {total_coefs_single_config}, "
          f"coefficients (all configs) = {total_coefs_all_configs}")

    # Read data
    # Structure of the data part of the data file:
    # [[[[[[
    #       sun_coefs(sun_nbreaks * half), zenith_scale(1 * double), zenith_coefs(zenith_nbreaks * half)
    #      ] * tensor_components, emph_coefs(emph_nbreaks * half)
    #     ] * channels
    #    ] * elevations
    #   ] * altitudes
    #  ] * albedos
    # ] * turbidities

    offset: int = 0
    radiance_dataset = np.zeros(total_coefs_all_configs, dtype='float64')

    for con in range(total_configs):
        for tc in range(tensor_components):
            sun_scale: float = 1
            radiance_temp = read_ushort_list(handle, len(sun_breaks))
            if len(radiance_temp) != len(sun_breaks):
                print_error_and_exit("Error reading sky model data: sun_coefs.")
            offset += compute_pp_coef(sun_breaks, radiance_temp, radiance_dataset, offset, sun_scale)

            zenith_scale: float = read_double(handle)
            radiance_temp = read_ushort_list(handle, len(zenith_breaks))
            offset += compute_pp_coef(zenith_breaks, radiance_temp, radiance_dataset, offset, zenith_scale)

        emph_scale: float = 1
        radiance_temp = read_ushort_list(handle, len(emph_breaks))
        offset += compute_pp_coef(emph_breaks, radiance_temp, radiance_dataset, offset, emph_scale)

    print(f"Radiance: min = {radiance_dataset.min():.1f}, max = {radiance_dataset.max():.1f}")

    return SkyModelState(
        # Radiance metadata
        turbidities=turbidities, albedos=albedos, altitudes=altitudes, elevations=elevations,
        channels=nb_channels, channel_start=channel_start, channel_width=channel_width,
        tensor_components=tensor_components,
        sun_breaks=sun_breaks, sun_offset=sun_offset, sun_stride=sun_stride,
        zenith_breaks=zenith_breaks, zenith_offset=zenith_offset, zenith_stride=zenith_stride,
        emph_breaks=emph_breaks, emph_offset=emph_offset,
        total_coefs_single_config=total_coefs_single_config,
        total_coefs_all_configs=total_coefs_all_configs, total_configs=total_configs,

        # Radiance data
        radiance_dataset=radiance_dataset,

        # Transmittance metadata
        trans_n_d=state.trans_n_d, trans_n_a=state.trans_n_a, trans_turbidities=state.trans_turbidities,
        trans_altitudes=state.trans_altitudes, trans_rank=state.trans_rank,
        transmission_altitudes=state.transmission_altitudes, transmission_turbities=state.trans_turbidities,

        # Transmittance data
        transmission_dataset_U=state.transmission_dataset_U, transmission_dataset_V=state.transmission_dataset_V,

        # Polarisation metadata
        tensor_components_pol=state.tensor_components_pol,
        sun_breaks_pol=state.sun_breaks_pol, sun_offset_pol=state.sun_offset_pol, sun_stride_pol=state.sun_stride_pol,
        zenith_breaks_pol=state.zenith_breaks_pol, zenith_offset_pol=state.zenith_offset_pol,
        zenith_stride_pol=state.zenith_stride_pol,
        total_coefs_single_config_pol=state.total_coefs_single_config_pol,
        total_coefs_all_configs_pol=state.total_coefs_all_configs_pol,

        # Polarisation data
        polarisation_dataset=state.polarisation_dataset
    )


def read_transmittance(handle, state: SkyModelState):
    if isinstance(handle, str):
        handle = open(handle, "rb")

    # Read metadata

    trans_n_d = read_int(handle)
    print(f"Transmittance n d: {trans_n_d}")

    trans_n_a = read_int(handle)
    print(f"Transmittance n a: {trans_n_a}")

    trans_turbidities = read_int(handle)
    print(f"Transmittance turbidities: {trans_turbidities}")

    trans_altitudes = read_int(handle)
    print(f"Transmittance altitudes: {trans_altitudes}")

    trans_rank = read_int(handle)
    print(f"Transmittance rank: {trans_rank}")

    transmission_altitudes = read_float_list(handle, size=trans_altitudes)
    print(f"Transmission altitudes: {len(transmission_altitudes)}")

    transmission_turbidities = read_float_list(handle, size=trans_turbidities)
    print(f"Transmission turbidities: {len(transmission_turbidities)}")

    total_coefs_U = trans_n_d * trans_n_a * trans_rank * trans_altitudes
    total_coefs_V = trans_turbidities * trans_rank * 11 * trans_altitudes

    # Read data

    transmission_dataset_U = read_float_list(handle, size=total_coefs_U)
    transmission_dataset_V = read_float_list(handle, size=total_coefs_V)

    return SkyModelState(
        # Radiance metadata
        turbidities=state.turbidities, albedos=state.albedos, altitudes=state.altitudes, elevations=state.elevations,
        channels=state.channels, channel_start=state.channel_start, channel_width=state.channel_width,
        tensor_components=state.tensor_components,
        sun_breaks=state.sun_breaks, sun_offset=state.sun_offset, sun_stride=state.sun_stride,
        zenith_breaks=state.zenith_breaks, zenith_offset=state.zenith_offset, zenith_stride=state.zenith_stride,
        emph_breaks=state.emph_breaks, emph_offset=state.emph_offset,
        total_coefs_single_config=state.total_coefs_single_config,
        total_coefs_all_configs=state.total_coefs_all_configs, total_configs=state.total_configs,

        # Radiance data
        radiance_dataset=state.radiance_dataset,

        # Transmittance metadata
        trans_n_d=trans_n_d, trans_n_a=trans_n_a, trans_turbidities=trans_turbidities,
        trans_altitudes=trans_altitudes, trans_rank=trans_rank,
        transmission_altitudes=transmission_altitudes, transmission_turbities=trans_turbidities,

        # Transmittance data
        transmission_dataset_U=transmission_dataset_U, transmission_dataset_V=transmission_dataset_V,

        # Polarisation metadata
        tensor_components_pol=state.tensor_components_pol,
        sun_breaks_pol=state.sun_breaks_pol, sun_offset_pol=state.sun_offset_pol, sun_stride_pol=state.sun_stride_pol,
        zenith_breaks_pol=state.zenith_breaks_pol, zenith_offset_pol=state.zenith_offset_pol,
        zenith_stride_pol=state.zenith_stride_pol,
        total_coefs_single_config_pol=state.total_coefs_single_config_pol,
        total_coefs_all_configs_pol=state.total_coefs_all_configs_pol,

        # Polarisation data
        polarisation_dataset=state.polarisation_dataset
    )


def read_polarisation(handle, state: SkyModelState):
    if isinstance(handle, str):
        handle = open(handle, "rb")

    # Read metadata
    # Structure of the metadata part of the data file:
    # tensor_components_pol(1 * int),
    # sun_nbreaks_pol(1 * int), sun_breaks_pol(sun_nbreaks_pol * double),
    # zenith_nbreaks_pol(1 * int), zenith_breaks_pol(zenith_nbreaks_pol * double),
    # emph_nbreaks_pol(1 * int), emph_breaks_pol(emph_nbreaks_pol * double)

    tensor_components_pol = read_int(handle)

    sun_breaks_pol = read_double_list(handle)
    zenith_breaks_pol = read_double_list(handle)

    # Calculate offsets and strides
    sun_offset_pol = 0
    sun_stride_pol = 2 * len(sun_breaks_pol) - 2 + 2 * len(zenith_breaks_pol) - 2
    print(f"Sun breaks (POL): {len(sun_breaks_pol)}, "
          f"offset (POL): {sun_offset_pol}, "
          f"stride (POL): {sun_stride_pol}")

    zenith_offset_pol = sun_offset_pol + 2 * len(sun_breaks_pol) - 2
    zenith_stride_pol = sun_stride_pol
    print(f"Zenith breaks (POL): {len(zenith_breaks_pol)},"
          f" offset (POL): {zenith_offset_pol}, "
          f"stride (POL): {zenith_stride_pol}")

    total_coefs_single_config_pol = sun_offset_pol + tensor_components_pol * sun_stride_pol
    total_coefs_all_configs_pol = total_coefs_single_config_pol * state.total_configs

    # Read data
    # Structure of the data part of the data file:
    # [[[[[[
    #       sun_coefs_pol(sun_nbreaks_pol * float), zenith_coefs_pol(zenith_nbreaks_pol * float)
    #      ] * tensor_components_pol
    #     ] * channels
    #    ] * elevations
    #   ] * altitudes
    #  ] * albedos
    # ] * turbidities

    offset = 0
    polarisation_dataset = np.zeros(total_coefs_all_configs_pol, dtype='float64')
    for con in range(state.total_configs):
        for tc in range(tensor_components_pol):
            polarisation_temp: np.ndarray = read_float_list(handle, size=len(sun_breaks_pol))
            offset += compute_pp_coef(sun_breaks_pol, polarisation_temp, polarisation_dataset, offset)

            polarisation_temp: np.ndarray = read_double_list(handle, size=len(zenith_breaks_pol))
            offset += compute_pp_coef(zenith_breaks_pol, polarisation_temp, polarisation_dataset, offset)

    return SkyModelState(
        # Radiance metadata
        turbidities=state.turbidities, albedos=state.albedos, altitudes=state.altitudes, elevations=state.elevations,
        channels=state.channels, channel_start=state.channel_start, channel_width=state.channel_width,
        tensor_components=state.tensor_components,
        sun_breaks=state.sun_breaks, sun_offset=state.sun_offset, sun_stride=state.sun_stride,
        zenith_breaks=state.zenith_breaks, zenith_offset=state.zenith_offset, zenith_stride=state.zenith_stride,
        emph_breaks=state.emph_breaks, emph_offset=state.emph_offset,
        total_coefs_single_config=state.total_coefs_single_config,
        total_coefs_all_configs=state.total_coefs_all_configs, total_configs=state.total_configs,

        # Radiance data
        radiance_dataset=state.radiance_dataset,

        # Transmittance metadata
        trans_n_d=state.trans_n_d, trans_n_a=state.trans_n_a, trans_turbidities=state.trans_turbidities,
        trans_altitudes=state.trans_altitudes, trans_rank=state.trans_rank,
        transmission_altitudes=state.transmission_altitudes, transmission_turbities=state.trans_turbidities,

        # Transmittance data
        transmission_dataset_U=state.transmission_dataset_U, transmission_dataset_V=state.transmission_dataset_V,

        # Polarisation metadata
        tensor_components_pol=tensor_components_pol,
        sun_breaks_pol=sun_breaks_pol, sun_offset_pol=sun_offset_pol, sun_stride_pol=sun_stride_pol,
        zenith_breaks_pol=zenith_breaks_pol, zenith_offset_pol=zenith_offset_pol, zenith_stride_pol=zenith_stride_pol,
        total_coefs_single_config_pol=total_coefs_single_config_pol,
        total_coefs_all_configs_pol=total_coefs_all_configs_pol,

        # Polarisation data
        polarisation_dataset=polarisation_dataset
    )


def read_ushort_list(handle, size=None):
    if size is None:
        size = read_int(handle)
    return np.array(list(struct.unpack('H' * size, handle.read(2 * size))), dtype='uint16')


def read_float_list(handle, size=None):
    if size is None:
        size = read_float(handle)
    return np.array(list(struct.unpack('f' * size, handle.read(4 * size))), dtype='float32')


def read_double_list(handle, size=None):
    if size is None:
        size = read_int(handle)
    return np.array(list(struct.unpack('d' * size, handle.read(8 * size))), dtype='float64')


def read_ushort(handle):
    val, = struct.unpack('H', handle.read(2))
    return val


def read_int(handle):
    val, = struct.unpack('i', handle.read(4))
    return val


def read_float(handle):
    val, = struct.unpack('f', handle.read(4))
    return val


def read_double(handle):
    val, = struct.unpack('d', handle.read(8))
    return val


def compute_pp_coef(breaks: np.ndarray, values: np.ndarray, coefs: np.ndarray, offset: int, scale: float = 1):
    for i in range(len(breaks) - 1):
        val1 = float(values[i+1]) / scale
        val2 = float(values[i]) / scale
        coefs[offset + 2 * i] = (val1 - val2) / (breaks[i+1] - breaks[i])
        coefs[offset + 2 * i + 1] = val2
    return 2 * len(breaks) - 2


if __name__ == "__main__":
    file_name = (r"C:\Users\Odin\OneDrive - University of Edinburgh\Projects\2022-InsectNeuroNano"
                 r"\Hosek and Wilkie sky model\SkyModelDataset-001.dat", "rb")
    s = load_sky_model(file_name)

    print(s)
