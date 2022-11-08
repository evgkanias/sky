import struct

from .io import *
from .skytypes import *
from .exceptions import *

from bisect import bisect_right
from numba import vectorize, njit, prange

import numpy as np
import warnings


class PragueSkyModel(object):
    def __init__(self):
        self.__nb_channels: int = 0
        self.__channel_start: float = 0
        self.__channel_width: float = 0

        self.__initialised: bool = False
        self.__total_configs: int = 0
        """
        Total number of configurations
        """

        self.__skipped_configs_begin: int = 0
        """
        Number of configurations skipped from the beginning of the radiance/polarisation coefficients part of
        the dataset file (if loading of just one visibility was requested)
        """

        self.__skipped_configs_end: int = 0
        """
        Number of configurations skipped till the end of the radiance/polarisation coefficients part of
         the dataset file (if loading of just one visibility was requested)
        """

        # Metadata common for radiance and polarisation
        self.__visibilities_rad: {np.ndarray, None} = None
        self.__albedos_rad: {np.ndarray, None} = None
        self.__altitudes_rad: {np.ndarray, None} = None
        self.__elevations_rad: {np.ndarray, None} = None

        # Radiance metadata
        self.__metadata_rad: {Metadata, None} = None

        # Radiance data
        #
        # Structure:
        # [[[[[[
        #       sun_coefs_rad (sun_breaks_count_rad * float),
        #       zenith_coefs_rad (zenith_breas_count_rad * float)
        #      ] * rank_rad,
        #      emph_coefs_rad (emph_breaks_count_rad * float)
        #     ] * channels
        #    ] * elevation_count
        #   ] * altitude_count
        #  ] * albedo_count
        # ] * visibility_count
        self.__data_rad: {np.ndarray, None} = None

        # Polarisation metadata
        self.__metadata_pol: {Metadata, None} = None

        # Polarisation data
        #
        # Structure:
        # [[[[[[
        #       sun_coefs_rad (sun_breaks_count_pol * float),
        #       zenith_coefs_rad (zenith_breas_count_pol * float)
        #      ] * rank_pol,
        #     ] * channels
        #    ] * elevation_count
        #   ] * altitude_count
        #  ] * albedo_count
        # ] * visibility_count
        self.__data_pol: {np.ndarray, None} = None

        # Transmittance metadata
        self.__a_dim: int = 0
        self.__d_dim: int = 0
        self.__rank_trans: int = 0
        self.__altitudes_trans: {np.ndarray, None} = None
        self.__visibilities_trans: {np.ndarray, None} = None

        # Transmittance data
        self.__data_trans_u: {np.ndarray, None} = None
        self.__data_trans_v: {np.ndarray, None} = None

    def reset(self, filename, single_visibility=0.0):
        """
        Prepares the model and loads the given dataset file into memory.

        If a positive visibility value is passed, only a portion of the dataset needed for evaluation of that
        particular visibility is loaded (two nearest visibilities are loaded if the value is included in the
        dataset or one nearest if not). Otherwise, the entire dataset is loaded.

        Throws:
        - DatasetNotFoundException: if the specified dataset file could not be found
        - DatasetReadException: if an error occurred while reading the dataset file

        Parameters
        ----------
        filename : str
        single_visibility : float
        """
        with open(filename, "rb") as handle:
            self.__read_metadata(handle, single_visibility)
            self.__read_radiance(handle)
            self.__read_transmittance(handle)
            try:
                self.__read_polarisation(handle)
            except struct.error as e:
                # raise NoPolarisationWarning() from e
                warnings.warn(f"The supplied dataset does not contain polarisation data.\n{e}")
        self.__initialised = True

    def sky_radiance(self, params, wavelength):
        """
        Computes sky radiance only (without direct sun contribution) for given parameters and wavelength (full
        dataset supports wavelengths from 280 nm to 2480 nm).

        Throws NotInitializedException if called without initializing the model first.

        Parameters
        ----------
        params : Parameters
        wavelength : np.ndarray[float]

        Returns
        -------
        float, np.ndarray[float]
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__evaluate_model(params, wavelength, self.data_rad, self.metadata_rad)

    def sun_radiance(self, params, wavelength):
        """
        Computes sun radiance only (without radiance inscattered from the sky) for given parameters and
        wavelength (full dataset supports wavelengths from 280 nm to 2480 nm).

        Checks whether the parameters correspond to view direction hitting the sun and returns 0 if not.

        Throws NotInitializedException if called without initializing the model first.

        Parameters
        ----------
        params : Parameters
        wavelength : np.ndarray[float]

        Returns
        -------
        np.ndarray[float]
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        sun_radiance = np.zeros((len(wavelength), len(params.gamma)), dtype='float64')
        print(f"sun_radiance: {sun_radiance.shape}")
        print(f"bool map: {(sun_radiance > 0).shape}")

        # Ignore wavelengths outside the dataset range.
        wl_in_range = np.all([wavelength >= SUN_RAD_START, wavelength < SUN_RAD_END], axis=0)
        # if wavelength < SUN_RAD_START or wavelength >= SUN_RAD_END:
        #     return 0
        print(f"wl_in_range: {wl_in_range.shape}, # = {wl_in_range.sum()}")

        # Return zero for rays not hitting the sun.
        ray_in_radius = params.gamma <= SUN_RADIUS
        # if params.gamma > SUN_RADIUS:
        #     return 0
        print(f"ray_in_radius: {ray_in_radius.shape}, # = {ray_in_radius.sum()}")

        valid = wl_in_range[:, None] & ray_in_radius[None, :]
        print(f"valid: {valid.shape}")

        # Compute index into the sun radiance table.
        idx = (wavelength - SUN_RAD_START) / SUN_RAD_STEP
        assert np.all([0 <= idx[wl_in_range], idx[wl_in_range] < len(SUN_RAD_TABLE) - 1])

        idx = np.repeat(idx[:, None], len(ray_in_radius), axis=1)
        print(f"idx: {idx.shape}")

        idx_int = np.int32(np.floor(idx[valid]))
        idx_float = idx[valid] - np.floor(idx[valid])

        # interpolate between the two closest values in the sun radiance table.
        sun_radiance[valid] = SUN_RAD_TABLE[idx_int] * (1 - idx_float) + SUN_RAD_TABLE[idx_int + 1] * idx_float
        assert np.all(sun_radiance[valid] > 0)

        # Compute transmittance towards the sun.
        tau = self.transmittance(params, wavelength[wl_in_range], np.finfo(float).max)
        assert np.all([0 <= tau, tau <= 1])

        # Combine
        return sun_radiance * tau

    def polarisation(self, params, wavelength):
        """
        Computes degree of polarisation for given parameters and wavelength (full
        dataset supports wavelengths from 280 nm to 2480 nm). Can be negative.

        Throws:
        - NoPolarisationException: if the polarisation method is called but the model does not contain
        polarisation data
        - NotInitializedException: if called without initializing the model first

        Parameters
        ----------
        params : Parameters
        wavelength : np.ndarray[float]

        Returns
        -------
        float
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        # If no polarisation data available
        if self.metadata_pol is None or self.metadata_pol.rank == 0:
            raise NoPolarisationWarning()

        return -self.__evaluate_model(params, wavelength, self.data_pol, self.metadata_pol)

    def transmittance(self, params, wavelength, distance):
        """
        Computes transmittance between view point and a point certain distance away from it along view
        direction.

        Expects the Parameters structure, wavelength (full dataset supports wavelengths from 280 nm
        to 2480 nm) and the distance (any positive number, use std::numeric_limits<double>::max() for
        infinity).

        Throws NotInitializedException if called without initializing the model first.

        Parameters
        ----------
        params : Parameters
        wavelength : np.ndarray[float]
        distance : float

        Returns
        -------
        float
        """
        assert distance > 0

        if not self.is_initialised:
            raise NotInitialisedException()

        # Ignore wavelengths outside the dataset range.
        wl_in_range = np.all([
            wavelength >= self.__channel_start,
            wavelength < self.__channel_start + self.__nb_channels * self.__channel_width
        ], axis=0)

        # Don't interpolate wavelengths inside the dataset range.
        channel_index = np.int32(np.floor((wavelength - self.__channel_start) / self.__channel_width))[wl_in_range]

        # Translate configuration values to indices and interpolation factors.
        visibility_param = self.get_interpolation_parameter(params.visibility, self.__visibilities_trans)
        altitude_param = self.get_interpolation_parameter(params.altitude, self.__altitudes_trans)

        # Calculate position in the atmosphere
        trans_params = self.__to_transmittance_params(params.theta, distance, params.altitude)

        # Get transmittance for the nearest lower visibility.
        trans = self.__interpolate_trans(visibility_param.index, altitude_param, trans_params, channel_index)

        # Interpolate with transmittance for the nearest higher visibility if needed.
        if visibility_param.factor > 0:
            trans_high = self.__interpolate_trans(
                visibility_param.index + 1, altitude_param, trans_params, channel_index)

            trans = lerp(trans, trans_high, visibility_param.factor)

        # Transmittance is stored as a square root. Needs to be restored here.
        trans = trans * trans

        assert np.all([0 <= trans, trans <= 1])

        return trans

    @staticmethod
    def compute_parameters(viewpoint, view_direction, ground_level_solar_elevation_at_origin,
                           ground_level_solar_azimuth_at_origin, visibility, albedo):
        """
        Computes all the parameters in the Parameters structure necessary for querying the model.

        Expects view point and direction, sun elevation and azimuth at origin, ground level visibility and
        ground albedo. Assumes origin at [0,0,0] with Z axis pointing up. Thus view point [0, 0, 100] defines
        observer altitude 100 m. Range of available values depends on the used dataset. The full version
        supports altitude from [0, 15000] m, elevation from [-0.073, PI/2] rad, azimuth from [0, PI] rad, visibility
        from [20, 131.8] km, and albedo from [0, 1]. Values outside range of the used dataset are clamped to the
        nearest supported value.

        Parameters
        ----------
        viewpoint : np.ndarray
        view_direction : np.ndarray
        ground_level_solar_elevation_at_origin : float
            in rad
        ground_level_solar_azimuth_at_origin : float
            in rad
        visibility : float
        albedo : float

        Returns
        -------
        Parameters
        """

        assert viewpoint[2] >= 0
        assert np.all(np.linalg.norm(view_direction, axis=-1) > 0)
        assert visibility >= 0
        assert 0 <= albedo <= 1

        # Shift viewpoint about safety altitude up
        centre_of_earth = np.array([0, 0, -PLANET_RADIUS], dtype='float64')
        to_viewpoint = viewpoint - centre_of_earth
        to_viewpoint_n = to_viewpoint / np.linalg.norm(to_viewpoint)

        distance_to_view = np.linalg.norm(to_viewpoint) + SAFETY_ALTITUDE
        to_shifted_viewpoint = to_viewpoint_n * distance_to_view
        shifted_viewpoint = centre_of_earth + to_shifted_viewpoint

        view_direction_n = (
                view_direction.T / np.maximum(np.linalg.norm(view_direction, axis=-1), np.finfo(float).eps)).T

        # Compute altitude of viewpoint
        altitude = np.maximum(distance_to_view - PLANET_RADIUS, 0)

        # Direction to sun
        direction_to_sun_n = np.array([
            np.cos(ground_level_solar_azimuth_at_origin) * np.cos(ground_level_solar_elevation_at_origin),
            np.sin(ground_level_solar_azimuth_at_origin) * np.cos(ground_level_solar_elevation_at_origin),
            np.sin(ground_level_solar_elevation_at_origin)
        ], dtype='float64')

        # Solar elevation at viewpoint
        # (more precisely, solar elevation at the point on the ground directly below viewpoint)
        dot_zenith_sun = np.dot(to_viewpoint_n, direction_to_sun_n)
        elevation = 0.5 * np.pi - np.arccos(dot_zenith_sun)

        # Altitude-corrected view direction
        if distance_to_view > PLANET_RADIUS:
            look_at = view_direction_n + shifted_viewpoint

            correction = np.sqrt(distance_to_view * distance_to_view - PLANET_RADIUS * PLANET_RADIUS) / distance_to_view

            to_new_origin = to_viewpoint_n * (distance_to_view - correction)
            new_origin = centre_of_earth + to_new_origin
            correct_view = look_at - new_origin

            correct_view_n = (correct_view.T / np.linalg.norm(correct_view, axis=-1)).T
        else:
            correct_view_n = view_direction_n

        # Sun angle (gamma) - no correction
        dot_product_sun = np.dot(view_direction_n, direction_to_sun_n)
        gamma = np.arccos(dot_product_sun)  # rad

        # Shadow angle - requires correction
        effective_elevation = ground_level_solar_elevation_at_origin  # rad
        effective_azimuth = ground_level_solar_azimuth_at_origin  # rad
        shadow_angle = effective_elevation + np.pi * 0.5  # rad

        shadow_direction_n = np.array([
            np.cos(shadow_angle) * np.cos(effective_azimuth),
            np.cos(shadow_angle) * np.sin(effective_azimuth),
            np.sin(shadow_angle)
        ], dtype='float64')
        dot_product_shadow = np.dot(correct_view_n, shadow_direction_n)
        shadow = np.arccos(dot_product_shadow)  # rad

        # Zenith angle (theta) - corrected version stored in otherwise unused zero angle
        cos_theta_cor = np.dot(correct_view_n, to_viewpoint_n)
        zero = np.arccos(cos_theta_cor)  # rad

        # Zenith angle (theta) - uncorrected version goes outside
        cos_theta = np.dot(view_direction_n, to_viewpoint_n)
        theta = np.arccos(cos_theta)  # rad

        return Parameters(
            theta=theta, gamma=gamma, shadow=shadow, zero=zero, elevation=elevation, altitude=altitude,
            visibility=visibility, albedo=albedo
        )

    @property
    def available_data(self):
        """
        Gets parameter ranges available in currently loaded dataset.

        Throws NotInitializedException if called without initializing the model first.

        Returns
        -------
        AvailableData
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return AvailableData(
            albedo_min=self.albedos_rad[0],
            albedo_max=self.albedos_rad[-1],
            altitude_min=self.altitudes_rad[0],
            altitude_max=self.altitudes_rad[-1],
            elevation_min=self.elevations_rad[0],
            elevation_max=self.elevations_rad[-1],
            visibility_min=self.visibilities_rad[0],
            visibility_max=self.visibilities_rad[-1],
            polarisation=self.metadata_pol.rank > 0,
            channels=self.nb_channels,
            channel_start=self.channel_start,
            channel_width=self.channel_width
        )

    @property
    def is_initialised(self):
        return self.__initialised

    @property
    def nb_channels(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__nb_channels

    @property
    def channel_start(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__channel_start

    @property
    def channel_width(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__channel_width

    @property
    def total_configs(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__total_configs

    @property
    def visibilities_rad(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__visibilities_rad

    @property
    def albedos_rad(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__albedos_rad

    @property
    def altitudes_rad(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__altitudes_rad

    @property
    def elevations_rad(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__elevations_rad

    @property
    def metadata_rad(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__metadata_rad

    @property
    def data_rad(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_rad

    @property
    def d_dim(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__d_dim

    @property
    def a_dim(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__a_dim

    @property
    def rank_trans(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__rank_trans

    @property
    def altitudes_trans(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__altitudes_trans

    @property
    def visibilities_trans(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__visibilities_trans

    @property
    def data_trans_u(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_trans_u

    @property
    def data_trans_v(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_trans_v

    @property
    def metadata_pol(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__metadata_pol

    @property
    def data_pol(self):
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_pol

    def __read_metadata(self, handle, single_visibility):
        # Read metadata

        # Structure of the metadata part of the data file:
        # visibilities   (1 * int), visibility_vals  (turbidities * double),
        # albedos        (1 * int), albedo_vals          (albedos * double),
        # altitudes      (1 * int), altitude_vals      (altitudes * double),
        # elevations     (1 * int), elevation_vals    (elevations * double),
        # channels       (1 * int), channel_start              (1 * double), channel_width(1 * double),

        visibilities_rad = read_double_list(handle)

        skipped_visibilities = 0
        if single_visibility <= 0 or len(visibilities_rad) <= 1:
            # If the given single visibility value is not valid or there is just one visibility present (so there
            # is nothing to save), all visibilities present are loaded.
            self.__visibilities_rad = visibilities_rad
        else:
            if single_visibility <= visibilities_rad[0]:
                self.__visibilities_rad = np.array(visibilities_rad[0])
            elif single_visibility >= visibilities_rad[-1]:
                self.__visibilities_rad = np.array(visibilities_rad[-1])
                skipped_visibilities = len(visibilities_rad) - 1
            else:
                vis_idx = 0
                while single_visibility >= visibilities_rad[vis_idx]:
                    vis_idx += 1
                self.__visibilities_rad = np.array([
                    visibilities_rad[vis_idx - 1],
                    visibilities_rad[vis_idx]
                ])
                skipped_visibilities = vis_idx - 1

        self.__albedos_rad = read_double_list(handle)

        self.__altitudes_rad = read_double_list(handle)

        self.__elevations_rad = read_double_list(handle)

        self.__nb_channels = read_int(handle)
        self.__channel_start = read_double(handle)
        self.__channel_width = read_double(handle)

        self.__total_configs = (
                self.__nb_channels * len(self.__elevations_rad) * len(self.__altitudes_rad) *
                len(self.__albedos_rad) * len(self.__visibilities_rad)
        )
        self.__skipped_configs_begin = (
                self.__nb_channels * len(self.__elevations_rad) * len(self.__altitudes_rad) *
                len(self.__albedos_rad) * skipped_visibilities
        )
        self.__skipped_configs_end = (
                self.__nb_channels * len(self.__elevations_rad) * len(self.__altitudes_rad) *
                len(self.__albedos_rad) * (len(visibilities_rad) - skipped_visibilities - len(self.__visibilities_rad))
        )

    def __read_radiance(self, handle):
        """
        Reads radiance part of the dataset file into memory.

        Structure:
        rank_rad                 (1 * int),
        sun_breaks_rad_count     (1 * int), sun_breaks_rad     (sun_breaks_rad_count * double)
        zenith_breaks_rad_count  (1 * int), zenith_breaks_rad  (zenith_breaks_rad_count * double)
        emph_breaks_rad_count    (1 * int), emph_breaks_rad    (emph_breaks_rad_count * double)

        If a positive visibility value is passed, only a portion of the dataset needed for evaluation of that
        particular visibility is loaded (two nearest visibilities are loaded if the value is included in the
        dataset or one nearest if not). Otherwise, the entire dataset is loaded.

        Throws DatasetReadException if an error occurred while reading the dataset file.

        Parameters
        ----------
        handle : file
        """

        rank = read_int(handle)

        sun_breaks = read_double_list(handle)
        zenith_breaks = read_double_list(handle)
        emph_breaks = read_double_list(handle)

        # Calculate offsets and strides

        sun_offset = 0
        sun_stride = len(sun_breaks) + len(zenith_breaks)

        zenith_offset = sun_offset + len(sun_breaks)
        zenith_stride = sun_stride

        emph_offset = sun_offset + rank * sun_stride

        total_coefs_single_config = emph_offset + len(emph_breaks)  # this is for one specific configuration

        total_coefs_all_configs = total_coefs_single_config * self.__total_configs

        self.__metadata_rad = Metadata(
            rank=rank,
            sun_offset=sun_offset, sun_stride=sun_stride, sun_breaks=sun_breaks,
            zenith_offset=zenith_offset, zenith_stride=zenith_stride, zenith_breaks=zenith_breaks,
            emph_offset=emph_offset, emph_breaks=emph_breaks,
            total_coefs_single_config=total_coefs_single_config,
            total_coefs_all_config=total_coefs_all_configs)

        # Read data
        #
        # Structure of the data part of the data file:
        # [[[[[[
        #       sun_coefs(sun_nbreaks * half), zenith_scale(1 * double), zenith_coefs(zenith_nbreaks * half)
        #      ] * rank_rad, emph_coefs (emph_nbreaks * half)
        #     ] * channels
        #    ] * elevations
        #   ] * altitudes
        #  ] * albedos
        # ] * visibilities

        # offset: int = 0
        self.__data_rad = np.zeros((self.__total_configs, total_coefs_single_config), dtype='float64')

        # one_config_byte_count = (
        #     ((len(self.__metadata_rad.sun_breaks) + len(self.__metadata_rad.zenith_breaks)) * USHORT_SIZE +
        #      DOUBLE_SIZE) * self.__metadata_rad.rank +
        #     len(self.__metadata_rad.emph_breaks) * USHORT_SIZE)

        # if a single visibility was requested, skip all configurations from the beginning till those needed for
        # the requested visibility.
        # handle.seek(one_config_byte_count * self.__skipped_configs_begin, SEEK_CUR)

        len_sun_breaks = len(sun_breaks)
        len_zenith_breaks = len(zenith_breaks)
        len_emph_breaks = len(emph_breaks)
        data_struct = ((
                               USHORT_CHAR * len_sun_breaks +
                               DOUBLE_CHAR * 1 +
                               USHORT_CHAR * len_zenith_breaks
                       ) * rank +
                       USHORT_CHAR * len_emph_breaks
                       ) * self.__total_configs
        data_list = read_list(handle, data_struct)

        read_rad_parallel(data_list, self.__data_rad, rank, len_sun_breaks, len_zenith_breaks, len_emph_breaks)

        # Skip remaining configurations till the end.
        # handle.seek(one_config_byte_count * self.__skipped_configs_end, SEEK_CUR)

    def __read_transmittance(self, handle):
        """
        Reads transmittance part of the dataset file into memory.

        Throws DatasetReadException if an error occurred while reading the dataset file.

        Parameters
        ----------
        handle : file
        """

        # Read metadata

        self.__d_dim = read_int(handle)
        if self.__d_dim < 1:
            raise DatasetReadException("d_dim")

        self.__a_dim = read_int(handle)
        if self.__a_dim < 1:
            raise DatasetReadException("a_dim")

        trans_visibilities = read_int(handle)

        trans_altitudes = read_int(handle)

        self.__rank_trans = read_int(handle)

        self.__altitudes_trans = np.float64(read_float_list(handle, size=trans_altitudes))

        self.__visibilities_trans = np.float64(read_float_list(handle, size=trans_visibilities))

        total_coefs_u = self.__d_dim * self.__a_dim * self.__rank_trans * trans_altitudes
        total_coefs_v = trans_visibilities * self.__rank_trans * self.__nb_channels * trans_altitudes

        # Read data

        self.__data_trans_u = read_float_list(handle, size=total_coefs_u)
        self.__data_trans_v = read_float_list(handle, size=total_coefs_v)

    def __read_polarisation(self, handle):
        """
        Reads polarisation part of the dataset file into memory.
        Throws DatasetReadException if an error occurred while reading the dataset file.

        Parameters
        ----------
        handle : file
        """

        # Read metadata
        # Structure of the metadata part of the data file:
        # rank_pol                 (1 * int),
        # sun_breaks_pol_count     (1 * int), sun_breaks_pol     (sun_breaks_pol_count * double),
        # zenith_breaks_pol_count  (1 * int), zenith_breaks_pol  (zenith_breaks_pol_count * double)

        rank_pol = read_int(handle)

        sun_breaks_pol = read_double_list(handle)
        zenith_breaks_pol = read_double_list(handle)

        # Calculate offsets and strides
        sun_offset_pol = 0
        sun_stride_pol = len(sun_breaks_pol) + len(zenith_breaks_pol)

        zenith_offset_pol = sun_offset_pol + len(sun_breaks_pol)
        zenith_stride_pol = sun_stride_pol

        total_coefs_single_config_pol = sun_offset_pol + rank_pol * sun_stride_pol
        total_coefs_all_configs_pol = total_coefs_single_config_pol * self.__total_configs

        self.__metadata_pol = Metadata(
            rank=rank_pol,
            sun_offset=sun_offset_pol, sun_stride=sun_stride_pol, sun_breaks=sun_breaks_pol,
            zenith_offset=zenith_offset_pol, zenith_stride=zenith_stride_pol, zenith_breaks=zenith_breaks_pol,
            emph_offset=0, emph_breaks=np.empty(0),
            total_coefs_single_config=total_coefs_single_config_pol, total_coefs_all_config=total_coefs_all_configs_pol
        )

        # Read data
        # Structure of the data part of the data file:
        # [[[[[[
        #       sun_coefs_pol (sun_nbreaks_pol * float), zenith_coefs_pol (zenith_nbreaks_pol * float)
        #      ] * rank
        #     ] * channels
        #    ] * elevations
        #   ] * altitudes
        #  ] * albedos
        # ] * visibilities

        len_sun_breaks = len(sun_breaks_pol)
        len_zenith_breaks = len(zenith_breaks_pol)
        data_struct = ((
                               FLOAT_CHAR * len_sun_breaks +
                               FLOAT_CHAR * len_zenith_breaks
                       ) * rank_pol
                       ) * self.__total_configs
        data_list = read_list(handle, data_struct)

        self.__data_pol = np.float64(np.reshape(data_list, (self.__total_configs, total_coefs_single_config_pol)))

        # if a single visibility was requested, skip all configurations from the beginning till those needed for
        # the requested visibility.
        # handle.seek(one_config_byte_count * self.__skipped_configs_begin, SEEK_CUR)

        # offset = 0
        # for con in range(self.__total_configs):
        #     for r in range(rank_pol):
        #         polarisation_temp = np.array(read_double_list(handle, size=len(sun_breaks_pol)), dtype='float64')
        #         self.__data_pol[offset:len(sun_breaks_pol)+offset] = polarisation_temp
        #         offset += len(sun_breaks_pol)
        #         # offset += compute_pp_coef(sun_breaks_pol, polarisation_temp, self.__data_pol, offset)
        #
        #         polarisation_temp = np.array(read_double_list(handle, size=len(zenith_breaks_pol)), dtype='float64')
        #         self.__data_pol[offset:len(zenith_breaks_pol)+offset] = polarisation_temp
        #         offset += len(zenith_breaks_pol)
        #         # offset += compute_pp_coef(zenith_breaks_pol, polarisation_temp, self.__data_pol, offset)

    def __evaluate_model(self, params, wavelength, data, metadata):
        """
        Evaluates the model. Used for computing sky radiance and polarisation.

        Parameters
        ----------
        params : Parameters
        wavelength : np.ndarray[float]
        data : np.ndarray[float]
        metadata : Metadata

        Returns
        -------
        float
        """

        result = np.zeros((len(wavelength), len(params.gamma)), dtype='float64')

        # Ignore wavelengths outside the dataset range
        wl_in_range = np.all([
            wavelength >= self.__channel_start,
            wavelength < self.__channel_start + self.__nb_channels * self.__channel_width
        ], axis=0)

        # Don't interpolate wavelengths inside the dataset range
        channel_index = np.int32(np.floor((wavelength - self.__channel_start) / self.__channel_width))[wl_in_range]

        # Translate angle values to indices and interpolation factors
        gamma = self.get_interpolation_parameter(params.gamma, metadata.sun_breaks)
        if metadata.emph_breaks is not None and len(metadata.emph_breaks) > 0:  # for radiance
            alpha = self.get_interpolation_parameter(params.shadow if params.elevation < 0 else params.zero,
                                                     metadata.zenith_breaks)
            zero = self.get_interpolation_parameter(params.zero, metadata.emph_breaks)
        else:  # for polarisation
            alpha = self.get_interpolation_parameter(params.zero, metadata.zenith_breaks)
            zero = None
        angle_parameters = AngleParameters(gamma=gamma, alpha=alpha, zero=zero)

        # Translate configuration values to indices and interpolation factors
        visibility_param = self.get_interpolation_parameter(params.visibility, self.__visibilities_rad)
        albedo_param = self.get_interpolation_parameter(params.albedo, self.__albedos_rad)
        altitude_param = self.get_interpolation_parameter(params.altitude, self.__altitudes_rad)
        elevation_param = self.get_interpolation_parameter(np.rad2deg(params.elevation), self.__elevations_rad)

        # Prepare parameters controlling the interpolation
        coefficients = np.zeros((16, len(channel_index), metadata.total_coefs_single_config), dtype='float64')
        for i in range(16):
            visibility_index = np.minimum(visibility_param.index + i // 8, len(self.__visibilities_rad) - 1)
            albedo_index = np.minimum(albedo_param.index + (i % 8) // 4, len(self.__albedos_rad) - 1)
            altitude_index = np.minimum(altitude_param.index + (i % 4) // 2, len(self.__altitudes_rad) - 1)
            elevation_index = np.minimum(elevation_param.index + (i % 2), len(self.__elevations_rad) - 1)

            coefficients[i] = self.__get_coefficients(data, elevation_index,
                                                      altitude_index, visibility_index,
                                                      albedo_index, channel_index)
        interpolation_factor = np.array([
            visibility_param.factor,
            albedo_param.factor,
            altitude_param.factor,
            elevation_param.factor
        ], dtype='float64')

        control_params = ControlParameters(coefficients, interpolation_factor)

        # Interpolate
        result[wl_in_range, :] = self.interpolate(0, 0, angle_parameters, control_params, metadata)

        # polarisation can be negative
        assert metadata.emph_breaks is None or len(metadata.emph_breaks) == 0 or np.all(result >= 0)

        return result

    def __get_coefficients(self, dataset, elevation, altitude, visibility, albedo, wavelengths):
        """
        Gets iterator to coefficients in the dataset array corresponding to the given configuration. Used for
        sky radiance and polarisation.

        Parameters
        ----------
        dataset : np.ndarray
        elevation : int
        altitude : int
        visibility : int
        albedo : int
        wavelengths : int,

        Returns
        -------
        np.ndarray[float]
        """
        ncha = self.__nb_channels
        nele = len(self.__elevations_rad)
        nalt = len(self.__altitudes_rad)
        nalb = len(self.__albedos_rad)
        return dataset[
                wavelengths +
                ncha * elevation +
                ncha * nele * altitude +
                ncha * nele * nalt * albedo +
                ncha * nele * nalt * nalb * visibility
        ]

    def __get_coefficients_trans(self, visibility, altitude, wavelength):
        """
        Gets iterator to base transmittance coefficients in the dataset array corresponding to the given
        configuration.

        Parameters
        ----------
        visibility : int
        altitude : int
        wavelength : np.ndarray[int]

        Returns
        -------
        np.ndarray
        """
        index_start = (((int(visibility) * len(self.__altitudes_trans) + int(altitude)
                         ) * int(self.__nb_channels) + np.int32(wavelength)
                        ) * int(self.__rank_trans))

        return self.__data_trans_v[np.linspace(index_start, index_start + self.__rank_trans,
                                               self.__rank_trans, endpoint=False, dtype='int32')]

    def __get_coefficients_trans_base(self, altitude, a, d):
        """
        Gets iterator to base transmittance coefficients in the dataset array corresponding to the given
        configuration.

        Parameters
        ----------
        altitude : int
        a : int, np.ndarray[int]
        d : int, np.ndarray[int]

        Returns
        -------
        np.ndarray
        """
        index_start = (altitude * int(self.__a_dim) * int(self.__d_dim) * int(self.__rank_trans) +
                       (np.int32(d) * int(self.__a_dim) + np.int32(a)) * int(self.__rank_trans))

        return self.__data_trans_u[np.linspace(index_start, index_start + self.__rank_trans,
                                               self.__rank_trans, endpoint=False, dtype='int32')]

    def __interpolate_trans(self, visibility_index, altitude_param, trans_params, channel_index):
        """
        Interpolates transmittances computed for two nearest altitudes.

        Parameters
        ----------
        visibility_index : int
        altitude_param : InterpolationParameter
        trans_params : TransmittanceParameters
        channel_index : np.ndarray[int]

        Returns
        -------
        np.ndarray[float]
        """

        # Get transmittance for the nearest lower altitude.
        trans = self.__reconstruct_trans(visibility_index, altitude_param.index, trans_params, channel_index)

        # Interpolate with transmittance for the nearest higher altitude if needed.
        if altitude_param.factor > 0:
            trans_high = self.__reconstruct_trans(
                visibility_index, altitude_param.index + 1, trans_params, channel_index)
            trans = lerp(trans, trans_high, altitude_param.factor)

        return trans

    def __reconstruct_trans(self, visibility_index, altitude_index, trans_params, channel_index):
        """
        For each channel reconstructs transmittances for the four nearest transmittance parameters and
        interpolates them.

        Parameters
        ----------
        visibility_index : int
        altitude_index : int
        trans_params : TransmittanceParameters
        channel_index : np.ndarray[int]

        Returns
        -------
        np.ndarray[float]
        """

        coefs = self.__get_coefficients_trans(visibility_index, altitude_index, channel_index)

        # Load transmittance values for bi-linear interpolation
        transmittance = np.zeros((self.__nb_channels, 4, trans_params.altitude.index.size), dtype='float64')

        a = np.linspace(trans_params.altitude.index, trans_params.altitude.index + 1, 2, dtype='int32')
        d = np.linspace(trans_params.distance.index, trans_params.distance.index + 1, 2, dtype='int32')
        a = np.repeat(a[:, None], d.shape[0], axis=1)
        d = np.repeat(d[None, :], a.shape[0], axis=0)
        a = np.reshape(a, (-1, a.shape[-1]))
        d = np.reshape(d, (-1, d.shape[-1]))
        valid = np.all([a < self.__a_dim, d < self.__d_dim], axis=0)
        base_coefs = self.__get_coefficients_trans_base(altitude_index, a[valid], d[valid])

        transmittance[:, valid] = np.sum(np.float64(base_coefs[:, None, :]) * np.float64(coefs[:, :, None]), axis=0)
        transmittance = np.transpose(transmittance, axes=(1, 0, 2))

        # Perform bi-linear interpolation
        valid_factor = trans_params.distance.factor > 0
        transmittance[0, :, valid_factor] = lerp(transmittance[0, :, valid_factor],
                                                 transmittance[1, :, valid_factor],
                                                 trans_params.distance.factor[valid_factor, None])
        transmittance[1, :, valid_factor] = lerp(transmittance[2, :, valid_factor],
                                                 transmittance[3, :, valid_factor],
                                                 trans_params.distance.factor[valid_factor, None])
        transmittance[0] = np.maximum(transmittance[0], 0)

        valid_factor = trans_params.altitude.factor > 0
        transmittance[1] = np.maximum(transmittance[1], 0)
        transmittance[0, :, valid_factor] = lerp(transmittance[0, :, valid_factor],
                                                 transmittance[1, :, valid_factor],
                                                 trans_params.altitude.factor[valid_factor, None])

        assert np.all(transmittance[0] >= 0)

        return transmittance[0]

    def __to_transmittance_params(self, theta, distance, altitude):
        """
        Transforms the given theta angle, observer altitude and distance along ray into transmittance model
        internal [altitude, distance] parametrization.

        Parameters
        ----------
        theta : float, np.ndarray[float]
        distance : float
        altitude : float

        Returns
        -------
        TransmittanceParameters
        """
        assert np.all([0 <= theta, theta <= np.pi])
        assert 0 <= distance
        assert 0 <= altitude

        ray_dir_x = np.sin(theta)
        ray_dir_y = np.cos(theta)
        ray_pos_y = PLANET_RADIUS + altitude  # ray_pos_x = 0

        ATMOSPHERE_EDGE = PLANET_RADIUS + ATMOSPHERE_WIDTH

        # Find intersection of the ground-to-sun ray with edge of the atmosphere (in 2D)
        dist_to_isect = np.full_like(theta, -1.0)
        LOW_ALTITUDE = 0.3

        if altitude < LOW_ALTITUDE:
            # Special handling of almost zero altitude case to avoid numerical issues.
            dist_to_isect[:] = 0.
            is_theta_close = theta <= 0.5 * np.pi
            dist_to_isect[is_theta_close] = intersect_ray_with_circle_2d(
                ray_dir_x[is_theta_close], ray_dir_y[is_theta_close], ray_pos_y, ATMOSPHERE_EDGE)
        else:
            dist_to_isect[:] = intersect_ray_with_circle_2d(ray_dir_x, ray_dir_y, ray_pos_y, PLANET_RADIUS)
            negative_dist = dist_to_isect < 0
            dist_to_isect[negative_dist] = intersect_ray_with_circle_2d(
                ray_dir_x[negative_dist], ray_dir_y[negative_dist], ray_pos_y, ATMOSPHERE_EDGE)

        # The ray should always hit either the edge of the atmosphere or the planet (we are starting inside the
        # atmosphere).
        assert np.all(dist_to_isect >= 0)

        dist_to_isect = np.minimum(dist_to_isect, distance)

        # Compute intersection coordinates
        isect_x = ray_dir_x * dist_to_isect
        isect_y = ray_dir_y * dist_to_isect + ray_pos_y

        # Get the internal [altitude, distance] parameters
        altitude_param, distance_param = isect_to_altitude_distance(isect_x, isect_y)

        # Convert to interpolation parameters
        return TransmittanceParameters(
            self.get_interpolation_parameter_trans(altitude_param, self.__a_dim, 3),
            self.get_interpolation_parameter_trans(distance_param, self.__d_dim, 4)
        )

    @staticmethod
    def get_interpolation_parameter(query_val, breaks):
        """
        Get interpolation parameter for the given query value, i.e. finds position of the query value between
        a pair of break values.

        Used for albedo, elevation, altitude, visibility and angles (theta, alpha, or gamma).

        Parameters
        ----------
        query_val : float, np.ndarray[float]
        breaks : np.ndarray[float]

        Returns
        -------
        InterpolationParameter
        """

        # Clamp the value to the valid range
        clamped = np.float64(np.clip(query_val, breaks[0], breaks[-1]))

        one = 1

        # Get the nearest greater parameter value
        next_greater = np.searchsorted(breaks[1:], clamped, side='right') + one
        # next_greater = bisect_right(breaks[:, None], clamped[None, ...], 1)

        # Compute the index and float factor
        index = next_greater - one
        factor = np.zeros(index.shape, dtype='float64')
        factor_valid = np.all([one < next_greater, next_greater < (len(breaks) - one)], axis=0)
        if factor.size == 1 and factor_valid:
            factor = (float(clamped - breaks[next_greater - one]) /
                      float(breaks[next_greater + 1 - one] - breaks[next_greater - one]))
        else:
            factor[factor_valid] = (np.float64(clamped[factor_valid] - breaks[next_greater[factor_valid] - one]) /
                                    np.float64(breaks[next_greater[factor_valid] + 1 - one] -
                                               breaks[next_greater[factor_valid] - one]))

        assert np.all([0 <= index, index < len(breaks), np.any([index < len(breaks) - 1, factor == 0], axis=0)])
        assert np.all([0 <= factor, factor <= 1]), (
            f"Factor must be in range [0, 1]. "
            f"Out of the {len(factor)} factors, there were "
            f"{np.sum(np.any([factor < 0], axis=0))} factors < 0 and "
            f"{np.sum(np.any([factor > 1], axis=0))} factors > 1."
        )

        return InterpolationParameter(factor=factor, index=index)

    @staticmethod
    def get_interpolation_parameter_trans(value, param_count, power):
        """
        Converts altitude or distance value used in the transmittance model into interpolation parameter.

        Parameters
        ----------
        value : np.ndarray[float]
        param_count : int
        power : int

        Returns
        -------
        InterpolationParameter
        """
        index = np.minimum(np.int32(value * param_count), param_count - 1)
        factor = np.zeros(value.shape, dtype='float64')

        in_range = index < param_count - 1
        factor[in_range] = np.clip(nonlerp(np.float64(index[in_range]) / param_count,
                                           np.float64(index[in_range] + 1) / param_count,
                                           value[in_range], power), 0, 1)

        # if index < param_count - 1:
        #     factor = nonlerp(float(index) / param_count, float(index + 1) / param_count, value, power)
        #     factor = np.clip(factor, 0, 1)
        return InterpolationParameter(factor, index)

    @staticmethod
    def reconstruct(radiance_parameters, channel_parameters, metadata):
        """
        Reconstructs sky radiance or polarisation from the given control parameters by inverse tensor decomposition.

        Parameters
        ----------
        radiance_parameters : AngleParameters
        channel_parameters : np.ndarray[float]
        metadata : Metadata

        Returns
        -------
        float
        """

        result = 0.0
        for r in range(metadata.rank):
            # Restore the right value in the 'sun' vector
            i_sun = r * metadata.sun_stride + metadata.sun_offset + radiance_parameters.gamma.index
            sun_param = eval_pl(
                np.array([channel_parameters[0, ..., i_sun].T, channel_parameters[0, ..., i_sun+1].T]),
                radiance_parameters.gamma.factor)

            # Restore the right value in the 'zenith' vector
            i_zen = r * metadata.zenith_stride + metadata.zenith_offset + radiance_parameters.alpha.index
            zenith_param = eval_pl(
                np.array([channel_parameters[0, ..., i_zen].T, channel_parameters[0, ..., i_zen+1].T]),
                radiance_parameters.alpha.factor)

            # Accumulate their "outer" product
            result += sun_param * zenith_param

        # De-emphasize (for radiance only)
        if metadata.emph_breaks is not None and len(metadata.emph_breaks) > 0:
            i_emp = metadata.emph_offset + radiance_parameters.zero.index
            emph_param = eval_pl(
                np.array([channel_parameters[0, ..., i_emp].T, channel_parameters[0, ..., i_emp+1].T]),
                radiance_parameters.zero.factor)
            result *= emph_param
            result = np.maximum(result, 0)

        return result

    @staticmethod
    def interpolate(t_offset, t_level, angle_parameters, control_parameters, metadata):
        """
        Recursive function controlling interpolation of reconstructed radiance between two neighboring visibility,
        albedo, altitude and elevation values.

        Parameters
        ----------
        t_offset : int
        t_level : int
        angle_parameters : AngleParameters
        control_parameters : ControlParameters
        metadata : Metadata

        Returns
        -------
        float
        """

        # Starts at level 0 and recursively goes down to level 4 while computing offset to the control
        # parameters array. There it reconstructs radiance. When returning from recursion interpolates
        # according to elevation, altitude, albedo and visibility at level 3, 2, 1 and 0, respectively.
        if t_level == 4:
            return PragueSkyModel.reconstruct(angle_parameters, control_parameters.coefficients[t_offset:], metadata)
        else:
            # Compute the first value
            result_low = PragueSkyModel.interpolate(t_offset, t_level + 1,
                                                    angle_parameters, control_parameters, metadata)

            # Skip the second value if not useful or not available.
            if control_parameters.interpolation_factor[t_level] < 1e-06:
                return result_low

            # Compute the second value
            result_high = PragueSkyModel.interpolate(t_offset + (1 << (3 - t_level)), t_level + 1,
                                                     angle_parameters, control_parameters, metadata)

            return lerp(result_low, result_high, control_parameters.interpolation_factor[t_level])


def lerp(start, end, factor):
    return (1 - factor) * np.float64(start) + factor * np.float64(end)


def nonlerp(a: float, b: float, w: float, p: float):
    c1 = np.power(a, p)
    c2 = np.power(b, p)
    return (np.power(w, p) - c1) / (c2 - c1)


def eval_pl(coefs, factor):
    """
    Evaluates piecewise linear approximation.

    Parameters
    ----------
    coefs : np.ndarray[float], list
    factor : float

    Returns
    -------
    float
    """
    return (np.float64(coefs[1]) - np.float64(coefs[0])) * factor + np.float64(coefs[0])


def intersect_ray_with_circle_2d(ray_dir_x, ray_dir_y, ray_pos_y, circle_radius):
    """
    Intersects the given ray (assuming ray_pos_x == 0) with a circle at origin with the given radius.

    Parameters
    ----------
    ray_dir_x : np.ndarray[float]
    ray_dir_y : np.ndarray[float]
    ray_pos_y : float
    circle_radius : float

    Returns
    -------
    np.ndarray[float]
        In the case the ray intersects the circle, distance to the circle is returned, otherwise the
        function returns negative number.
    """

    assert ray_pos_y > 0
    assert circle_radius > 0

    # Compute discriminant
    qa = ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y
    qb = 2.0 * ray_pos_y * ray_dir_y
    qc = ray_pos_y * ray_pos_y - circle_radius * circle_radius
    discrim = qb * qb - 4.0 * qa * qc

    distance_to_isect = np.zeros(discrim.shape, dtype='float64')

    # No intersection or touch only
    touch = discrim <= 0
    distance_to_isect[touch] = -1

    discrim[~touch] = np.sqrt(discrim[~touch])

    # Compute distances to both intersections
    d1 = (-qb + discrim) / np.maximum(2.0 * qa, np.finfo(float).eps)
    d2 = (-qb - discrim) / np.maximum(2.0 * qa, np.finfo(float).eps)

    # Try to take the nearest positive one
    both_positive = np.all([d1 > 0, d2 > 0], axis=0)

    no_touch_both_positive = np.all([~touch, both_positive], axis=0)
    distance_to_isect[no_touch_both_positive] = np.minimum(d1[no_touch_both_positive], d2[no_touch_both_positive])

    no_touch_not_both_positive = np.all([~touch, ~both_positive], axis=0)
    distance_to_isect[no_touch_not_both_positive] = np.maximum(d1[no_touch_not_both_positive],
                                                               d2[no_touch_not_both_positive])

    return distance_to_isect


def isect_to_altitude_distance(isect_x, isect_y):
    """
    Auxiliary function for \ref toTransmittanceParams. Computes [altitude, distance] parameters from
    coordinates of intersection of view ray and ground or atmosphere edge.

    Note: using floats here instead of doubles will cause banding artifacts.

    Parameters
    ----------
    isect_x : np.ndarray[float]
    isect_y : np.ndarray[float]

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[float]]
    """

    # Distance to the intersection from world origin (not along ray as dist_to_isect in the calling method).
    isect_dist = np.sqrt(isect_x * isect_x + isect_y * isect_y)
    assert np.all(isect_dist > 0)

    # Compute normalized and non-linearly scaled position in the atmosphere
    altitude = np.clip(isect_dist - PLANET_RADIUS, 0, ATMOSPHERE_WIDTH)
    altitude = np.power(altitude / ATMOSPHERE_WIDTH, 1.0 / 3.0)
    distance = np.arccos(isect_y / isect_dist) * PLANET_RADIUS
    distance = np.sqrt(distance / DIST_TO_EDGE)
    distance = np.sqrt(distance)  # Calling twice sqrt, since it is faster than np.power(..., 0.25)
    distance = np.minimum(1, distance)
    assert np.all([0 <= altitude, altitude <= 1])
    assert np.all([0 <= distance, distance <= 1])

    return altitude, distance


# @njit(parallel=True)
def read_rad_parallel(data_list, data_set, rank, len_sun_breaks, len_zenith_breaks, len_emph_breaks):
    total_config, single_config = data_set.shape
    list_offset = 0
    for con in range(total_config):
        offset = 0
        for r in range(rank):
            # list_offset = con * single_config + r * (len_sun_breaks + len_zenith_breaks + 1)
            # offset = r * (len_sun_breaks + len_zenith_breaks)

            # np.view changes the interpretation of the bytes from uint16 to float16
            # np.float64 simply casts the values to be handled as doubles
            ushort_temp = np.array(data_list[list_offset:len_sun_breaks+list_offset], dtype=USHORT_TYPE)
            double_temp = np.float64(ushort_temp.view(FLOAT2_TYPE))
            data_set[con, offset:len_sun_breaks+offset] = double_temp
            list_offset += len_sun_breaks
            offset += len_sun_breaks

            # sometimes zenith_scale is very small
            zenith_scale = float(data_list[list_offset])
            list_offset += 1

            ushort_temp = np.array(data_list[list_offset:len_zenith_breaks+list_offset], dtype=USHORT_TYPE)
            double_temp = np.float64(ushort_temp.view(FLOAT2_TYPE))
            data_set[con, offset:len_zenith_breaks+offset] = double_temp / zenith_scale
            list_offset += len_zenith_breaks
            offset += len_zenith_breaks

        ushort_temp = np.array(data_list[list_offset:len_emph_breaks+list_offset], dtype=USHORT_TYPE)
        double_temp = np.float64(ushort_temp.view(FLOAT2_TYPE))
        data_set[con, offset:len_emph_breaks+offset] = double_temp
        list_offset += len_emph_breaks
        offset += len_emph_breaks


@vectorize(["float64(uint16)"], nopython=True)
def half2double(half):
    h_sig = float(half >> 15)
    h_exp = float(max((half & 0x7fff) >> 10, 1))
    h_fra = float(half & 0x0400)

    return (-1) ** h_sig * 2 ** (h_exp - 15) * (h_fra / 1024)

# @vectorize(["uint64(uint16)"], nopython=True)
# def half2double(half):
#     half = half & 0xfff  # make sure it's uint16
#     hi = ((half & 0x8000) << 16) & 0xffffffff  # uint32
#     ab = half & 0x7fff  # uint16
#     if ab:
#         hi = hi | (0x3f000000 << (int(ab >= 0x7c00) & 0xffff))
#         while ab < 0x400:
#             ab = (ab << 1) & 0xffff
#             hi = (hi - 0x100000) & 0xffffffff
#         # hi = (hi + (ab & 0xffffffff) << 10) & 0xffffffff
#         hi = (hi + ((ab & 0xffffffff) << 10)) & 0xffffffff
#     d_bits = (hi << 32) & 0xffffffffffffffff
#     # d_bytes = struct.pack('Q', d_bits)
#     # d_float, = struct.unpack('d', d_bytes)
#
#     return d_bits
#     # return float(d_float if not np.isnan(d_float) else 0.)

# def find_segment(x, breaks):
#     segment = 0
#     for segment in range(len(breaks)):
#         if breaks[segment+1] > x:
#             break
#     return segment


# def eval_pp(x, segment, breaks, coefs):
#     x0 = x - breaks[segment]
#     sc = coefs[2 * segment:]  # segment coefs
#     return sc[0] * x0 + sc[1]


# def control_params_single_config(state: SkyModelState, dataset, total_coef_single_config,
#                                  elevation, altitude, turbidity, albedo, wavelength):
#     return dataset[total_coef_single_config * (
#             wavelength +
#             state.channels * elevation +
#             state.channels * len(state.elevations) + altitude +
#             state.channels * len(state.elevations) * len(state.altitudes) * albedo +
#             state.channels * len(state.elevations) * len(state.altitudes) * len(state.albedos) * turbidity)]


# def reconstruct(state: SkyModelState, gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment, control_params):
#     res: float = 0
#     for t in range(state.tensor_components):
#         sun_val_t = eval_pp(
#             gamma, gamma_segment, state.sun_breaks, control_params[state.sun_offset + t * state.sun_stride]
#         )
#         zenith_val_t = eval_pp(
#             alpha, alpha_segment, state.zenith_breaks, control_params[state.zenith_offset + t * state.zenith_stride]
#         )
#         res += sun_val_t * zenith_val_t
#         emph_val_t = eval_pp(
#             zero, zero_segment, state.emph_breaks, control_params[state.emph_offset]
#         )
#         res *= emph_val_t
#
#     return np.maximum(res, 0)


# def map_parameter(param, values):
#     if param < values[0]:
#         mapped: float = 0
#     elif param > values[-1]:
#         mapped: float = len(values) - 1
#     else:
#         for v, val in enumerate(values):
#             if abs(val - param) < 1e-6:
#                 mapped: float = v
#                 break
#             elif param < val:
#                 mapped: float = v - (val - param) / (val - values[v - 1])
#                 break
#     return mapped


# def interpolate_elevation(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
#                           gamma_segment, alpha_segment, zero_segment):
#     elevation_low = float(int(elevation))
#     factor = elevation - elevation_low
#
#     control_params_low = control_params_single_config(
#         state, state.radiance_dataset, state.total_coefs_single_config,
#         elevation_low, altitude, turbidity, albedo, wavelength
#     )
#     res_low = reconstruct(
#         state, gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment, control_params_low
#     )
#     if factor < 01e-6 or elevation_low >= len(state.elevations) - 1:
#         return res_low
#
#     control_params_high = control_params_single_config(
#         state, state.radiance_dataset, state.total_coefs_single_config,
#         elevation_low+1, altitude, turbidity, albedo, wavelength
#     )
#     res_high = reconstruct(
#         state, gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment, control_params_high
#     )
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_altitude(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
#                          gamma_segment, alpha_segment, zero_segment):
#     altitude_low = float(int(altitude))
#     factor = altitude - altitude_low
#
#     res_low = interpolate_elevation(
#         state, elevation, altitude_low, turbidity, albedo, wavelength, gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )
#     if factor < 01e-6 or altitude_low >= (len(state.altitudes) - 1):
#         return res_low
#     res_high = interpolate_elevation(
#         state, elevation, altitude_low + 1, turbidity, albedo, wavelength, gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_turbidity(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
#                           gamma_segment, alpha_segment, zero_segment):
#     turbidity_low = float(int(turbidity))
#     factor = turbidity - turbidity_low
#
#     res_low = interpolate_altitude(
#         state, elevation, altitude, turbidity_low, albedo, wavelength, gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )
#     if factor < 01e-6 or turbidity_low >= (len(state.turbidities) - 1):
#         return res_low
#     res_high = interpolate_altitude(
#         state, elevation, altitude, turbidity_low + 1, albedo, wavelength, gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_albedo(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength, gamma, alpha, zero,
#                        gamma_segment, alpha_segment, zero_segment):
#     albedo_low = float(int(albedo))
#     factor = albedo - albedo_low
#
#     res_low = interpolate_turbidity(
#         state, elevation, altitude, turbidity, albedo_low, wavelength, gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )
#     if factor < 01e-6 or albedo_low >= (len(state.albedos) - 1):
#         return res_low
#     res_high = interpolate_turbidity(
#         state, elevation, altitude, turbidity, albedo_low + 1, wavelength, gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_wavelengths(state: SkyModelState, elevation, altitude, turbidity, albedo, wavelength,
#                             gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment):
#     # Don't interpolate, use the bin it belongs to
#     return interpolate_albedo(
#         state, elevation, altitude, turbidity, albedo, float(int(wavelength)), gamma, alpha, zero,
#         gamma_segment, alpha_segment, zero_segment
#     )


# def sky_model_radiance(state, theta, gamma, shadow, zero, elevation, altitude, turbidity, albedo, wavelength):
#     """
#
#     Parameters
#     ----------
#     state : SkyModelState
#     theta : float
#     gamma : float
#     shadow : float
#     zero : float
#     elevation : float
#     altitude : float
#     turbidity : float
#     albedo : float
#     wavelength : float
#
#     Returns
#     -------
#     float
#     """
#
#     # translate parameter values to indices
#     turbidity_control = map_parameter(turbidity, state.turbidities)
#     albedo_control = map_parameter(albedo, state.albedos)
#     altitude_control = map_parameter(altitude, state.altitudes)
#     elevation_control = map_parameter(elevation, state.elevations)
#
#     channel_control = (wavelength - state.channel_start) / state.channel_width
#
#     if channel_control >= state.channels or channel_control < 0:
#         return 0.
#
#     # Get params corresponding to the indices, reconstruct result and interpolate
#
#     alpha = shadow if elevation < 0 else zero
#     gamma_segment = find_segment(gamma, state.sun_breaks)
#     alpha_segment = find_segment(alpha, state.zenith_breaks)
#     zero_segment = find_segment(zero, state.emph_breaks)
#
#     res = interpolate_wavelengths(
#         state, elevation_control, altitude_control, turbidity_control, albedo_control, channel_control,
#         gamma, alpha, zero, gamma_segment, alpha_segment, zero_segment
#     )
#
#     return res


# def sky_model_solar_radiance(state, theta, gamma, shadow, zero, elevation, altitude, turbidity, albedo, wavelength):
#     """
#     This computes transmittance between a point at 'altitude' and infinity in
#     the direction 'theta' at a wavelength 'wavelength'.
#
#     Parameters
#     ----------
#     state : SkyModelState
#     theta : float
#     gamma : float
#     shadow : float
#     zero : float
#     elevation : float
#     altitude : float
#     turbidity : float
#     albedo : float
#     wavelength : float
#
#     Returns
#     -------
#     float
#     """
#     idx = (wavelength - sun_rad_star_wl) / sun_rad_increment_wl
#     sun_radiance = 0
#
#     if idx >= 0:
#         low_idx = int(idx)
#         idx_float = idx - float(low_idx)
#
#         sun_radiance = sun_rad_table[low_idx] * (1 - idx_float) + sun_rad_table[low_idx + 1] * idx_float
#
#     tau = sky_model_tau(state, theta, altitude, turbidity, wavelength, np.infty)
#
#     return sun_radiance * tau


# def sky_model_circle_bounds_2d(x_v: float, y_v: float, y_c: float, radius: float):
#     qa = x_v * x_v + y_v * y_v
#     qb = 2.0 * y_c * y_v
#     qc = y_c * y_c - radius * radius
#     n = qb * qb - 4.0 * qa * qc
#
#     if n <= 0:
#         return 0.
#
#     n = np.sqrt(n)
#     d1 = (-qb + n) / (2.0 * qa)
#     d2 = (-qb - n) / (2.0 * qa)
#     d = np.minimum(d1, d2) if (d1 > 0 and d2 > 0) else np.maximum(d1, d2)
#
#     return d > 0, d


# def sky_model_scale_ad(x_p: float, y_p: float):
#     n = np.sqrt(x_p * x_p + y_p * y_p)
#
#     a = n - PSM_PLANET_RADIUS
#     a = np.maximum(a, 0)
#     a = np.power(a / 100000, 1 / 3)
#
#     d = np.arccos(y_p / n) * PSM_PLANET_RADIUS
#     d = d / 1571524.413613  # Maximum distance to the edge of the atmosphere in the transmittance model
#     d = np.power(d, 0.25)
#     d = np.minimum(d, 1)
#
#     return a, d


# def sky_model_to_ad(theta, distance, altitude):
#     """
#
#     Parameters
#     ----------
#     theta : float
#     distance : float
#     altitude : float
#
#     Returns
#     -------
#     float
#     """
#     x_v = np.sin(theta)
#     y_v = np.cos(theta)
#     x_c = 0
#     y_c = PSM_PLANET_RADIUS + altitude
#     atmo_edge = PSM_PLANET_RADIUS + 90000
#
#     # Handle altitudes close to 0 separately to avoid reporting intersection on the other side of the planet
#     if altitude < 0.001:
#         if theta <= 0.5 * np.pi:
#             n_greater_than_0, n = sky_model_circle_bounds_2d(x_v, y_v, y_c, atmo_edge)
#             if not n_greater_than_0:
#                 # Then we have a problem!
#                 # Return something, but this should never happen so long as the camera is inside the atmosphere
#                 # which it should be in this work
#                 a = 0.
#                 d = 0.
#                 return a, d
#         else:
#             n = 0.
#     else:
#         n_greater_than_0, n = sky_model_circle_bounds_2d(x_v, y_v, y_c, PSM_PLANET_RADIUS)
#         if n_greater_than_0:  # Check for planet intersection
#             if n <= distance:  # We do intersect the planet so return a and d at the surface
#                 x_p = x_v * n
#                 y_p = y_v * n + PSM_PLANET_RADIUS + altitude
#                 a, d = sky_model_scale_ad(x_p, y_p)
#                 return a, d
#         n_greater_than_0, n = sky_model_circle_bounds_2d(x_v, y_v, y_c, atmo_edge)
#         if not n_greater_than_0:
#             # Then we have a problem!
#             # Return something, but this should never happen so long as the camera is inside the atmosphere
#             # which it should be in this work
#             a = 0
#             d = 0
#             return a, d
#
#     # Use the smaller of the distances
#     distance_corrected = np.minimum(distance, n)
#
#     # Points in world space
#     x_p = x_v * distance_corrected
#     y_p = y_v * distance_corrected + PSM_PLANET_RADIUS + altitude
#     a, d = sky_model_scale_ad(x_p, y_p)
#
#     return a, d


# def sky_model_transmittance_coefs_index(state: SkyModelState, turbidity: int, altitude: int, wavelength: int):
#     transmittance_values_per_turbidity = state.trans_rank * 11 * state.trans_altitudes
#     return state.transmission_dataset_V[
#         turbidity * transmittance_values_per_turbidity + (altitude * 11 + wavelength) * state.trans_rank]

#
# def sky_model_calc_transmittance_interpolate_wavelength(state: SkyModelState, turbidity: int, altitude: int,
#                                                         wavelength_low: int, wavelength_inc: int, wavelength_w: float):
#     wll = sky_model_transmittance_coefs_index(state, turbidity, altitude, wavelength_low)
#     wlu = sky_model_transmittance_coefs_index(state, turbidity, altitude, wavelength_low + wavelength_inc)
#
#     coefficients = np.zeros(state.trans_rank, dtype='float64')
#     for i in range(state.trans_rank):
#         coefficients[i] = lerp(wll[i], wlu[i], wavelength_w)
#
#     return coefficients
#
#
# def sky_model_calc_transmittance_svd_altitude(state: SkyModelState, turbidity: int, altitude: int,
#                                               wavelength_low: int, wavelength_inc: int, wavelength_factor: float,
#                                               a_int: int, d_int: int, a_inc: int, d_inc: int, wa: float, wd: float):
#     t = np.zeros(4, dtype='float32')
#
#     interpolated_coefficients = sky_model_calc_transmittance_interpolate_wavelength(
#         state, turbidity, altitude, wavelength_low, wavelength_inc, wavelength_factor)
#
#     index = 0
#     for al in range(a_int, a_int + a_inc + 1):
#         for dl in range(d_int, d_int + d_inc + 1):
#             for i in range(state.trans_rank):
#                 t[index] = t[index] + state.transmission_dataset_U[
#                     altitude * state.trans_n_a * state.trans_n_d * state.trans_rank +
#                     (dl * state.trans_n_a + al) * state.trans_rank + i
#                 ] * interpolated_coefficients[i]
#             index += 1
#     if d_inc == 1:
#         t[0] = lerp(t[0], t[1], wd)
#         t[1] = lerp(t[2], t[3], wd)
#     if a_inc == 1:
#         t[0] = lerp(t[0], t[1], wa)
#
#     return t[0]
#
#
# def sky_model_calc_transittance_svd(state: SkyModelState, a: float, d: float, turbidity: int,
#                                     wavelength_low: int, wavelength_inc: int, wavelength_factor: float,
#                                     altitude_low: int, altitude_inc: int, altitude_factor: float):
#     a_int = int(np.floor(a * float(state.trans_n_a)))
#     d_int = int(np.floor(d * float(state.trans_n_d)))
#     a_inc = 0
#     d_inc = 0
#     wa = (a * float(state.trans_n_a)) - float(a_int)
#     wd = (d * float(state.trans_n_d)) - float(d_int)
#
#     if a_int < state.trans_n_a - 1:
#         a_inc = 1
#         wa = nonlerp(float(a_int) / float(state.trans_n_a), float(a_int + a_inc) / float(state.trans_n_a), a, 3)
#     else:
#         a_int = state.trans_n_a - 1
#         wa = 0
#
#     if d_int < state.trans_n_d - 1:
#         d_inc = 1
#         wd = nonlerp(float(d_int) / float(state.trans_n_d), float(d_int + d_inc) / float(state.trans_n_d), d, 4)
#     else:
#         d_int = state.trans_n_d - 1
#         wd = 0
#
#     wa = float(np.clip(wa, 0, 1))
#     wd = float(np.clip(wd, 0, 1))
#
#     trans = np.zeros(2, dtype='float32')
#     trans[0] = sky_model_calc_transmittance_svd_altitude(state, turbidity, altitude_low,
#                                                          wavelength_low, wavelength_inc, wavelength_factor,
#                                                          a_int, d_int, a_inc, d_inc, wa, wd)
#     if altitude_inc == 1:
#         trans[1] = sky_model_calc_transmittance_svd_altitude(state, turbidity, altitude_low + altitude_inc,
#                                                              wavelength_low, wavelength_inc, wavelength_factor,
#                                                              a_int, d_int, a_inc, d_inc, wa, wd)
#         trans[0] = lerp(trans[0], trans[1], altitude_factor)
#
#     return trans[0]
#
#
# def cbrt(x: float):
#     return np.power(x, 1 / 3)
#
#
# def sky_model_find_in_array(arr: np.ndarray, value: float):
#     index = 0
#     inc = 0
#     w = 0
#
#     if value <= arr[0]:
#         index = 0
#         w = 1
#         return index, inc, w
#
#     if value >= arr[-1]:
#         index = len(arr) - 1
#         w = 0
#         return index, inc, w
#
#     for i in range(len(arr)):
#         if value < arr[i]:
#             index = i - 1
#             inc = 1
#             w = (value - arr[i - 1]) / (arr[i] - arr[i - 1])  # Assume linear
#             return index, inc, w
#
#     return index, inc, w
#
#
# def sky_model_tau(state, theta, altitude, turbidity, wavelength, distance):
#     """
#     This computes transmittance between a point at 'altitude' and infinity in
#     the direction 'theta' at a wavelength 'wavelength'.
#
#     Parameters
#     ----------
#     state : SkyModelState
#     theta : float
#     altitude : float
#     turbidity : float
#     wavelength : float
#     distance : float
#
#     Returns
#     -------
#     float
#     """
#     assert isinstance(theta, float) and 0 <= theta <= np.pi
#     assert isinstance(altitude, float) and 0 <= altitude <= 150000
#     assert isinstance(turbidity, float) and 0 <= turbidity
#     assert isinstance(wavelength, float) and 0 < wavelength
#     assert isinstance(distance, float) and 0 < distance
#
#     wavelength_norm = (wavelength - state.channel_start) / state.channel_width
#     if wavelength_norm >= state.channels or wavelength_norm < 0:
#         return 0
#
#     wavelength_low = int(wavelength_norm)
#     wavelength_factor: float = 0
#     wavelength_inc: int = 1 if wavelength_low < 10 else 0
#
#     assert isinstance(wavelength_low, int) and 0 <= wavelength_low <= 10
#     assert isinstance(wavelength_inc, int) and 0 <= wavelength_inc <= 1
#     assert isinstance(wavelength_factor, float) and 0 <= wavelength_factor <= 1
#
#     altitude_low, altitude_inc, altitude_factor = sky_model_find_in_array(state.transmission_altitudes, altitude)
#
#     assert isinstance(altitude_low, int) and 0 <= altitude_low <= 21
#     assert isinstance(altitude_inc, int) and 0 <= altitude_inc <= 1
#     assert isinstance(altitude_factor, float) and 0 <= altitude_factor <= 1
#
#     turb_low, turb_inc, turb_w = sky_model_find_in_array(state.transmission_turbidities, turbidity)
#
#     assert isinstance(turb_low, int) and 0 <= turb_low <= 2
#     assert isinstance(turb_inc, int) and 0 <= turb_inc <= 1
#     assert isinstance(turb_w, float) and 0 <= turb_w <= 1
#
#     # Calculate normalised and non-linearly scaled position in the atmosphere
#     a, d = sky_model_to_ad(theta, distance, altitude)
#
#     assert isinstance(a, float) and 0 <= a
#     assert isinstance(d, float) and 0 <= d
#
#     # Evaluate basis at low turbidity
#     trans_low = sky_model_calc_transittance_svd(state, a, d, turb_low,
#                                                 wavelength_low, wavelength_inc, wavelength_factor,
#                                                 altitude_low, altitude_inc, altitude_factor)
#
#     # Evaluate basis at high turbidity
#     trans_high = sky_model_calc_transittance_svd(state, a, d, turb_low + turb_inc,
#                                                  wavelength_low, wavelength_inc, wavelength_factor,
#                                                  altitude_low, altitude_inc, altitude_factor)
#
#     # Return interpolated transmittance values
#     trans = lerp(trans_low, trans_high, turb_w)
#     trans = np.clip(trans, 0, 1)
#
#     return trans * trans
#
#
# def reconstruct_pol(state: SkyModelState, gamma: float, alpha: float, gamma_segment: int, alpha_segment: int,
#                     control_params: np.ndarray):
#     res: float = 0
#     for t in range(state.tensor_components_pol):
#         sun_val_t = eval_pp(gamma, gamma_segment, state.sun_breaks_pol,
#                             control_params[state.sun_offset_pol + t * state.sun_stride_pol])
#         zenith_val_t = eval_pp(alpha, alpha_segment, state.zenith_breaks_pol,
#                                control_params[state.zenith_offset_pol + t * state.zenith_stride_pol])
#         res += sun_val_t * zenith_val_t
#
#     return res
#
#
# def interpolate_elevation_pol(state: SkyModelState, elevation: float, altitude: int, turbidity: int, albedo: int,
#                               wavelength: int, gamma: float, alpha: float, gamma_segment: int, alpha_segment: int):
#     elevation_low: int = int(elevation)
#     factor = elevation - float(elevation_low)
#
#     control_params_low = control_params_single_config(
#         state, state.polarisation_dataset, state.total_coefs_single_config_pol,
#         elevation_low, altitude, turbidity, albedo, wavelength
#     )
#     res_low = reconstruct_pol(state, gamma, alpha, gamma_segment, alpha_segment, control_params_low)
#
#     if factor < 1e-06 or elevation_low >= len(state.elevations) - 1:
#         return res_low
#
#     control_params_high = control_params_single_config(
#         state, state.polarisation_dataset, state.total_coefs_single_config_pol,
#         elevation_low + 1, altitude, turbidity, albedo, wavelength
#     )
#     res_high = reconstruct_pol(state, gamma, alpha, gamma_segment, alpha_segment, control_params_high)
#
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_altitude_pol(state: SkyModelState, elevation: float, altitude: float, turbidity: int, albedo: int,
#                              wavelength: int, gamma: float, alpha: float, gamma_segment: int, alpha_segment: int):
#     altitude_low = int(altitude)
#     factor = altitude - float(altitude_low)
#
#     res_low = interpolate_elevation_pol(
#         state, elevation, altitude_low, turbidity, albedo, wavelength, gamma, alpha, gamma_segment, alpha_segment
#     )
#
#     if factor < 1e-06 or altitude_low >= len(state.altitudes) - 1:
#         return res_low
#
#     res_high = interpolate_elevation_pol(
#         state, elevation, altitude_low + 1, turbidity, albedo, wavelength, gamma, alpha, gamma_segment, alpha_segment
#     )
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_turbidity_pol(state: SkyModelState, elevation: float, altitude: float, turbidity: float, albedo: int,
#                               wavelength: int, gamma: float, alpha: float, gamma_segment: int, alpha_segment: int):
#
#     # Ignore turbidity
#     return interpolate_altitude_pol(
#         state, elevation, altitude, int(turbidity), albedo, wavelength, gamma, alpha, gamma_segment, alpha_segment
#     )
#
#
# def interpolate_albedo_pol(state: SkyModelState, elevation: float, altitude: float, turbidity: float, albedo: float,
#                            wavelength: int, gamma: float, alpha: float, gamma_segment: int, alpha_segment: int):
#     albedo_low = int(albedo)
#     factor = albedo - float(albedo_low)
#
#     res_low = interpolate_turbidity_pol(
#         state, elevation, altitude, turbidity, albedo_low, wavelength, gamma, alpha, gamma_segment, alpha_segment
#     )
#
#     if factor < 1e-06 or albedo_low >= len(state.albedos) - 1:
#         return res_low
#
#     res_high = interpolate_turbidity_pol(
#         state, elevation, altitude, turbidity, albedo_low + 1, wavelength, gamma, alpha, gamma_segment, alpha_segment
#     )
#     return lerp(res_low, res_high, factor)
#
#
# def interpolate_wavelength_pol(state: SkyModelState, elevation: float, altitude: float, turbidity: float, albedo: float,
#                                wavelength: float, gamma: float, alpha: float, gamma_segment: int, alpha_segment: int):
#
#     # Don't interpolate, use the bin it belongs to
#     return interpolate_albedo_pol(
#         state, elevation, altitude, turbidity, albedo, int(wavelength), gamma, alpha, gamma_segment, alpha_segment
#     )
#
#
# def sky_model_polarisation(state, theta, gamma, shadow, zero, elevation, altitude, turbidity, albedo, wavelength):
#     """
#
#     Parameters
#     ----------
#     state : SkyModelState
#     theta : float
#     gamma : float
#     shadow : float
#     zero : float
#     elevation : float
#     altitude : float
#     turbidity : float
#     albedo : float
#     wavelength : float
#
#     Returns
#     -------
#     float
#     """
#
#     # If no polarisation data available
#     if state.tensor_components_pol == 0:
#         return 0
#
#     # Translate parameter values to indices
#
#     turbidity_control = map_parameter(turbidity, state.turbidities)
#     albedo_control = map_parameter(albedo, state.albedos)
#     altitude_control = map_parameter(altitude, state.altitudes)
#     elevation_control = map_parameter(elevation, state.elevations)
#     channel_control = (wavelength - state.channel_start) / state.channel_width
#
#     if channel_control >= state.channels or channel_control < 0:
#         return 0
#
#     gamma_segment = find_segment(gamma, state.sun_breaks_pol)
#     theta_segment = find_segment(theta, state.zenith_breaks_pol)
#
#     return -interpolate_wavelength_pol(
#         state, elevation_control, altitude_control, turbidity_control, albedo_control, channel_control,
#         gamma, theta, gamma_segment, theta_segment
#     )
#
#
# def compute_pp_coefs(breaks, values, coefs, offset):
#     """
#
#     Parameters
#     ----------
#     breaks : list[float]
#     values : list[float]
#     coefs : list[float]
#     offset : int
#
#     Returns
#     -------
#     int
#     """
#     nb_breaks = len(breaks)
#     for i in range(nb_breaks):
#         coefs[offset + 2 * i] = (values[i+1] - values[i]) / (breaks[i+1] - breaks[i])
#         coefs[offset + 2 * i + 1] = values[i]
#
#     return 2 * nb_breaks - 2


