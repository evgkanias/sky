"""
Package that allows computations of the skylight properties by using the Vevoda el al. (2022) model.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias", "Petr Vévoda", "Alexander Wilkie"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from .._static import *
from .io import *
from skylight.exceptions import *

import skylight.geometry as geo

import numpy as np
import warnings

eps = np.finfo(float).eps


class PragueSkyModelManager(object):
    def __init__(self):
        """
        Implementation of the physically-based skylight model presented by [1]_ and [2]_. Improves on previous work
        especially in accuracy of sunset scenarios. Based on reconstruction of radiance from a small dataset
        fitted to a large set of images obtained by brute force atmosphere simulation.

        Provides evaluation of spectral skylight radiance, sun radiance, transmittance and polarisation for observer at
        a specific altitude above ground. The range of configurations depends on supplied dataset. The full
        version models atmosphere of visibility (meteorological range) from 20 km to 131.8 km for sun elevations
        from -4.2 degrees to 90 degrees, observer altitudes from 0 km to 15 km and ground albedo from 0 to 1, and
        provides results for wavelengths from 280 nm to 2480 nm.

        Usage
        -----
        1. Create PragueSkyModel object and call its initialise method with a path to the dataset file.
        2. The model is parametrized by several values that are gathered in a Parameters structure.
        You can either fill this structure manually (in that case see its description) or just call
        compute_parameters, which will compute it for you based on a few basic parameters.
        3. Use the Parameters structure when calling sky_radiance, sun_radiance, transmittance, or polarisation
        methods to obtain the respective quantities.

        Examples
        --------
        >>> my_sky = PragueSkyModelManager()
        >>> print(my_sky.is_initialised)
        False

        Raises
        ------
        DatasetNotFoundException
            if the specified dataset file could not be found
        DatasetReadException
            if an error occurred while reading the dataset file
        NoPolarisationException
            if the polarisation method is called but the model does not contain polarisation data
        NotInitializedException
            if the model is used without calling the initialize method first

        Note
        ----
        The entire model is written in a single class and does not depend on anything except of numpy. It defines a
        simple numpy.ndarray[float] to simplify working with points and directions and expects using this class when
        passing viewing point and direction to the compute_parameters method.

        See Also
        --------
        Parameters
        sky_radiance
        sun_radiance
        polarisation
        transmittance

        References
        ----------
        .. [1] Wilkie, A. et al. A fitted radiance and attenuation model for realistic atmospheres.
           Acm T Graphic 40, 1–14 (2021). (https://cgg.mff.cuni.cz/publications/skymodel-2021/)

        .. [2] Vévoda, P., Bashford-Rogers, T., Kolárová, M. & Wilkie, A. A Wide Spectral Range Sky Radiance Model.
           Pacific Graphics 41, (2022).
        """

        self.__nb_channels: int = 0
        """
        Number of channels (wavelengths).
        """
        self.__channel_start: float = 0
        """
        The first available wavelength.
        """
        self.__channel_width: float = 0
        """
        The range that the wavelength covers.
        """

        self.__initialised: bool = False
        """
        Weather the model has been initialised.
        """
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
        self.__data_rad: {np.ndarray, None} = None

        # Polarisation metadata
        self.__metadata_pol: {Metadata, None} = None

        # Polarisation data
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

    def initialise(self, filename, single_visibility=0.0):
        """
        Prepares the model and loads the given dataset file into memory.
        If a positive visibility value is passed, only a portion of the dataset needed for evaluation of that
        particular visibility is loaded (two nearest visibilities are loaded if the value is included in the
        dataset or one nearest if not). Otherwise, the entire dataset is loaded.

        Examples
        --------
        >>> import os
        >>>
        >>> my_sky = PragueSkyModelManager()
        >>> print(my_sky.is_initialised)
        False
        >>> if not os.path.exists('PragueSkyModelDatasetGroundInfra.dat'):  # doctest: +SKIP
        ...     import urllib.request  # doctest: +SKIP
        ...     with urllib.request.urlopen('https://drive.google.com/u/0/uc?id=1ZOizQCN6tH39JEwyX8KvAj7WEdX-EqJl&export=download&confirm=t&uuid=feb46385-9cae-4e37-801d-d12a363bcbe0&at=ALAFpqxYftsd-qRDTKfAWGXOGNe1:1668004693925', timeout=100) as f:  # doctest: +SKIP
        ...         with open('PragueSkyModelDatasetGroundInfra.dat', 'wb') as fw:  # doctest: +SKIP
        ...             fw.write(f.read())  # doctest: +SKIP
        >>> my_sky.initialise('PragueSkyModelDatasetGroundInfra.dat')  # doctest: +SKIP
        >>> print(my_sky.is_initialised)  # doctest: +SKIP
        True

        Parameters
        ----------
        filename : str
            the filename that contains the dataset
        single_visibility : float
            specifies the batch of visibilities to load

        Raises
        ------
        DatasetNotFoundException
            if the specified dataset file could not be found
        DatasetReadException
            if an error occurred while reading the dataset file

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

        Examples
        --------
        >>> from skylight.prague import Parameters  # doctest: +SKIP
        >>>
        >>> my_sky = PragueSkyModelManager()
        >>> my_sky.initialise('PragueSkyModelDatasetGroundInfra.dat')  # doctest: +SKIP
        >>>
        >>> parameters = Parameters(
        ...     theta=np.full(1, np.pi/4), gamma=np.full(1, np.pi/4), shadow=np.full(1, np.pi/2),
        ...     zero=np.full(1, np.full(1, np.pi/4)), elevation=np.pi/4, altitude=0., visibility=50, albedo=0.5)  # doctest: +SKIP
        >>> print(my_sky.sky_radiance(parameters, wavelength=np.full(1, 350)))  # doctest: +SKIP
        [[0.09110355]]

        Parameters
        ----------
        params : Parameters
            a structure holding all parameters necessary for querying the model
        wavelength : np.ndarray[float]
            an array specifying the wavelengths to compute the radiance for

        Returns
        -------
        np.ndarray[float]
            the sky radiance for the given parameters and wavelengths

        Raises
        ------
        NotInitialisedException
            when the model is not initialised (calling the initialise method)
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__evaluate_model(params, wavelength, self.data_rad, self.metadata_rad)

    def sun_radiance(self, params, wavelength):
        """
        Computes sun radiance only (without radiance inscattered from the sky) for given parameters and
        wavelength (full dataset supports wavelengths from 280 nm to 2480 nm).

        Checks whether the parameters correspond to view direction hitting the sun and returns 0 if not.

        Examples
        --------
        >>> from skylight.prague import Parameters  # doctest: +SKIP
        >>>
        >>> my_sky = PragueSkyModelManager()  # doctest: +SKIP
        >>> my_sky.initialise('PragueSkyModelDatasetGroundInfra.dat')  # doctest: +SKIP
        >>>
        >>> parameters = Parameters(
        ...     theta=np.full(1, np.pi/4), gamma=np.full(1, np.pi/4), shadow=np.full(1, np.pi/2),
        ...     zero=np.full(1, np.full(1, np.pi/4)), elevation=np.pi/4, altitude=0., visibility=50, albedo=0.5)  # doctest: +SKIP
        >>> print(my_sky.sun_radiance(parameters, wavelength=np.full(1, 350)))  # doctest: +SKIP
        [[0.]]

        Parameters
        ----------
        params : Parameters
            a structure holding all parameters necessary for querying the model
        wavelength : np.ndarray[float]
            an array specifying the wavelengths to compute the radiance for

        Returns
        -------
        np.ndarray[float]
            the sun radiance for the given parameters and wavelengths

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        sun_radiance = np.zeros((len(wavelength), len(params.gamma)), dtype='float64')

        # Ignore wavelengths outside the dataset range.
        wl_in_range = np.all([wavelength >= SUN_RAD_START, wavelength < SUN_RAD_END], axis=0)

        # Return zero for rays not hitting the sun.
        ray_in_radius = params.gamma <= SUN_RADIUS

        valid = wl_in_range[:, None] & ray_in_radius[None, :]

        # Compute index into the sun radiance table.
        idx = (wavelength - SUN_RAD_START) / SUN_RAD_STEP
        assert np.all([0 <= idx[wl_in_range], idx[wl_in_range] < len(SUN_RAD_TABLE) - 1])

        # copy the indices for the different gammas
        idx = np.repeat(idx[:, None], len(ray_in_radius), axis=1)

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

        Examples
        --------
        >>> from skylight.prague import Parameters  # doctest: +SKIP
        >>>
        >>> my_sky = PragueSkyModelManager()  # doctest: +SKIP
        >>> my_sky.initialise('PragueSkyModelDatasetGroundInfra.dat')  # doctest: +SKIP
        >>>
        >>> parameters = Parameters(
        ...     theta=np.full(1, np.pi/4), gamma=np.full(1, np.pi/4), shadow=np.full(1, np.pi/2),
        ...     zero=np.full(1, np.full(1, np.pi/4)), elevation=np.pi/4, altitude=0., visibility=50, albedo=0.5)  # doctest: +SKIP
        >>> print(my_sky.polarisation(parameters, wavelength=np.full(1, 350)))  # doctest: +SKIP
        [[0.13283859]]

        Parameters
        ----------
        params : Parameters
            a structure holding all parameters necessary for querying the model
        wavelength : np.ndarray[float]
            an array specifying the wavelengths to compute the radiance for

        Returns
        -------
        np.ndarray[float]
            the degree of polarisation for the given parameters and wavelengths

        Raises
        ------
        NoPolarisationException
            if the polarisation method is called but the model does not contain polarisation data
        NotInitializedException
            if called without initializing the model first
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
        to 2480 nm) and the distance (any positive number, use np.finfo(float).max for infinity).

        Examples
        --------
        >>> from skylight.prague import Parameters  # doctest: +SKIP
        >>>
        >>> my_sky = PragueSkyModelManager()  # doctest: +SKIP
        >>> my_sky.initialise('PragueSkyModelDatasetGroundInfra.dat')  # doctest: +SKIP
        >>>
        >>> parameters = Parameters(
        ...     theta=np.full(1, np.pi/4), gamma=np.full(1, np.pi/4), shadow=np.full(1, np.pi/2),
        ...     zero=np.full(1, np.full(1, np.pi/4)), elevation=np.pi/4, altitude=0., visibility=50, albedo=0.5)
        >>> print(my_sky.transmittance(parameters, wavelength=np.full(1, 350), distance=np.finfo(float).max))  # doctest: +SKIP
        0.2613840873796399
        >>> # this should be array([[0.26138409]])

        Parameters
        ----------
        params : Parameters
            a structure holding all parameters necessary for querying the model
        wavelength : np.ndarray[float]
            an array specifying the wavelengths to compute the radiance for
        distance : float
            distance in meters of the viewing point from the point of interest

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.

        Returns
        -------
        np.ndarray[float]
            transmittance between view point and a point certain distance away from it along view direction
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

        Examples
        --------
        >>> PragueSkyModelManager.compute_parameters(
        ...     viewpoint=np.array([0, 0, 0]), view_direction=np.array([[0, 2/np.sqrt(2), 2/np.sqrt(2)]]),
        ...     ground_level_solar_elevation_at_origin=np.pi/4, ground_level_solar_azimuth_at_origin= np.pi,
        ...     visibility=50, albedo=0.5)  # doctest: +ELLIPSIS
        Parameters(theta=array([0.78539...]), gamma=array([1.04719...]), shadow=array([1.04558...]), zero=array([0.78260...]), elevation=0.78539..., altitude=50.0, visibility=50, albedo=0.5)
        >>> PragueSkyModelManager.compute_parameters(
        ...     viewpoint=np.array([0, 0, 0]), view_direction=np.array([[0, 1, 0]]),
        ...     ground_level_solar_elevation_at_origin=np.pi/3, ground_level_solar_azimuth_at_origin= np.pi,
        ...     visibility=50, albedo=0.5)  # doctest: +ELLIPSIS
        Parameters(theta=array([1.57079...]), gamma=array([1.57079...]), shadow=array([1.56881...]), zero=array([1.56683...]), elevation=1.04719..., altitude=50.0, visibility=50, albedo=0.5)

        Parameters
        ----------
        viewpoint : np.ndarray[float]
            a vector determining the view point on Earth
        view_direction : np.ndarray[float]
            vectors determining the view direction
        ground_level_solar_elevation_at_origin : float
            ground level solar elevation at origin (in rad)
        ground_level_solar_azimuth_at_origin : float
            ground level solar azimuth at origin (in rad)
        visibility : float, np.ndarray[float]
            ground level visibility in kilometers
        albedo : float, np.ndarray[float]
            ground level albedo

        Returns
        -------
        Parameters
            a structure holding all parameters necessary for querying the model
        """

        assert viewpoint[2] >= 0
        assert np.all(np.linalg.norm(view_direction, axis=-1) > 0)
        assert np.all(visibility >= 0)
        assert np.all([0 <= albedo, albedo <= 1])

        # Shift viewpoint about safety altitude up
        centre_of_earth = np.array([0, 0, -PLANET_RADIUS], dtype='float64')
        to_viewpoint = viewpoint - centre_of_earth
        to_viewpoint_n = to_viewpoint / np.linalg.norm(to_viewpoint)

        distance_to_view = np.linalg.norm(to_viewpoint) + SAFETY_ALTITUDE
        to_shifted_viewpoint = to_viewpoint_n * distance_to_view
        shifted_viewpoint = centre_of_earth + to_shifted_viewpoint

        view_direction_n = (view_direction.T / np.maximum(np.linalg.norm(view_direction, axis=-1), eps)).T

        # Compute altitude of viewpoint
        altitude = np.maximum(distance_to_view - PLANET_RADIUS, 0)

        # Direction to sun
        direction_to_sun_n = geo.sph2xyz(ground_level_solar_elevation_at_origin, ground_level_solar_azimuth_at_origin)

        # Solar elevation at viewpoint
        # (more precisely, solar elevation at the point on the ground directly below viewpoint)
        elevation = 0.5 * np.pi - geo.angle_between(to_viewpoint_n, direction_to_sun_n)

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
        gamma = geo.angle_between(view_direction_n, direction_to_sun_n)

        # Shadow angle - requires correction
        effective_elevation = ground_level_solar_elevation_at_origin  # rad
        effective_azimuth = ground_level_solar_azimuth_at_origin  # rad
        shadow_angle = effective_elevation + np.pi * 0.5  # rad

        shadow_direction_n = geo.sph2xyz(shadow_angle, effective_azimuth)
        shadow = geo.angle_between(correct_view_n, shadow_direction_n)

        # Zenith angle (theta) - corrected version stored in otherwise unused zero angle
        zero = geo.angle_between(correct_view_n, to_viewpoint_n)

        # Zenith angle (theta) - uncorrected version goes outside
        theta = geo.angle_between(view_direction_n, to_viewpoint_n)

        return Parameters(
            theta=theta, gamma=gamma, shadow=shadow, zero=zero, elevation=elevation, altitude=altitude,
            visibility=visibility, albedo=albedo
        )

    @property
    def available_data(self):
        """
        Gets parameter ranges available in currently loaded dataset.

        Returns
        -------
        AvailableData

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
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
        """
        True when the model has been initialised, False otherwise.

        Returns
        -------
        bool
        """
        return self.__initialised

    @property
    def nb_channels(self):
        """
        The number of channels (wavelengths) in the dataset.

        Returns
        -------
        int

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__nb_channels

    @property
    def channel_start(self):
        """
        The first channel (wavelength) available in the dataset.

        Returns
        -------
        float

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__channel_start

    @property
    def channel_width(self):
        """
        The range of each of the channels (wavelengths) in the dataset.

        Returns
        -------
        float

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__channel_width

    @property
    def total_configs(self):
        """
        The total number of configurations in the dataset.

        Returns
        -------
        int

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__total_configs

    @property
    def visibilities_rad(self):
        """
        The different visibility bins in the dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__visibilities_rad

    @property
    def albedos_rad(self):
        """
        The different albedo bins in the dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__albedos_rad

    @property
    def altitudes_rad(self):
        """
        The different altitude bins in the dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__altitudes_rad

    @property
    def elevations_rad(self):
        """
        The different elevation bins in the dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__elevations_rad

    @property
    def metadata_rad(self):
        """
        Metadata of the radiance dataset.

        Returns
        -------
        Metadata

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__metadata_rad

    @property
    def data_rad(self):
        """
        The radiance dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_rad

    @property
    def d_dim(self):
        """
        The number of dimensions of the distances in the transmittance data.

        Returns
        -------
        int

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__d_dim

    @property
    def a_dim(self):
        """
        The number of dimensions of the altitudes in the transmittance data.

        Returns
        -------
        int

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__a_dim

    @property
    def rank_trans(self):
        """
        The number of ranks available in the transmittance data.

        Returns
        -------
        int

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__rank_trans

    @property
    def altitudes_trans(self):
        """
        The different altitude bins in the transmittance dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__altitudes_trans

    @property
    def visibilities_trans(self):
        """
        The different visibility bins in the transmittance dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__visibilities_trans

    @property
    def data_trans_u(self):
        """
        The U component of the transmittance data.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_trans_u

    @property
    def data_trans_v(self):
        """
        The V component of the transmittance data.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__data_trans_v

    @property
    def metadata_pol(self):
        """
        Metadata of the polarisation dataset.

        Returns
        -------
        Metadata

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
        if not self.is_initialised:
            raise NotInitialisedException()

        return self.__metadata_pol

    @property
    def data_pol(self):
        """
        The polarisation dataset.

        Returns
        -------
        np.ndarray[float]

        Raises
        ------
        NotInitializedException
            if called without initializing the model first.
        """
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
        Evaluates the model. Used for computing skylight radiance and polarisation.

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
        skylight radiance and polarisation.

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
        transmittance = np.zeros((channel_index.size, 4, trans_params.altitude.index.size), dtype='float64')

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

        Examples
        --------
        >>> PragueSkyModelManager.get_interpolation_parameter(query_val=5.6, breaks=np.arange(2, 10))  # doctest: +ELLIPSIS
        InterpolationParameter(factor=0.59999..., index=3)
        >>> PragueSkyModelManager.get_interpolation_parameter(query_val=1.1, breaks=np.arange(2, 10))  # doctest: +ELLIPSIS
        InterpolationParameter(factor=0.0, index=0)
        >>> PragueSkyModelManager.get_interpolation_parameter(query_val=2.1, breaks=np.arange(2, 10))  # doctest: +ELLIPSIS
        InterpolationParameter(factor=0.10000..., index=0)
        >>> PragueSkyModelManager.get_interpolation_parameter(query_val=8.1, breaks=np.arange(2, 10))  # doctest: +ELLIPSIS
        InterpolationParameter(factor=0.09999..., index=6)

        Parameters
        ----------
        query_val : float, np.ndarray[float]
            the query value
        breaks : np.ndarray[float]
            the break values for interpolation

        Returns
        -------
        InterpolationParameter
            the lowest indices in the breaks array whose values are close to the query values, and a factor that shows
            how bigger is the query value (towards the next index)
        """

        # Clamp the value to the valid range
        clamped = np.float64(np.clip(query_val, breaks[0], breaks[-1]))

        # Get the nearest greater parameter value
        next_greater = np.searchsorted(breaks, clamped, side='right')

        # Compute the index and float factor
        factor = np.zeros(next_greater.shape, dtype='float64')
        factor_valid = np.all([0 < next_greater, next_greater <= len(breaks) - 1], axis=0)
        if factor.size == 1 and factor_valid:
            factor = (float(clamped - breaks[next_greater - 1]) /
                      float(breaks[next_greater] - breaks[next_greater - 1]))
        else:
            factor[factor_valid] = (np.float64(clamped[factor_valid] - breaks[next_greater[factor_valid] - 1]) /
                                    np.float64(breaks[next_greater[factor_valid]] -
                                               breaks[next_greater[factor_valid] - 1]))

        index = next_greater - 1
        assert np.all([0 <= index, index < len(breaks),
                       np.any([index < len(breaks) - 1, factor == 0], axis=0)])
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

        Examples
        --------
        >>> PragueSkyModelManager.get_interpolation_parameter_trans(np.full(1, 0.2), 2, 4)  # doctest: +ELLIPSIS
        InterpolationParameter(factor=array([0.0256]), index=array([0]...))
        >>> PragueSkyModelManager.get_interpolation_parameter_trans(np.full(1, 0.2), 2, 3)  # doctest: +ELLIPSIS
        InterpolationParameter(factor=array([0.064]), index=array([0]...))
        >>> PragueSkyModelManager.get_interpolation_parameter_trans(np.full(1, 0.1), 2, 3)  # doctest: +ELLIPSIS
        InterpolationParameter(factor=array([0.008]), index=array([0]...))
        >>> PragueSkyModelManager.get_interpolation_parameter_trans(np.full(1, 0.5), 2, 3)  # doctest: +ELLIPSIS
        InterpolationParameter(factor=array([0.]), index=array([1]...))

        Parameters
        ----------
        value : np.ndarray[float]
            the altitude or distance query values
        param_count : int
            the number of parameters
        power : int
            power for the non-linear interpolation

        Returns
        -------
        InterpolationParameter
            the lowest indices in the number of parameters whose values are close to the query values, and
            a factor that shows how bigger is the query value (towards the next index)
        """
        index = np.minimum(np.int32(value * param_count), param_count - 1)
        factor = np.zeros(value.shape, dtype='float64')

        in_range = index < param_count - 1
        factor[in_range] = np.clip(nonlerp(np.float64(index[in_range]) / param_count,
                                           np.float64(index[in_range] + 1) / param_count,
                                           value[in_range], power), 0, 1)

        return InterpolationParameter(factor, index)

    @staticmethod
    def reconstruct(radiance_parameters, control_parameters, metadata):
        """
        Reconstructs sky radiance or polarisation from the given control parameters by inverse tensor decomposition.

        Examples
        --------
        >>> from skylight.prague import AngleParameters, Metadata
        >>> rad_params = AngleParameters(
        ...     gamma=PragueSkyModelManager.get_interpolation_parameter(np.full(1, np.pi/3), np.linspace(0, np.pi, 19, endpoint=True)),
        ...     alpha=PragueSkyModelManager.get_interpolation_parameter(np.full(1, np.pi/6), np.linspace(0, np.pi/2, 10, endpoint=True)),
        ...     zero=PragueSkyModelManager.get_interpolation_parameter(np.full(1, np.pi/6), np.linspace(0, np.pi/2, 10, endpoint=True))
        ... )
        >>> con_params = np.linspace(1, 2, 25).reshape((1, 1, -1))
        >>> meta = Metadata(
        ...     rank=1, sun_offset=0, sun_stride=20, sun_breaks=np.linspace(-np.pi/36, 2*np.pi+np.pi/36, 10),
        ...     zenith_offset=10, zenith_stride=20, zenith_breaks=np.linspace(-np.pi/36, 2*np.pi+np.pi/36, 10),
        ...     emph_offset=20, emph_breaks=np.linspace(0, 2*np.pi, 5),
        ...     total_coefs_single_config=25, total_coefs_all_config=25)
        >>> PragueSkyModelManager.reconstruct(rad_params, con_params, meta)  # doctest: +ELLIPSIS
        3.77387...

        Parameters
        ----------
        radiance_parameters : AngleParameters
            the gamma, alpha, and zero interpolation parameters
        control_parameters : np.ndarray[float]
            the parameters that control the interpolation
        metadata : Metadata
            the metadata of the dataset

        Returns
        -------
        np.ndarray[float]
            an array with the radiance or polarisation on the given parameters
        """

        result = 0.0
        for r in range(metadata.rank):
            # Restore the right value in the 'sun' vector
            i_sun = r * metadata.sun_stride + metadata.sun_offset + radiance_parameters.gamma.index
            sun_param = eval_pl(
                np.array([control_parameters[0, ..., i_sun].T, control_parameters[0, ..., i_sun + 1].T]),
                radiance_parameters.gamma.factor)

            # Restore the right value in the 'zenith' vector
            i_zen = r * metadata.zenith_stride + metadata.zenith_offset + radiance_parameters.alpha.index
            zenith_param = eval_pl(
                np.array([control_parameters[0, ..., i_zen].T, control_parameters[0, ..., i_zen + 1].T]),
                radiance_parameters.alpha.factor)

            # Accumulate their "outer" product
            result += sun_param * zenith_param

        # De-emphasize (for radiance only)
        if metadata.emph_breaks is not None and len(metadata.emph_breaks) > 0:
            i_emp = metadata.emph_offset + radiance_parameters.zero.index
            emph_param = eval_pl(
                np.array([control_parameters[0, ..., i_emp].T, control_parameters[0, ..., i_emp + 1].T]),
                radiance_parameters.zero.factor)
            result *= emph_param
            result = np.maximum(result, 0)

        return result

    @staticmethod
    def interpolate(t_offset, t_level, angle_parameters, control_parameters, metadata):
        """
        Recursive function controlling interpolation of reconstructed radiance between two neighboring visibility,
        albedo, altitude and elevation values.

        Examples
        --------
        >>> from skylight.prague import AngleParameters, Metadata
        >>> rad_params = AngleParameters(
        ...     gamma=PragueSkyModelManager.get_interpolation_parameter(np.full(1, np.pi/3), np.linspace(0, np.pi, 19, endpoint=True)),
        ...     alpha=PragueSkyModelManager.get_interpolation_parameter(np.full(1, np.pi/6), np.linspace(0, np.pi/2, 10, endpoint=True)),
        ...     zero=PragueSkyModelManager.get_interpolation_parameter(np.full(1, np.pi/6), np.linspace(0, np.pi/2, 10, endpoint=True))
        ... )
        >>> meta = Metadata(
        ...     rank=1, sun_offset=0, sun_stride=20, sun_breaks=np.linspace(-np.pi/36, 2*np.pi+np.pi/36, 10),
        ...     zenith_offset=10, zenith_stride=20, zenith_breaks=np.linspace(-np.pi/36, 2*np.pi+np.pi/36, 10),
        ...     emph_offset=20, emph_breaks=np.linspace(0, 2*np.pi, 5),
        ...     total_coefs_single_config=25, total_coefs_all_config=25)
        >>> con_params = ControlParameters(
        ...     coefficients=np.linspace(1, 2, 400).reshape((16, 1, -1)), interpolation_factor=np.full(4, 0.5))
        >>> PragueSkyModelManager.interpolate(0, 0, rad_params, con_params, meta)  # doctest: +ELLIPSIS
        3.78492...
        >>> con_params = ControlParameters(
        ...     coefficients=np.linspace(1, 2, 400).reshape((16, 1, -1)), interpolation_factor=np.full(4, 0.1))
        >>> PragueSkyModelManager.interpolate(0, 0, rad_params, con_params, meta)  # doctest: +ELLIPSIS
        1.55091...

        Parameters
        ----------
        t_offset : int
            offset of the coefficients
        t_level : int
            the interpolation level in [1-4]
        angle_parameters : AngleParameters
            the gamma, alpha, and zero interpolation parameters
        control_parameters : ControlParameters
            the parameters that control the interpolation
        metadata : Metadata
            the metadata of the dataset

        Returns
        -------
        float
            an array with the radiance or polarisation on the given parameters after the interpolation
        """

        # Starts at level 0 and recursively goes down to level 4 while computing offset to the control
        # parameters array. There it reconstructs radiance. When returning from recursion interpolates
        # according to elevation, altitude, albedo and visibility at level 3, 2, 1 and 0, respectively.
        if t_level == 4:
            return PragueSkyModelManager.reconstruct(angle_parameters, control_parameters.coefficients[t_offset:], metadata)
        else:
            # Compute the first value
            result_low = PragueSkyModelManager.interpolate(t_offset, t_level + 1,
                                                           angle_parameters, control_parameters, metadata)

            # Skip the second value if not useful or not available.
            if control_parameters.interpolation_factor[t_level] < 1e-06:
                return result_low

            # Compute the second value
            result_high = PragueSkyModelManager.interpolate(t_offset + (1 << (3 - t_level)), t_level + 1,
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
    coefs : np.ndarray[float]
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
    d1 = (-qb + discrim) / np.maximum(2.0 * qa, eps)
    d2 = (-qb - discrim) / np.maximum(2.0 * qa, eps)

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
