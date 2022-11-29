"""
Package that allows computations of the skylight properties.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from .__helpers import __root__, eps

from . import prague
from .observer import Observer, get_seville_observer
from .ephemeris import Sun

import sky.geometry as geo

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from dataclasses import dataclass

import numpy as np

T_L = np.array([[ 0.1787, -1.4630],
                [-0.3554,  0.4275],
                [-0.0227,  5.3251],
                [ 0.1206, -2.5771],
                [-0.0670,  0.3703]])
"""Transformation matrix of turbidity to luminance coefficients"""


@dataclass
class SkyInstance:

    sun_direction: np.ndarray
    view_direction: np.ndarray
    sky_radiance: np.ndarray
    degree_of_polarisation: np.ndarray
    angle_of_polarisation: np.ndarray
    albedo: np.ndarray
    wavelengths: np.ndarray


class SkyBase(object):
    def __init__(self, theta_s=0., phi_s=0., degrees=False, name="sky"):
        self.__max_radiance = 1.
        theta_s = np.deg2rad(theta_s) if degrees else theta_s
        self.__theta_s = theta_s
        self.__phi_s = np.deg2rad(phi_s) if degrees else phi_s

        self._is_generated = False
        self.__name = name

    def __call__(self, ori, wavelengths=None, albedo=None):
        """
        Generates the skylight properties for the given orientations and spectral influences.

        Parameters
        ----------
        ori: R, np.ndarray[float], optional
            orientation of the interesting elements. Default is None
        irgbu: np.ndarray[float], optional
            the spectral influence of the observer
        noise: float, optional
            the noise level (sigma)
        eta: np.ndarray[float], optional
            :param eta: array of noise level in each point of interest
        rng
            the random generator

        Returns
        -------
        SkyInstance
            the luminance, degree and angle of polarisation
        """

        if wavelengths is None:
            wavelengths = np.array([prague.SPECTRAL_RESPONSE_START + 240])

        # set default arguments
        xyz = geo.get_coordinates(ori)

        # calculate light properties
        y = np.full(xyz.shape[0], self.__max_radiance * np.power(np.sin(self.theta_s), 4))  # radiance
        p = np.full(xyz.shape[0], 0.2)  # degree of polarisation
        a = self.compute_angle_of_polarisation(ori, self.phi_s, self.theta_s)

        # create cloud disturbance
        if albedo is None:
            albedo = np.ones_like(y)
        y *= albedo

        # if wavelengths is not None:
        #     y = spectrum_influence(y, wavelengths).sum(axis=1)

        self._is_generated = True

        sun_xyz = geo.sph2xyz(self.theta_s, self.phi_s)

        return SkyInstance(
            sun_direction=sun_xyz,
            view_direction=xyz,
            sky_radiance=y,
            degree_of_polarisation=p,
            angle_of_polarisation=a,
            albedo=albedo,
            wavelengths=wavelengths
        )

    @property
    def max_radiance(self):
        return self.__max_radiance

    @max_radiance.setter
    def max_radiance(self, value):
        self.__max_radiance = value

    @property
    def theta_s(self):
        return self.__theta_s

    @theta_s.setter
    def theta_s(self, value):
        self.__theta_s = value
        self._is_generated = False

    @property
    def phi_s(self):
        return self.__phi_s

    @phi_s.setter
    def phi_s(self, value):
        self.__phi_s = value
        self._is_generated = False

    @staticmethod
    def compute_angle_of_polarisation(ori, phi_s, theta_s):
        v_s = geo.sph2xyz(theta_s, phi_s)
        v_p = geo.ori2xyz(ori)
        c = np.cross(v_s, v_p)

        a = np.arctan2(c[:, 1], c[:, 0])
        return (a + np.pi / 2) % np.pi - np.pi / 2


class UniformSky(SkyBase):
    def __init__(self, theta_s=0., phi_s=0., degrees=False, radiance=1., name="uniform-sky"):
        super(UniformSky, self).__init__(theta_s=theta_s, phi_s=phi_s, degrees=degrees, name=name)
        self.max_radiance = radiance


class AnalyticalSky(SkyBase):

    def __init__(self, theta_s=0., phi_s=0., degrees=False, name="sky"):
        """
        The Sky environment class. This environment class provides skylight cues.

        Parameters
        ----------
        theta_s: float, optional
            sun elevation (distance from horizon). Default is 0
        phi_s: float, optional
            sun azimuth (clockwise from North). Default is 0
        degrees: bool, optional
            True if the angles are given in degrees, False otherwise. Default is False
        name: str, optional
            a name for the sky instance. Default is 'sky'
        """
        SkyBase.__init__(self, theta_s=theta_s, phi_s=phi_s, degrees=degrees, name=name)
        self.__a, self.__b, self.__c, self.__d, self.__e = 0., 0., 0., 0., 0.
        self.__tau_L = 2.
        self._update_luminance_coefficients(self.__tau_L)
        self.__c1 = .6
        self.__c2 = 4.

    def __call__(self, ori, wavelengths=None, albedo=None):
        """
        Generates the skylight properties for the given orientations and spectral influences.

        Parameters
        ----------
        ori: R
            orientation of the interesting elements. Default is None
        wavelengths: np.ndarray[float], optional
            the spectral influence of the observer
        albedo: np.ndarray[float], optional
            :param eta: array of noise level in each point of interest

        Returns
        -------
        SkyInstance
            the luminance, degree and angle of polarisation
        """

        # set default arguments
        xyz = geo.get_coordinates(ori)
        theta = geo.xyz2elevation(xyz)
        # phi = geo.xyz2azimuth(xyz)

        sun_xyz = geo.sph2xyz(self.theta_s, self.phi_s)

        # SKY INTEGRATION

        # distance of the element from the sun position
        gamma = geo.angle_between(xyz, sun_xyz)

        # Intensity
        i_prez = self.L(gamma, np.pi/2 - theta)
        i_00 = self.L(0., np.pi/2 - self.theta_s)  # the luminance (Cd/m^2) at the zenith point
        i_90 = self.L(np.pi / 2, np.absolute(self.theta_s))  # the luminance (Cd/m^2) on the horizon
        # influence of env intensity
        i = (1. / (i_prez + eps) - 1. / (i_00 + eps)) * i_00 * i_90 / (i_00 - i_90 + eps)
        y = np.maximum(self.Y_z * i_prez / (i_00 + eps), 0.) / 25  # Illumination

        # Degree of Polarisation
        lp = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
        # p = np.clip(2. / np.pi * self.M_p * lp * (theta * np.cos(theta) + (np.pi / 2 - theta) * i), 0., 1.)
        p = np.clip(2. / np.pi * self.M_p * lp * ((np.pi / 2 - theta) * np.sin(theta) + theta * i), 0., 1.)
        p = 0.7 * np.sqrt(p)

        # Angle of polarisation
        a = self.compute_angle_of_polarisation(ori, self.phi_s, self.theta_s)

        # create cloud disturbance
        if albedo is None:
            albedo = np.ones_like(y)
        y *= albedo
        p *= albedo  # destroy the polarisation pattern
        a[np.isclose(albedo, 0)] = np.nan

        y[gamma < np.pi/180] = 100

        self._is_generated = True

        if wavelengths is not None:
            y = spectrum_influence(y, wavelengths).sum(axis=1)

        return SkyInstance(
            sun_direction=sun_xyz,
            view_direction=xyz,
            sky_radiance=y,
            degree_of_polarisation=p,
            angle_of_polarisation=a,
            wavelengths=wavelengths,
            albedo=albedo
        )

    def L(self, chi, z):
        """
        Prez. et. al. Luminance function.
        Combines the scattering indicatrix and luminance gradation functions to compute the total
        luminance observed at the given env element(s).

        Parameters
        ----------
        chi: np.ndarray[float] | float
            angular distance between the observed element and the sun location -- [0, pi]
        z: np.ndarray[float]
            angular distance between the observed element and the zenith point -- [0, pi/2]

        Returns
        -------
        np.ndarray[float]
            the total observed luminance (Cd/m^2) at the given element(s)
        """
        z = np.array(z)
        i = z < (np.pi / 2)
        f = np.zeros_like(z)
        if z.ndim > 0:
            f[i] = (1. + self.A * np.exp(self.B / (np.cos(z[i]) + eps)))
        elif i:
            f = (1. + self.A * np.exp(self.B / (np.cos(z) + eps)))
        phi = (1. + self.C * np.exp(self.D * chi) + self.E * np.square(np.cos(chi)))
        return f * phi

    @property
    def A(self):
        """
        Darkening or brightening of the horizon

        Returns
        -------
        float
        """
        return self.__a

    @A.setter
    def A(self, value):
        self.__a = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self._is_generated = False

    @property
    def B(self):
        """
        Luminance gradient near the horizon

        Returns
        -------
        float
        """
        return self.__b

    @B.setter
    def B(self, value):
        self.__b = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self._is_generated = False

    @property
    def C(self):
        """
        Relative intensity of the circumsolar region

        Returns
        -------
        float
        """
        return self.__c

    @C.setter
    def C(self, value):
        self.__c = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self._is_generated = False

    @property
    def D(self):
        """
        Width of the circumsolar region

        Returns
        -------
        float
        """
        return self.__d

    @D.setter
    def D(self, value):
        self.__d = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self._is_generated = False

    @property
    def E(self):
        """
        Relative backscattered light

        Returns
        -------
        float
        """
        return self.__e

    @E.setter
    def E(self, value):
        self.__e = value
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)
        self._is_generated = False

    @property
    def c1(self):
        """
        1st coefficient of the maximum degree of polarisation

        Returns
        -------
        float
        """
        return self.__c1

    @property
    def c2(self):
        """
        2nd coefficient of the maximum degree of polarisation

        Returns
        -------
        float
        """
        return self.__c2

    @property
    def tau_L(self):
        """
        The atmospheric turbidity

        Returns
        -------
        float
        """
        return self.__tau_L

    @tau_L.setter
    def tau_L(self, value):
        assert value >= 1., "Turbidity must be greater or eaqual to 1."
        self._is_generated = self.__tau_L == value and self._is_generated
        self._update_luminance_coefficients(value)

    @property
    def Y_z(self):
        """
        The zenith luminance (K cd/m^2)

        Returns
        -------
        float
        """
        chi = (4. / 9. - self.tau_L / 120.) * (np.pi - 2 * (np.pi/2 - self.theta_s))
        return (4.0453 * self.tau_L - 4.9710) * np.tan(chi) - 0.2155 * self.tau_L + 2.4192

    @property
    def M_p(self):
        """
        Maximum degree of polarisation

        Returns
        -------
        float
        """
        return np.exp(-(self.tau_L - self.c1) / (self.c2 + eps))

    def _update_luminance_coefficients(self, tau_L):
        """
        Updates the luminance coefficients given the atmospheric turbidity.

        Parameters
        ----------
        tau_L: float
            the atmospheric turbidity
        """
        self.__a, self.__b, self.__c, self.__d, self.__e = T_L.dot(np.array([tau_L, 1.]))
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

    def _update_turbidity(self, a, b, c, d, e):
        """
        Updates the atmospheric turbidity given the luminance coefficients.

        Parameters
        ----------
        a: float
            the darkening or brightening of horizon
        b: float
            the luminance gradient near the horizon
        c: float
            the relative intensity of the circumsolar region
        d: float
            the width of the circumsolar region
        e: float
            the relative backscattered light
        """
        T_T = np.linalg.pinv(T_L)
        tau_L, c = T_T.dot(np.array([a, b, c, d, e]))
        self.__tau_L = tau_L / c  # turbidity correction

    def copy(self):
        """
        Generates a copy of the instance.

        Returns
        -------
        AnalyticalSky
        """
        sky = AnalyticalSky()
        sky.tau_L = self.tau_L
        sky.theta_s = self.theta_s
        sky.phi_s = self.phi_s
        sky.__c1 = self.__c1
        sky.__c2 = self.__c2

        sky._is_generated = False
        return sky

    @staticmethod
    def from_observer(obs=None, date=None, ori=None):
        """
        Creates a sky instance using an observer on Earth.

        Parameters
        ----------
        obs: Observer, optional
            the observer (location on Earth). Default is the Seville observer
        date: datetime, optional
            the date of the observation. Default is 21/06/2021 - 10:00 am
        ori: R, optioanl
            the heading orientation of the observer. Default is 0.

        Returns
        -------
        AnalyticalSky
        """

        sun = Sun()
        if obs is None:
            obs = get_seville_observer()
            obs.date = datetime(2021, 6, 21, 10, 0, 0) if date is None else date
        sun.compute(obs)
        if ori is not None:
            yaw, pitch, roll = ori.as_euler('ZYX', degrees=False)
        else:
            yaw = 0.
        theta_s, phi_s = np.pi/2 - sun.alt, (sun.az - yaw + np.pi) % (2 * np.pi) - np.pi

        return AnalyticalSky(theta_s=theta_s, phi_s=phi_s)

    @staticmethod
    def from_type(sky_type):
        """
        Creates a sky model using a type description.

        - 1: Steep luminance gradation towards zenith, azimuthal uniformity
        - 2: Overcast, with steep luminance gradation and slight brightening towards the sun
        - 3: Overcast, moderately graded with azimuthal uniformity
        - 4: Overcast, moderately graded and slightly brightening towards the sun
        - 5: Sky uniform luminance
        - 6: Partly cloudy agent_old, no gradation towards zenith, slight brighening towards the sun
        - 7: Partly cloudy agent_old, no gradation towards zenith, brighter circumsolar region
        - 8: Partly cloudy agent_old, no gradation towards zenith, distinct solar corona
        - 9: Partly cloudy, with the obscured sun
        - 10: Partly cloudy, with brighter circumsolar region
        - 11: White-blue agent_old with distinct solar corona
        - 12: CIE Standard Clear Sky, low illuminance turbidity
        - 13: CIE Standard Clear Sky, polluted atmosphere
        - 14: Cloudless turbid agent_old with broad solar corona
        - 15: White-blue turbid agent_old with broad solar corona

        Parameters
        ----------
        sky_type: int
            a number in range [1-15] identifying the type of the sky

        Returns
        -------
        AnalyticalSky
        """
        import os
        import yaml

        sp = os.path.join(__root__, 'data', 'standard-parameters.yaml')
        with open(sp, 'r') as f:
            try:
                sp = yaml.load(f)
            except yaml.YAMLError as exc:
                print("Could not load the env types.", exc)
                return None

        rep = sp['type'][sky_type-1]
        a = sp['gradation'][rep['gradation']]['a']
        b = sp['gradation'][rep['gradation']]['b']
        c = sp['indicatrix'][rep['indicatrix']]['c']
        d = sp['indicatrix'][rep['indicatrix']]['d']
        e = sp['indicatrix'][rep['indicatrix']]['e']

        s = AnalyticalSky()
        s._update_turbidity(a, b, c, d, e)
        # s.__tau_L = 2.

        for description in rep['description']:
            print(description)

        return s


class PragueSky(SkyBase):
    def __init__(self, theta_s=0., phi_s=0., degrees=False, name="prague-sky"):
        SkyBase.__init__(self, theta_s=theta_s, phi_s=phi_s, degrees=degrees, name=name)
        self.__model = prague.PragueSkyModel()

    def initialise(self, filename, single_visibility=0.0):
        self.__model.initialise(filename, single_visibility)

    def __call__(self, ori, wavelengths=None, albedo=None, altitude=None, visibility=None):
        assert self.is_initialised

        if wavelengths is None:
            wavelengths = np.array([prague.SPECTRAL_RESPONSE_START + 240])
        if albedo is None:
            albedo = 0.5
        if altitude is None:
            altitude = 1.0
        if visibility is None:
            visibility = 50.0

        # We are viewing the sky from 'altitude' meters above the origin
        viewpoint = np.array([0, 0, altitude], dtype='float64')
        views_arr = geo.get_coordinates(ori)

        # Get internal model parameters for the desired confirmation.
        params = self.__model.compute_parameters(viewpoint, views_arr, self.theta_s, self.phi_s, visibility, albedo)

        y = self.__model.sky_radiance(params, wavelengths) + self.__model.sun_radiance(params, wavelengths)
        p = np.abs(self.__model.polarisation(params, wavelengths))
        a = AnalyticalSky.compute_angle_of_polarisation(ori, phi_s=self.phi_s, theta_s=self.theta_s)

        sun_xyz = geo.sph2xyz(self.theta_s, self.phi_s)

        return SkyInstance(
            sun_direction=sun_xyz,
            view_direction=views_arr,
            sky_radiance=y,
            degree_of_polarisation=p,
            angle_of_polarisation=a,
            albedo=albedo,
            wavelengths=wavelengths
        )

    @property
    def is_initialised(self):
        return self.__model.is_initialised


def spectrum_influence(v, irgbu):
    """
    Decomposes the luminance into 5 distinct spectral channels based on the sensitivity provided.

    Parameters
    ----------
    v: np.ndarray[float]
        received luminance (white light)
    irgbu: np.ndarray[float]
        array of sensitivities for each channel (IR, R, G, B, UV)

    Returns
    -------
    np.ndarray[float]
        the luminance received in each channel
    """
    wl = np.array([1200, 715, 535, 475, 350], dtype='float32')
    v = v[..., np.newaxis]
    irgbu = np.vstack([irgbu] + [irgbu for _ in range(v.shape[0] // irgbu.shape[0] - 1)])

    l1 = 10.0 * irgbu * np.power(wl / 1000., 8) * np.square(v) / float(v.size)
    l2 = 0.001 * irgbu * np.power(1000. / wl, 8) * np.nansum(np.square(v)) / float(v.size)

    v_max = np.nanmax(v)
    w_max = np.nanmax(v + l1 + l2)
    w = v_max * (v + l1 + l2) / w_max
    if isinstance(irgbu, np.ndarray):
        if irgbu.shape[0] == 1 and w.shape[0] > irgbu.shape[0]:
            irgbu = np.vstack([irgbu] * w.shape[0])
        w[irgbu < 0] = np.hstack([v] * irgbu.shape[1])[irgbu < 0]
    elif irgbu < 0:
        w = v
    return w
