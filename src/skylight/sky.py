"""
Package that allows computations of the skylight properties.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from ._static import __root__, eps, T_L, MODES, SPECTRUM_WAVELENGTHS, SkyInfo

from . import prague
from .observer import Observer, get_seville_observer
from .ephemeris import Sun

import skylight.geometry as geo

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from copy import copy

import numpy as np


class SkyBase(object):
    def __init__(self, theta_s=0., phi_s=0., degrees=False, name="skylight"):
        """
        The basic Sky class.

        This is a callable class, which provides skylight information on the given location, direction, and atmospheric
        conditions.

        Examples
        --------
        >>> SkyBase(theta_s=0, phi_s=40, degrees=True, name='my skylight')
        SkyBase(sun_azimuth=40.0, sun_elevation=0.0, name='my skylight', is_generated=False)

        Parameters
        ----------
        theta_s : float
            the solar elevation.
        phi_s : float
            the solar azimuth.
        degrees : bool
            flag of whether the angles are represented in degrees or radians.
        name : str
            a name for the skylight.
        """
        self.__max_radiance = 1.
        theta_s = np.deg2rad(theta_s) if degrees else theta_s
        self.__theta_s = theta_s
        self.__phi_s = np.deg2rad(phi_s) if degrees else phi_s

        self._last_generated = None
        self._is_generated = False
        self.__name = name

    def __call__(self, ori, wavelengths=None, albedo=None, altitude=None, visibility=None, mode=None):
        """
        Generates the skylight properties for the given orientations and spectral influences.

        Examples
        --------
        >>> s = SkyBase(theta_s=0, phi_s=40, degrees=True, name='my skylight')
        >>> s.is_generated
        False
        >>> s(np.array([[0., 0., 1.]]), wavelengths=np.array([[540]]))  # doctest: +ELLIPSIS
        SkyInfo(sun_direction=array([0.76..., 0.64..., 0.        ]), view_direction=array([[0., 0., 1.]]), sky_radiance=array([0.]), sun_radiance=None, transmittance=None, degree_of_polarisation=array([0.2]), angle_of_polarisation=array([-0.87...]), wavelengths=array([[540]]), albedo=1.0, altitude=1.0, visibility=200.0)
        >>> s.is_generated
        True

        Parameters
        ----------
        ori: R, np.ndarray[float], optional
            orientation of the interesting elements. Default is None
        wavelengths: np.ndarray[float], optional
            the wavelengths to render
        albedo: float, optional
            the abledo
        altitude: float, optional
            the altitude of the observer (hight above the ground)
        visibility: float, option
            the visibility in kilometers
        mode: str, int, optional
            option of what to render. Default is to render all.

        Returns
        -------
        SkyInstance
            the available skylight information with respect to the given parameters
        """

        assert self.is_initialised

        if wavelengths is None:
            wavelengths = np.copy(SPECTRUM_WAVELENGTHS)
        if albedo is None:
            albedo = 1.
        if altitude is None:
            altitude = 1.
        if visibility is None:
            visibility = 200.

        # set default arguments
        xyz = geo.get_coordinates(ori)

        # calculate light properties
        y, sy, p, tr, a = None, None, None, None, None
        if mode is None or mode == 0:
            y = np.full(xyz.shape[0], self.__max_radiance * np.power(np.sin(self.theta_s), 4))  # radiance
            # create cloud disturbance
            if albedo is None:
                albedo = np.ones_like(y)
            y *= albedo
        if mode is None or mode == 2:
            p = np.full(xyz.shape[0], 0.2)  # degree of polarisation
        if mode is None or mode == 4:
            a = self.compute_angle_of_polarisation(ori, self.phi_s, self.theta_s)

        sun_xyz = geo.sph2xyz(self.theta_s, self.phi_s)

        self._is_generated = True

        self._last_generated = SkyInfo(
            sun_direction=sun_xyz,
            view_direction=xyz,
            sky_radiance=y,
            sun_radiance=sy,
            degree_of_polarisation=p,
            angle_of_polarisation=a,
            transmittance=tr,
            wavelengths=wavelengths,
            albedo=albedo,
            altitude=altitude,
            visibility=visibility
        )
        return self._last_generated

    def __repr__(self):
        return (f"SkyBase(sun_azimuth={np.rad2deg(self.phi_s):.1f}, "
                f"sun_elevation={np.rad2deg(self.theta_s):.1f}, "
                f"name='{self.__name}', "
                f"is_generated={self.is_generated})")

    def copy(self):
        """
        Generates a copy of the instance.

        Examples
        --------
        >>> s = SkyBase(theta_s=0, phi_s=40, degrees=True, name='my skylight')
        >>> s2 = s.copy()
        >>> s2
        SkyBase(sun_azimuth=40.0, sun_elevation=0.0, name='my skylight', is_generated=False)
        >>> s == s2
        False
        >>> str(s) == str(s2)
        True
        >>> su = UniformSky(theta_s=0, phi_s=40, degrees=True, name='my uniform skylight')
        >>> su.copy()
        UniformSky(sun_azimuth=40.0, sun_elevation=0.0, name='my uniform skylight', is_generated=False)
        >>> sa = AnalyticalSky(theta_s=0, phi_s=40, degrees=True, name='my analytical skylight')
        >>> sa.copy()
        AnalyticalSky(sun_azimuth=40.0, sun_elevation=0.0, a=-1.11, b=-0.28, c=5.28, d=-2.34, e=0.24, name='my analytical skylight', is_generated=False)
        >>> sp = PragueSky(theta_s=0, phi_s=40, degrees=True, name='my prague skylight')
        >>> sp.copy()
        PragueSky(sun_azimuth=40.0, sun_elevation=0.0, name='my prague skylight', is_generated=False, is_initialised=False)

        Returns
        -------
        SkyBase
        """
        sky = self.__class__()
        for key in self.__dict__:
            sky.__dict__[key] = copy(self.__dict__[key])

        sky._is_generated = False
        return sky

    @property
    def max_radiance(self):
        return self.__max_radiance

    @max_radiance.setter
    def max_radiance(self, value):
        self._is_generated = value == self.__max_radiance and self._is_generated
        if not self.is_generated:
            self.__max_radiance = value

    @property
    def theta_s(self):
        return self.__theta_s

    @theta_s.setter
    def theta_s(self, value):
        self._is_generated = self.__theta_s == value and self._is_generated
        if not self.is_generated:
            self.__theta_s = value

    @property
    def phi_s(self):
        return self.__phi_s

    @phi_s.setter
    def phi_s(self, value):
        self._is_generated = self.__phi_s == value and self._is_generated
        if not self.is_generated:
            self.__phi_s = value

    @property
    def last_generated(self):
        return self._last_generated

    @property
    def is_generated(self):
        return self._is_generated

    @property
    def is_initialised(self):
        return True

    @staticmethod
    def compute_angle_of_polarisation(ori, phi_s, theta_s):
        """
        Computes the angle of polarisation at a given view direction and for a specific sun position relative
        to the observer.

        Examples
        --------
        >>> ori = geo.xyz2ori(np.array([[0., 0., 1.]]))
        >>> SkyBase.compute_angle_of_polarisation(ori, 0, np.pi/4)  # doctest: +ELLIPSIS
        array([-1.57...])

        Parameters
        ----------
        ori : R
            the view direction of the observer.
        phi_s : float
            the solar azimuth.
        theta_s : float
            the solar elevation.

        Returns
        -------
        np.ndarray[float]
        """
        v_s = geo.sph2xyz(theta_s, phi_s)

        if isinstance(ori, R):
            v_p = geo.ori2xyz(ori)
        else:
            v_p = ori
        c = np.cross(v_s, v_p)

        a = np.arctan2(c[:, 1], c[:, 0])
        return (a + np.pi / 2) % np.pi - np.pi / 2

    @classmethod
    def from_observer(cls, obs=None, date=None, ori=None, name=None):
        """
        Creates a skylight instance using an observer on Earth.

        Examples
        --------
        >>> o = get_seville_observer()
        >>> SkyBase.from_observer(o, datetime(2022, 12, 1, 10))
        SkyBase(sun_azimuth=115.8, sun_elevation=79.6, name='Seville basic skylight', is_generated=False)
        >>> AnalyticalSky.from_observer(o, datetime(2022, 12, 1, 10))
        AnalyticalSky(sun_azimuth=115.8, sun_elevation=79.6, a=-1.11, b=-0.28, c=5.28, d=-2.34, e=0.24, name='Seville analytical skylight', is_generated=False)
        >>> UniformSky.from_observer(o, datetime(2022, 12, 1, 10))
        UniformSky(sun_azimuth=115.8, sun_elevation=79.6, name='Seville uniform skylight', is_generated=False)
        >>> PragueSky.from_observer(o, datetime(2022, 12, 1, 10))
        PragueSky(sun_azimuth=115.8, sun_elevation=79.6, name='Seville prague skylight', is_generated=False, is_initialised=False)

        Parameters
        ----------
        obs : Observer, optional
            the observer (location on Earth). Default is the Seville observer
        date : datetime, optional
            the date of the observation. Default is 21/06/2021 - 10:00 am
        ori : R, optioanl
            the heading orientation of the observer. Default is 0.
        name : str, optional
            the name of the skylight instance.

        Returns
        -------
        SkyBase
        """

        if obs is None:
            obs = get_seville_observer()

        obs.date = datetime(2021, 6, 21, 10, 0, 0) if date is None else date
        sun = Sun(obs)

        if ori is not None:
            yaw, pitch, roll = ori.as_euler('ZYX', degrees=False)
        else:
            yaw = 0.
        theta_s, phi_s = sun.alt, (sun.az - yaw + np.pi) % (2 * np.pi) - np.pi

        if name is None:
            name = f"{obs.city} {cls.__name__.lower().replace('skybase', 'basic ').replace('skylight', ' ')}skylight"

        return cls(theta_s=theta_s, phi_s=phi_s, name=name)


class UniformSky(SkyBase):
    def __init__(self, theta_s=0., phi_s=0., degrees=False, radiance=1., name="uniform-skylight"):
        """
        A skylight with uniform intensity of light for all the wavelengths.

        Examples
        --------
        >>> UniformSky(theta_s=0, phi_s=40, degrees=True, name='my uniform skylight')
        UniformSky(sun_azimuth=40.0, sun_elevation=0.0, name='my uniform skylight', is_generated=False)

        Parameters
        ----------
        theta_s : float
            the solar elevation.
        phi_s : float
            the solar azimuth.
        degrees : bool
            whether the solar elevation and azimuth are given in degrees or not. Default is radians.
        radiance : float
            the uniform skylight radiance level.
        name : str
            the name of the skylight instance.
        """
        super(UniformSky, self).__init__(theta_s=theta_s, phi_s=phi_s, degrees=degrees, name=name)
        self.max_radiance = radiance

    def __repr__(self):
        return SkyBase.__repr__(self).replace("SkyBase", "UniformSky")


class AnalyticalSky(SkyBase):

    def __init__(self, theta_s=0., phi_s=0., degrees=False, name="skylight"):
        """
        The Analycal Sky class.
        This class provides skylight cues by using an analytical solution.

        Examples
        --------
        >>> AnalyticalSky(theta_s=0, phi_s=40, degrees=True, name='my analytical skylight')
        AnalyticalSky(sun_azimuth=40.0, sun_elevation=0.0, a=-1.11, b=-0.28, c=5.28, d=-2.34, e=0.24, name='my analytical skylight', is_generated=False)

        Parameters
        ----------
        theta_s: float, optional
            sun elevation (distance from horizon). Default is 0
        phi_s: float, optional
            sun azimuth (clockwise from North). Default is 0
        degrees: bool, optional
            True if the angles are given in degrees, False otherwise. Default is False
        name: str, optional
            a name for the skylight instance. Default is 'skylight'
        """
        SkyBase.__init__(self, theta_s=theta_s, phi_s=phi_s, degrees=degrees, name=name)
        self.__a, self.__b, self.__c, self.__d, self.__e = 0., 0., 0., 0., 0.
        self.__tau_L = 2.
        self._update_luminance_coefficients(self.__tau_L)
        self.__c1 = .6
        self.__c2 = 4.

    def __call__(self, ori, wavelengths=None, albedo=None, altitude=None, visibility=None, mode=None):
        """
        Generates the skylight properties for the given orientations and spectral influences.

        Examples
        --------
        >>> s = AnalyticalSky(theta_s=0, phi_s=40, degrees=True, name='my skylight')
        >>> s.is_generated
        False
        >>> s(np.array([[0., 0., 1.]]), wavelengths=np.array([540]))  # doctest: +ELLIPSIS
        SkyInfo(sun_direction=array([0.76..., 0.64..., 0.        ]), view_direction=array([[0., 0., 1.]]), sky_radiance=array([[3.39...e+13]]), sun_radiance=array([0.]), transmittance=None, degree_of_polarisation=array([[0.]]), angle_of_polarisation=array([-0.87...]), wavelengths=array([540]), albedo=0.5, altitude=0.0, visibility=59.4)
        >>> s.is_generated
        True

        Parameters
        ----------
        ori: R, np.ndarray[float]
            orientation of the interesting elements, either as Rotation instances or as 3D unit vectors
        wavelengths: np.ndarray[float], optional
            the spectral influence of the observer
        albedo: np.ndarray[float], optional
            :param eta: array of noise level in each point of interest

        Returns
        -------
        SkyInstance
            the luminance, degree and angle of polarisation
        """

        if wavelengths is None:
            wavelengths = np.copy(SPECTRUM_WAVELENGTHS)
        if albedo is None:
            albedo = 0.50
        if altitude is None:
            altitude = 0.0
        if visibility is None:
            visibility = 59.4

        # set default arguments
        xyz = geo.get_coordinates(ori)
        theta = geo.xyz2elevation(xyz)
        # phi = geo.xyz2azimuth(xyz)

        sun_xyz = geo.sph2xyz(self.theta_s, self.phi_s)

        # SKY INTEGRATION

        # distance of the element from the sun position
        gamma = geo.angle_between(xyz, sun_xyz)

        # Intensity
        i_prez = self.L(gamma, np.pi / 2 - theta)
        i_00 = self.L(0., np.pi / 2 - self.theta_s)  # the luminance (Cd/m^2) at the zenith point
        i_90 = self.L(np.pi / 2, np.absolute(self.theta_s))  # the luminance (Cd/m^2) on the horizon
        # influence of env intensity
        i = (1. / (i_prez + eps) - 1. / (i_00 + eps)) * i_00 * i_90 / (i_00 - i_90 + eps)

        # create cloud disturbance
        if albedo is None:
            albedo = np.ones_like(gamma)

        # calculate light properties
        y, sy, p, tr, a = None, None, None, None, None
        if mode is None or mode == 0:
            y = np.maximum(self.Y_z * i_prez / (i_00 + eps), 0.) / 25  # Illumination

            # create cloud disturbance
            if albedo is None:
                albedo = np.ones_like(y)
            y *= albedo

            y = self.spectrum_influence(y, wavelengths)
        if mode is None or mode == 1:
            sy = np.zeros_like(gamma)
            sy[gamma < np.pi / 180] = 100
        if mode is None or mode == 2:
            # Degree of Polarisation
            lp = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
            # p = np.clip(2. / np.pi * self.M_p * lp * (theta * np.cos(theta) + (np.pi / 2 - theta) * i), 0., 1.)
            p = np.clip(2. / np.pi * self.M_p * lp * ((np.pi / 2 - theta) * np.sin(theta) + theta * i), 0., 1.)
            p = self.spectrum_influence(p, wavelengths)
            # p = 0.7 * np.sqrt(p)

            p *= albedo  # destroy the polarisation pattern
        if mode is None or mode == 4:
            # Angle of polarisation
            a = self.compute_angle_of_polarisation(ori, self.phi_s, self.theta_s)
            a[np.isclose(albedo, 0)] = np.nan

        self._is_generated = True

        self._last_generated = SkyInfo(
            sun_direction=sun_xyz,
            view_direction=xyz,
            sky_radiance=y,
            sun_radiance=sy,
            degree_of_polarisation=p,
            angle_of_polarisation=a,
            transmittance=tr,
            wavelengths=wavelengths,
            albedo=albedo,
            altitude=altitude,
            visibility=visibility
        )

        return self._last_generated

    def __repr__(self):
        s = SkyBase.__repr__(self).replace("SkyBase", "AnalyticalSky")
        s = s.replace("name=", f"a={self.A:.2f}, b={self.B:.2f}, c={self.C:.2f}, d={self.D:.2f}, e={self.E:.2f}, name=")
        return s

    def L(self, chi, z):
        """
        Prez. et. al. Luminance function.
        Combines the scattering indicatrix and luminance gradation functions to compute the total
        luminance observed at the given env element(s).

        Examples
        --------
        >>> s = AnalyticalSky(theta_s=0, phi_s=40, degrees=True, name='my analytical skylight')
        >>> s.L(np.pi/2, np.pi/2)
        0.0
        >>> s.L(0., np.pi/2)  # doctest: +ELLIPSIS
        0.0
        >>> s.L(np.pi/2, 0.)  # doctest: +ELLIPSIS
        0.18966...
        >>> s.L(np.pi/2, np.pi/3)  # doctest: +ELLIPSIS
        0.42278...
        >>> s.L(0., 0.)  # doctest: +ELLIPSIS
        1.08920...

        Parameters
        ----------
        chi: np.ndarray[float], float
            angular distance between the observed element and the sun location -- [0, pi]
        z: np.ndarray[float], float
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
        self._is_generated = self.__a == value and self._is_generated
        if not self.is_generated:
            self.__a = value
            self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

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
        self._is_generated = self.__a == value and self._is_generated
        if not self.is_generated:
            self.__b = value
            self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

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
        self._is_generated = self.__a == value and self._is_generated
        if not self.is_generated:
            self.__c = value
            self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

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
        self._is_generated = self.__a == value and self._is_generated
        if not self.is_generated:
            self.__d = value
            self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

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
        self._is_generated = self.__a == value and self._is_generated
        if not self.is_generated:
            self.__e = value
            self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

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
        if not self.is_generated:
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

    @staticmethod
    def from_type(sky_type):
        """
        Creates a skylight model using a type description.
            1) Steep luminance gradation towards zenith, azimuthal uniformity
            2) Overcast, with steep luminance gradation and slight brightening towards the sun
            3) Overcast, moderately graded with azimuthal uniformity
            4) Overcast, moderately graded and slightly brightening towards the sun
            5) Sky uniform luminance
            6) Partly cloudy skylight, no gradation towards zenith, slight brighening towards the sun
            7) Partly cloudy skylight, no gradation towards zenith, brighter circumsolar region
            8) Partly cloudy skylight, no gradation towards zenith, distinct solar corona
            9) Partly cloudy, with the obscured sun
            10) Partly cloudy, with brighter circumsolar region
            11) White-blue skylight with distinct solar corona
            12) CIE Standard Clear Sky, low illuminance turbidity
            13) CIE Standard Clear Sky, polluted atmosphere
            14) Cloudless turbid skylight with broad solar corona
            15) White-blue turbid skylight with broad solar corona

        Examples
        --------
        >>> AnalyticalSky.from_type(1)
        CIE Standard Overcast Sky, alternative form
        Steep luminance gradation towards zenith, azimuthal uniformity.
        AnalyticalSky(sun_azimuth=0.0, sun_elevation=0.0, a=4.00, b=-0.70, c=0.00, d=-1.00, e=0.00, name='CIE Standard Overcast Sky, alternative form', is_generated=False)
        >>> AnalyticalSky.from_type(12)
        CIE Standard Clear Sky, low illuminance turbidity
        AnalyticalSky(sun_azimuth=0.0, sun_elevation=0.0, a=-1.00, b=-0.32, c=10.00, d=-3.00, e=0.45, name='CIE Standard Clear Sky, low illuminance turbidity', is_generated=False)

        Parameters
        ----------
        sky_type: int
            a number in range [1-15] identifying the type of the skylight

        Returns
        -------
        AnalyticalSky
        """
        import os
        import yaml

        sp = os.path.abspath(os.path.join('..', 'data', 'standard-parameters.yaml'))
        with open(sp, 'r') as f:
            try:
                sp = yaml.load(f, yaml.Loader)
            except yaml.YAMLError as exc:
                print("Could not load the env types.", exc)
                return None

        rep = sp['type'][sky_type-1]

        s = AnalyticalSky(name=rep['description'][0])

        s.__a = sp['gradation'][rep['gradation']]['a']
        s.__b = sp['gradation'][rep['gradation']]['b']
        s.__c = sp['indicatrix'][rep['indicatrix']]['c']
        s.__d = sp['indicatrix'][rep['indicatrix']]['d']
        s.__e = sp['indicatrix'][rep['indicatrix']]['e']

        s._update_turbidity(s.__a, s.__b, s.__c, s.__d, s.__e)

        for description in rep['description']:
            print(description)

        return s

    @staticmethod
    def spectrum_influence(v, wavelengths, sensitivities=None):
        """
        Approximates the relative influence of the wavelengths to the values.

        Examples
        --------
        >>> AnalyticalSky.spectrum_influence(np.array([0.0]), np.array([540]))
        array([[0.]])
        >>> AnalyticalSky.spectrum_influence(np.array([1.0]), np.array([540]))
        array([[1.]])
        >>> AnalyticalSky.spectrum_influence(np.array([0.00, 0.25, 0.50, 0.75, 1.00]), np.array([300, 540, 1000]))
        array([[0.32  , 0.49  , 0.66  , 0.83  , 1.    ],
               [0.    , 0.1876, 0.4584, 0.7292, 1.    ],
               [0.    , 0.    , 0.    , 0.375 , 1.    ]])
        >>> AnalyticalSky.spectrum_influence(np.array([0.0, 0.5, 1.0, 1.5, 2.0]), np.array([300, 540, 1000]))
        array([[0.64  , 0.98  , 1.32  , 1.66  , 2.    ],
               [0.    , 0.3752, 0.9168, 1.4584, 2.    ],
               [0.    , 0.    , 0.    , 0.75  , 2.    ]])

        Parameters
        ----------
        v: np.ndarray[float]
            received value (typically the radiance or polarisation)
        wavelengths: np.ndarray[float]
            array of wavelengths
        sensitivities: np.ndarray[float], optional
            array of sensitivities for each channel

        Returns
        -------
        np.ndarray[float]
        """
        if sensitivities is None:
            sensitivities = np.ones(wavelengths.shape, dtype='float64')
        v = v[..., np.newaxis]

        # l1 = 10.0 * sensitivities * np.power(wavelengths / 1000., 8) * np.square(v) / float(v.size)
        # l2 = 0.0001 * sensitivities * np.power(1000. / wavelengths, 8) * np.nansum(np.square(v)) / float(v.size)
        l1 = sensitivities * v * np.power(wavelengths / 500., 2)
        l2 = v.max() - l1.max(axis=0)

        v_max = np.nanmax(v)
        w = np.maximum(v + l1 + l2, 0)
        w_max = np.nanmax(w)
        w = v_max * w / np.maximum(w_max, eps)

        return w.T


class PragueSky(SkyBase):
    def __init__(self, theta_s=0., phi_s=0., degrees=False, name="prague-skylight"):
        """
        The Sky environment class. This class provides skylight cues by using an analytical solution.

        Examples
        --------
        >>> PragueSky(theta_s=0, phi_s=40, degrees=True, name='my prague skylight')
        PragueSky(sun_azimuth=40.0, sun_elevation=0.0, name='my prague skylight', is_generated=False, is_initialised=False)

        Parameters
        ----------
        theta_s: float, optional
            sun elevation (distance from horizon). Default is 0
        phi_s: float, optional
            sun azimuth (clockwise from North). Default is 0
        degrees: bool, optional
            True if the angles are given in degrees, False otherwise. Default is False
        name: str, optional
            a name for the skylight instance. Default is 'prague-skylight'
        """
        SkyBase.__init__(self, theta_s=theta_s, phi_s=phi_s, degrees=degrees, name=name)
        self.__model = prague.PragueSkyModelManager()

    def initialise(self, filename, single_visibility=0.0):
        self.__model.initialise(filename, single_visibility)

    def __call__(self, ori, wavelengths=None, albedo=None, altitude=None, visibility=None, mode=None):
        assert self.is_initialised

        if wavelengths is None:
            wavelengths = np.copy(SPECTRUM_WAVELENGTHS)
        if albedo is None:
            albedo = 0.50
        if altitude is None:
            altitude = 0.0
        if visibility is None:
            visibility = 59.4

        # We are viewing the skylight from 'altitude' meters above the origin
        viewpoint = np.array([0, 0, altitude], dtype='float64')
        views_arr = geo.get_coordinates(ori)

        # Get internal model parameters for the desired confirmation.
        params = self.__model.compute_parameters(viewpoint, views_arr, self.theta_s, self.phi_s, visibility, albedo)

        mode = get_mode_number(mode)
        y, sy, p, tr, a = None, None, None, None, None
        if mode is None or mode == 0:
            y = self.__model.sky_radiance(params, wavelengths)
        if mode is None or mode == 1:
            sy = self.__model.sun_radiance(params, wavelengths)
        if mode is None or mode == 2:
            p = np.abs(self.__model.polarisation(params, wavelengths))
        if mode is None or mode == 3:
            tr = self.__model.transmittance(params, wavelengths, np.finfo(float).max)
        if mode is None or mode == 4:
            a = AnalyticalSky.compute_angle_of_polarisation(ori, phi_s=self.phi_s, theta_s=self.theta_s)

        sun_xyz = geo.sph2xyz(self.theta_s, self.phi_s)

        self._is_generated = True

        self._last_generated = SkyInfo(
            sun_direction=sun_xyz,
            view_direction=views_arr,
            sky_radiance=y,
            sun_radiance=sy,
            transmittance=tr,
            degree_of_polarisation=p,
            angle_of_polarisation=a,
            wavelengths=wavelengths,
            albedo=albedo,
            altitude=altitude,
            visibility=visibility
        )

        return self._last_generated

    def __repr__(self):
        s = SkyBase.__repr__(self).replace("SkyBase", "PragueSky")
        s = s.replace(")", f", is_initialised={self.is_initialised}")
        if self.is_initialised:
            s += f", {self.available_data}"
        return s + ")"

    @property
    def is_initialised(self):
        return self.__model.is_initialised

    @property
    def available_data(self):
        if self.is_initialised:
            return self.__model.available_data
        else:
            return None


def get_mode_number(mode):
    if isinstance(mode, str):
        return MODES.index(mode.lower())
    else:
        return mode


def get_mode(i):
    if i < len(MODES):
        return MODES[i]
    else:
        return None
