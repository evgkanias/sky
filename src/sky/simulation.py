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

from .__helpers import __root__, eps, RNG, add_noise

from .observer import Observer, get_seville_observer
from .ephemeris import Sun

from scipy.spatial.transform import Rotation as R
from datetime import datetime

import numpy as np

T_L = np.array([[ 0.1787, -1.4630],
                [-0.3554,  0.4275],
                [-0.0227,  5.3251],
                [ 0.1206, -2.5771],
                [-0.0670,  0.3703]])
"""Transformation matrix of turbidity to luminance coefficients"""


class UniformSky(object):
    def __init__(self, luminance=1., name="uniform-sky"):
        self.__luminance = luminance

        self.__y = np.full(1, np.nan)
        self.__aop = np.full(1, np.nan)
        self.__dop = np.full(1, np.nan)
        self.__eta = np.full(1, False)
        self.__theta = np.full(1, np.nan)
        self.__phi = np.full(1, np.nan)

        self._is_generated = False
        self.__name = name

    def __call__(self, ori=None, irgbu=None, noise=0., eta=None, rng=RNG):
        """
        Generates the skylight properties for the given orientations and spectral influences.

        Parameters
        ----------
        ori: R, optional
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
        Y: np.ndarray[float]
            the luminance
        P: np.ndarray[float]
            the degree of polarisation
        A: np.ndarray[float]
            the angle of polarisation
        """

        # set default arguments
        self._update_coordinates(ori)

        # calculate light properties
        y = np.full_like(self.__phi, self.__luminance)  # Illumination
        p = np.full_like(self.__phi, 0.)  # Illumination
        a = np.full_like(self.__phi, np.nan)  # Illumination

        # create cloud disturbance
        if eta is None:
            eta = add_noise(noise=noise, shape=y.shape, rng=rng)
        y[eta] = 0.

        self.__y = y
        self.__dop = p
        self.__aop = a
        self.__eta = eta

        self._is_generated = True

        if irgbu is not None:
            y = spectrum_influence(y, irgbu).sum(axis=1)

        return y, p, a

    def _update_coordinates(self, ori=None):
        if ori is not None:
            xyz = np.clip(ori.apply([-1, 0, 0]), -1, 1)
            phi = np.arctan2(xyz[..., 1], xyz[..., 0])
            theta = np.arccos(xyz[..., 2])
            # theta[xyz[..., 2] > 0] = np.pi - theta[xyz[..., 2] > 0]
            phi = (phi + np.pi) % (2 * np.pi) - np.pi

            # save points of interest
            self.theta = theta.copy()
            self.phi = phi.copy()
        else:
            ori = R.from_euler('ZY', np.vstack([self.phi, self.theta]).T, degrees=False)

        return ori

    @property
    def Y(self):
        """
        The luminance of the sky (K cd/m^2)

        Returns
        -------
        np.ndarray[float]
        """
        assert self._is_generated, "Sky is not generated yet. In order to generate the env, use the call function."
        return self.__y

    @property
    def DOP(self):
        """
        The linear degree of polarisation in the sky

        Returns
        -------
        np.ndarray[float]
        """
        assert self._is_generated, "Sky is not generated yet. In order to generate the env, use the call function."
        return self.__dop

    @property
    def AOP(self):
        """
        The angle of linear polarisation in the sky

        Returns
        -------
        np.ndarray[float]
        """
        assert self._is_generated, "Sky is not generated yet. In order to generate the env, use the call function."
        return self.__aop

    @property
    def theta(self):
        """
        The elevation of the last used elements.

        Returns
        -------
        np.ndarray[float]
        """
        assert self._is_generated, "Sky is not generated yet. In order to generate sky env, use the call function."
        return self.__theta

    @theta.setter
    def theta(self, value):
        self.__theta = value
        self._is_generated = False

    @property
    def phi(self):
        """
        The azimuth of the last used elements.

        Returns
        -------
        np.ndarray[float]
        """
        assert self._is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__phi

    @phi.setter
    def phi(self, value):
        self.__phi = value
        self._is_generated = False

    @property
    def eta(self):
        """
        The percentage of noise induced in each element.

        Returns
        -------
        np.ndarray[float]
        """
        assert self._is_generated, "Sky is not generated yet. In order to generate the sky, use the call function."
        return self.__eta

    @eta.setter
    def eta(self, value):
        self.__eta = value
        self._is_generated = False

    @property
    def _y(self):
        return self.__y

    @_y.setter
    def _y(self, value):
        self.__y = value

    @property
    def _dop(self):
        return self.__dop

    @_dop.setter
    def _dop(self, value):
        self.__dop = value

    @property
    def _aop(self):
        return self.__aop

    @_aop.setter
    def _aop(self, value):
        self.__aop = value

    @property
    def _theta(self):
        return self.__theta

    @_theta.setter
    def _theta(self, value):
        self.__theta = value

    @property
    def _phi(self):
        return self.__phi

    @_phi.setter
    def _phi(self, value):
        self.__phi = value


class Sky(UniformSky):

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
        super(Sky, self).__init__(name=name)
        self.__a, self.__b, self.__c, self.__d, self.__e = 0., 0., 0., 0., 0.
        self.__tau_L = 2.
        self._update_luminance_coefficients(self.__tau_L)
        self.__c1 = .6
        self.__c2 = 4.
        self.theta_s = np.deg2rad(theta_s) if degrees else theta_s
        self.theta_s = np.pi/2 - theta_s
        self.phi_s = np.deg2rad(phi_s) if degrees else phi_s

    def __call__(self, ori=None, irgbu=None, noise=0., eta=None, rng=RNG):
        """
        Generates the skylight properties for the given orientations and spectral influences.

        Parameters
        ----------
        ori: R, optional
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
        Y: np.ndarray[float]
            the luminance
        P: np.ndarray[float]
            the degree of polarisation
        A: np.ndarray[float]
            the angle of polarisation
        """

        # set default arguments
        ori = self._update_coordinates(ori)
        theta = self._theta
        phi = self._phi

        theta_s, phi_s = self.theta_s, self.phi_s

        # SKY INTEGRATION

        # distance of the element from the sun position
        gamma = np.arccos(np.sin(theta) * np.sin(theta_s) +
                          np.cos(theta) * np.cos(theta_s) * np.cos(phi - phi_s))

        # Intensity
        i_prez = self.L(gamma, theta)
        i_00 = self.L(0., np.pi/2 - theta_s)  # the luminance (Cd/m^2) at the zenith point
        i_90 = self.L(np.pi / 2, np.absolute(theta_s))  # the luminance (Cd/m^2) on the horizon
        # influence of env intensity
        i = (1. / (i_prez + eps) - 1. / (i_00 + eps)) * i_00 * i_90 / (i_00 - i_90 + eps)
        y = np.maximum(self.Y_z * i_prez / (i_00 + eps), 0.)  # Illumination

        # Degree of Polarisation
        lp = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
        p = np.clip(2. / np.pi * self.M_p * lp * (theta * np.cos(theta) + (np.pi / 2 - theta) * i), 0., 1.)

        # Angle of polarisation
        ori_s = R.from_euler('ZY', [phi_s, theta_s], degrees=False)
        x_s, y_s, _ = ori_s.apply([1, 0, 0]).T
        x_p, y_p, _ = ori.apply([1, 0, 0]).T
        a_x = np.arctan2(y_p - y_s, x_p - x_s) + np.pi/2
        a = (a_x + np.pi) % (2 * np.pi) - np.pi

        # create cloud disturbance
        if eta is None:
            eta = add_noise(noise=noise, shape=y.shape, rng=rng)
        y[eta] = 0.
        p[eta] = 0.  # destroy the polarisation pattern
        a[eta] = np.nan

        y[gamma < np.pi/60] = 17

        self._y = y
        self._dop = p
        self._aop = a
        self._eta = eta

        self._is_generated = True

        if irgbu is not None:
            y = spectrum_influence(y, irgbu).sum(axis=1)

        return y, p, a

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
        Sky
        """
        sky = Sky()
        sky.tau_L = self.tau_L
        sky.theta_s = self.theta_s
        sky.phi_s = self.phi_s
        sky.__c1 = self.__c1
        sky.__c2 = self.__c2

        sky.__theta = self.__theta
        sky.__phi = self.__phi
        sky.__aop = self.__aop
        sky.__dop = self.__dop
        sky.__y = self.__y
        sky.__eta = self.__eta

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
        Sky
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

        return Sky(theta_s=theta_s, phi_s=phi_s)

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
        Sky
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

        s = Sky()
        s._update_turbidity(a, b, c, d, e)
        # s.__tau_L = 2.

        for description in rep['description']:
            print(description)

        return s


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
