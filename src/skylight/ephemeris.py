"""
Package that contains functions related to the sun course given the day, time and an observer on Earth.
"""
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from .observer import Observer

from datetime import datetime

import numpy as np

eps = np.finfo(float).eps


class Sun(object):
    def __init__(self, observer=None):
        """
        Instance of the sun based on the observer on Earth. The observer contains information like their longitude and
        latitude and the date and time, which are used by the sun in order to compute it position with respect to the
        observer.

        Examples
        --------
        >>> o = Observer(lon=np.deg2rad(0), lat=np.deg2rad(42), date=datetime(2022, 12, 1, 9))
        >>> Sun(o)
        Sun(azimuth=2.4383157531483253, elevation=0.26511958245980255, lon=0.0, lat=0.7330382858376184, ready=True)

        Parameters
        ----------
        observer: Observer
            the observer who observes the sun
        """
        self._jd = 0.
        """The Julian day"""
        self._srv = 0.
        """The solar relative vector"""

        self._sd = 0.
        """
        The declination angle.
        """
        self._eot = 0.
        """
        The Equation of Time (EoT) (in minutes) is an empirical equation that corrects for the eccentricity of the
        Earth's orbit and the Earth's axial tilt
        """
        self._sn = 0.
        """The solar noon."""
        self._srt = 0.
        """The sunrise relative time."""
        self._sst = 0.
        """The sunset relative time."""
        self._sld = 0.
        """The duration of teh sunlight."""
        self._sea = 0.
        """Solar elevation without the correction for atmospheric refraction."""
        self._aar = 0.
        """The approximate atmospheric refraction."""
        self._hra = 0.
        """The hour angle."""
        self._tst = 0.
        """The true solar time."""

        self._alt = 0.
        """The altitude of the sun (rads). Solar elevation (altitude) corrected for atmospheric refraction."""
        self._azi = 0.
        """The azimuth of the sun (rads)."""
        self._is_ready = False
        """If all the parameters has been computed based on the updated observer."""

        self._obs = None
        """The observer of the sun on Earth."""

        # set the observer of the sun on Earth
        if observer is not None:
            self.compute(observer)

    def compute(self, observer):
        """
        Computes all the parameters of the sun given an observer.

        Parameters
        ----------
        observer: Observer
        """
        self.obs = observer
        lon, lat = observer._lon, observer._lat

        jd = self._jd = julian_day(observer.date)
        jc = julian_century(jd)

        gmls = geom_mean_long_sun(jc)
        gmas = geom_mean_anom_sun(jc)
        eeo = eccent_earth_orbit(jc)
        seoc = sun_eq_of_ctr(jc, gmas)
        stl = sun_true_long(gmls, seoc)
        sta = sun_true_anom(gmas, seoc)
        self._srv = sun_rad_vector(eeo, sta)

        sal = sun_app_long(jc, stl)
        moe = mean_obliq_ecliptic(jc)
        oc = obliq_corr(jc, moe)
        sra = sun_rt_ascen(sal, oc)
        sd = self._sd = sun_declin(sal, oc)

        vy = var_y(oc)
        eot = self._eot = eq_of_time(gmls, gmas, eeo, vy)

        hasr = ha_sunrise(lat, sd)
        sn = self._sn = solar_noon(lon, eot, tz=self.obs.tzgmt)
        self._srt = sunrise_time(hasr, sn)
        self._sst = sunset_time(hasr, sn)
        self._sld = sunlight_duration(hasr)

        tst = self._tst = true_solar_time(lon, observer.date, eot, tz=self.obs.tzgmt)
        ha = self._hra = hour_angle(tst)
        sza = solar_zenith_angle(lat, sd, ha)
        sea = self._sea = solar_elevation_angle(sza)
        aar = self._aar = approx_atmospheric_refraction(sea)
        self._alt = solar_elevation_corrected_for_atm_refraction(sea, aar)
        self._azi = solar_azimuth_angle(lat, ha, sza, sd)

        self._is_ready = True

    def update(self):
        """
        Computes the parameters of the sun using the internal observer.
        """
        assert self.obs is not None, (
            "Observer has not been set. Please set the observer before you update the sun position."
        )

        self.compute(self.obs)

    def __repr__(self):
        o = ""
        if self.obs is not None:
            o = f", lon={self.obs.lon}, lat={self.obs.lat}"
        return f"Sun(azimuth={self.az}, elevation={self.alt}{o}, ready={self.is_ready})"

    @property
    def obs(self):
        """
        The observer who observes the sun.
        """
        return self._obs

    @obs.setter
    def obs(self, value):
        value.on_change = self.update
        self._obs = value
        self._is_ready = False

    @property
    def alt(self):
        """
        The altitude of the sun (rads). Solar elevation (altitude) corrected for atmospheric refraction.
        """
        return self._alt

    @property
    def az(self):
        """
        The azimuth of the sun (rads). Clockwise from North.
        """
        return self._azi

    @property
    def zenith_angle(self):
        """
        The angular distance of the sun from the zenith
        """
        return np.pi/2 - self._alt

    @property
    def equation_of_time(self):
        """
        The Equation of Time (EoT) (in minutes) is an empirical equation that corrects for the eccentricity of the
        Earth's orbit and the Earth's axial tilt
        """
        return self._eot

    @property
    def solar_elevation_angle(self):
        """
        Solar elevation without the correction for atmospheric refraction.
        """
        return self._sea

    @property
    def approximate_atmospheric_refraction(self):
        """
        The approximate atmospheric refraction
        """
        return self._aar

    @property
    def hour_angle(self):
        """
        The Hour Angle converts the local solar time (LST) into the number of degrees which the sun moves across the
        env. By definition, the HRA is 0° at solar noon. Since the Earth rotates 15° per hour away from solar noon
        corresponds to an angular motion of the sun in the env of 15°. In the morning the hour angle is negative, in
        the afternoon the hour angle is positive.
        """
        return self._hra

    @property
    def declination(self):
        """
        The declination angle.
        """
        return self._sd

    @property
    def sunrise(self):
        """
        The sunrise (absolute) time.
        """
        return relative_to_absolute_time(self._obs, self._srt)

    @property
    def sunset(self):
        """
        The sunset (absolute) time.
        """
        return relative_to_absolute_time(self._obs, self._sst)

    @property
    def is_ready(self):
        """
        True if the sun has been updated, otherwise False.
        """
        return self._is_ready


def julian_day(date):
    """
    The Julian day is the continuous count of days since the beginning of the Julian period (January 1 noon, 4713 BC),
    and is used primarily by astronomers, and in software for easily calculating elapsed days between two events.

    Examples
    --------
    >>> julian_day(datetime(1, 1, 1, 0, 0, 0))
    1721425.5
    >>> julian_day(datetime(2022, 12, 1, 12, 0, 0))
    2459915.0

    Parameters
    ----------
    date: datetime
        the date and time to be converted into the Julian day.

    Returns
    -------
    float
        the Julian day
    """
    return date.toordinal() + 1721424.5 + (date.hour + (date.minute + date.second / 60) / 60) / 24


def julian_century(jd):
    """
    The Julian century is the Julian day divided by 36525.

    Examples
    --------
    >>> julian_century(1721425.5)  # doctest: +ELLIPSIS
    -19.98958...
    >>> julian_century(2459915.0)  # doctest: +ELLIPSIS
    0.229158...

    Parameters
    ----------
    jd: float
        the Julian day

    Returns
    -------
    float
        the Julian century
    """
    return (jd - 2451545) / 36525


def geom_mean_long_sun(jc):
    """
    The geometric mean longitude of the sun (correct for aberration) at the given Julian century.

    Examples
    --------
    >>> geom_mean_long_sun(-19.98958247775496)  # doctest: +ELLIPSIS
    4.89093...
    >>> geom_mean_long_sun(0.22915811088295687)  # doctest: +ELLIPSIS
    4.36916...

    Parameters
    ----------
    jc: float
        the Julian century

    Returns
    -------
    float
    """
    return np.deg2rad((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)


def geom_mean_anom_sun(jc):
    """
    The geometric mean anomaly of the sun during the given Julian century.

    Examples
    --------
    >>> geom_mean_anom_sun(-19.98958247775496)  # doctest: +ELLIPSIS
    -12553.25476...
    >>> geom_mean_anom_sun(0.22915811088295687)  # doctest: +ELLIPSIS
    150.22054...

    Parameters
    ----------
    jc: float
        the Julian century

    Returns
    -------
    float
    """
    return np.deg2rad(357.52911 + jc * (35999.05029 - 0.0001537 * jc))


def eccent_earth_orbit(jc):
    """
    Eccentricity of Earth's orbit. Inclination of the plane of the Earth's orbit during the Julian century.

    Examples
    --------
    >>> eccent_earth_orbit(-19.98958247775496)  # doctest: +ELLIPSIS
    0.01749...
    >>> eccent_earth_orbit(0.22915811088295687)  # doctest: +ELLIPSIS
    0.01669...

    Parameters
    ----------
    jc: float
        the Julian century

    Returns
    -------
    float
    """
    return 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)


def sun_eq_of_ctr(jc, gmas):
    """
    The sun equation of center is the angular difference between the actual position of the sun with
    respect to the position of Earth, in its elliptical orbit and the position it would occupy if its motion were
    uniform, in a circular orbit of the same period.

    Examples
    --------
    >>> sun_eq_of_ctr(-19.98958247775496, -12553.254765205656)  # doctest: +ELLIPSIS
    0.01862...
    >>> sun_eq_of_ctr(0.22915811088295687, 150.22054910694445)  # doctest: +ELLIPSIS
    -0.01851...

    Parameters
    ----------
    jc: float
        the Julian century
    gmas: float
        the mean anomaly of the sun during the given Julian century

    Returns
    -------
    float
    """
    return np.deg2rad(np.sin(gmas) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
                      np.sin(2 * gmas) * (0.019993 - 0.000101 * jc) +
                      np.sin(3 * gmas) * 0.000289)


def sun_true_long(gmls, seoc):
    """
    The true longitude of the sun.

    Examples
    --------
    >>> sun_true_long(4.89093326966249, 0.018625224810268362)  # doctest: +ELLIPSIS
    4.90955...
    >>> sun_true_long(4.3691678972595875, -0.0185108976521297)  # doctest: +ELLIPSIS
    4.35065...

    Parameters
    ----------
    gmls: float
        the mean longitude of the sun at the given Julian century
    seoc: float
        the equation of the center of the sun

    Returns
    -------
    float
    """
    return gmls + seoc


def sun_true_anom(gmas, seoc):
    """
    The true anomaly of the sun.

    Examples
    --------
    >>> sun_true_anom(-12553.254765205656, 0.018625224810268362)  # doctest: +ELLIPSIS
    -12553.23613...
    >>> sun_true_anom(150.22054910694445, -0.0185108976521297)  # doctest: +ELLIPSIS
    150.20203...

    Parameters
    ----------
    gmas: float
        the mean anomaly of the sun during the given Julian century
    seoc: float
        the equation of the center of the sun

    Returns
    -------
    float
    """
    return gmas + seoc


def sun_rad_vector(eeo, sta):
    """
    Sun radius vector is the distance from the sun to earth.

    Examples
    --------
    >>> sun_rad_vector(0.017498308860870036, -12553.236139980847)  # doctest: +ELLIPSIS
    0.98516...
    >>> sun_rad_vector(0.016698994227039993, 150.2020382092923)  # doctest: +ELLIPSIS
    0.98607...

    Parameters
    ----------
    eeo: float
        inclination of the plane of the Earth's orbit during the Julian century
    sta: float
        the true anomaly of the sun

    Returns
    -------
    float
    """
    return (1.000001018 * (1 - np.square(eeo))) / (1 + eeo * np.cos(sta) + eps)


def sun_app_long(jc, stl):
    """
    The apparent longitude of the sun is the celestial longitude corrected for aberration and nutation as opposed
    to the mean longitude.

    Examples
    --------
    >>> sun_app_long(-19.98958247775496, 4.909558494472758)  # doctest: +ELLIPSIS
    4.90954...
    >>> sun_app_long(0.22915811088295687, 4.350656999607458)  # doctest: +ELLIPSIS
    4.35050...

    Parameters
    ----------
    jc: float
        the Julian century
    stl: float
        the true longitude of the sun

    Returns
    -------
    float
    """
    return stl - np.deg2rad(0.00569 + 0.00478 * np.sin(np.deg2rad(125.04 - 1934.136 * jc)))


def mean_obliq_ecliptic(jc):
    """
    The mean obliquity of the ecliptic given the Julian century. The angle between the plane of the earth's orbit and
    the plane of the earth's equator; the "tilt" of the earth.

    Examples
    --------
    >>> mean_obliq_ecliptic(-19.98958247775496)  # doctest: +ELLIPSIS
    0.41355...
    >>> mean_obliq_ecliptic(0.22915811088295687)  # doctest: +ELLIPSIS
    0.40904...

    Parameters
    ----------
    jc: float
        the Julian century

    Returns
    -------
    float
    """
    return np.deg2rad(23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60)


def obliq_corr(jc, moe):
    """
    The oblique correction refers to a particular type of the radiative corrections in the electroweak sector of the
    Standard model

    Examples
    --------
    >>> obliq_corr(-19.98958247775496, 0.4135583997778804)  # doctest: +ELLIPSIS
    0.41355...
    >>> obliq_corr(0.22915811088295687, 0.409040793186992)  # doctest: +ELLIPSIS
    0.40907...

    Parameters
    ----------
    jc: float
        the Julian century
    moe: float
        the mean obliquity of the ecliptic

    Returns
    -------
    float
    """
    return moe + np.deg2rad(0.00256) * np.cos(np.deg2rad(125.04 - 1934.136 * jc))


def sun_rt_ascen(sal, oc):
    """
    The right ascension of the sun. This is the angular distance of the sun measured eastward along the celestial
    equator from the North at the March equinox to the (hour circle of the) point in question above the earth.

    Examples
    --------
    >>> sun_rt_ascen(4.909542539472096, 0.4135565374004205)  # doctest: +ELLIPSIS
    -1.35602...
    >>> sun_rt_ascen(4.350502065240473, 0.4090740925117191)  # doctest: +ELLIPSIS
    -1.96211...

    Parameters
    ----------
    sal: float
        the apparent longitude of the sun
    oc: float

    Returns
    -------
    float
    """
    return np.arctan2(np.cos(sal), np.cos(oc)) * np.sin(sal)


def sun_declin(sal, oc):
    """
    The declination of the sun. This is the angle between the rays of the sun and the plane of the earth's equator.

    Examples
    --------
    >>> sun_declin(4.909542539472096, 0.4135565374004205)  # doctest: +ELLIPSIS
    -0.40507...
    >>> sun_declin(4.350502065240473, 0.4090740925117191)  # doctest: +ELLIPSIS
    -0.38115...

    Parameters
    ----------
    sal: float
        the apparent longitude of the sun
    oc: float
        the oblique correction

    Returns
    -------
    float
    """
    return np.arcsin(np.sin(oc) * np.sin(sal))


def var_y(oc):
    """
    The var Y.

    Examples
    --------
    >>> var_y(0.4135565374004205)  # doctest: +ELLIPSIS
    0.04400...
    >>> var_y(0.4090740925117191)  # doctest: +ELLIPSIS
    0.04303...

    Parameters
    ----------
    oc: float
        the oblique correction

    Returns
    -------
    float
    """
    return np.square(np.tan(oc / 2))


def eq_of_time(gmls, gmas, eeo, vy):
    """
    The equation of time. Describes the discrepancy between two kinds of solar time.

    Examples
    --------
    >>> eq_of_time(4.89093326966249, -12553.254765205656, 0.017498308860870036, 0.04400624305111601)  # doctest: +ELLIPSIS
    -8.28296...
    >>> eq_of_time(4.3691678972595875, 150.22054910694445, 0.016698994227039993, 0.04303048062858904)  # doctest: +ELLIPSIS
    10.97725...

    Parameters
    ----------
    gmls: float
        the mean longitude of the sun at the given Julian century
    gmas: float
        the mean anomaly of the sun during the given Julian century
    eeo: float
        inclination of the plane of the Earth's orbit during the Julian century
    vy: float
        the var Y

    Returns
    -------
    float
    """
    return 4 * np.rad2deg(
        vy * np.sin(2 * gmls) -
        2 * eeo * np.sin(gmas) +
        4 * eeo * vy * np.sin(gmas) * np.cos(2 * gmls) -
        0.5 * np.square(vy) * np.sin(4 * gmls) - 1.25 * np.square(eeo) * np.sin(2 * gmas))


def ha_sunrise(lat, sd):
    """
    The sunrise hour angle.

    Examples
    --------
    >>> ha_sunrise(55.946388, -0.40507056917407647)  # doctest: +ELLIPSIS
    1.89017...
    >>> ha_sunrise(55.946388, -0.3811597637734667)  # doctest: +ELLIPSIS
    1.86975...

    Parameters
    ----------
    lat: float
        the latitude of the observer
    sd: float
        the declination of the sun

    Returns
    -------
    float
    """
    return np.arccos(np.clip(np.cos(np.deg2rad(90.833)) / (np.cos(lat) * np.cos(sd)) - np.tan(lat) * np.tan(sd) + eps, -1, 1))


def solar_noon(lon, eot, tz=0):
    """
    The solar noon.

    Examples
    --------
    >>> solar_noon(-3.200000, -8.282966477106893)  # doctest: +ELLIPSIS
    1.01504...
    >>> solar_noon(-3.200000, 10.977258972213589)  # doctest: +ELLIPSIS
    1.00167...
    >>> solar_noon(-3.200000, 10.977258972213589, tz=+1)  # doctest: +ELLIPSIS
    1.04333...
    >>> solar_noon(-3.200000, 10.977258972213589, tz=-1)  # doctest: +ELLIPSIS
    0.96000...

    Parameters
    ----------
    lon: float
        the longitude of the observer
    eot: float
        the equation of time
    tz: int
        the timezone (from GMT)

    Returns
    -------
    float
    """
    return (720 - 4 * np.rad2deg(lon) - eot + tz * 60) / 1440


def sunrise_time(hasr, sn):
    """
    The sunrise time.

    Examples
    --------
    >>> sunrise_time(1.8901777282494314, 1.0150478779476115)  # doctest: +ELLIPSIS
    0.71421...
    >>> sunrise_time(1.8697527309361108, 1.0016727213855836)  # doctest: +ELLIPSIS
    0.70409...

    Parameters
    ----------
    hasr: float
        the sunrise hour angle
    sn: float
        the solar noon

    Returns
    -------
    float
    """
    return sn - np.rad2deg(hasr) * 4 / 1440


def sunset_time(hasr, sn):
    """
    The sunset time.

    Examples
    --------
    >>> sunset_time(1.8901777282494314, 1.0150478779476115)  # doctest: +ELLIPSIS
    1.31587...
    >>> sunset_time(1.8697527309361108, 1.0016727213855836)  # doctest: +ELLIPSIS
    1.29925...

    Parameters
    ----------
    hasr: float
        the sunrise hour angle
    sn: float
        the solar noon

    Returns
    -------
    float
    """
    return sn + np.rad2deg(hasr) * 4 / 1440


def sunlight_duration(hasr):
    """
    The duration of the sunlight during the current day.

    Examples
    --------
    >>> sunlight_duration(1.8901777282494314)  # doctest: +ELLIPSIS
    866.39365...
    >>> sunlight_duration(1.8697527309361108)  # doctest: +ELLIPSIS
    857.03152...

    Parameters
    ----------
    hasr: float
        the sunrise hour angle

    Returns
    -------
    float
    """
    return 8 * np.rad2deg(hasr)


def true_solar_time(lon, date, eot, tz=0):
    """
    The true solar time.

    Examples
    --------
    >>> true_solar_time(-3.200000, datetime(1, 1, 1, 0, 0, 0), -8.282966477106893)  # doctest: +ELLIPSIS
    698.33105...
    >>> true_solar_time(-3.200000, datetime(2022, 12, 1, 12, 0, 0), 10.977258972213589)  # doctest: +ELLIPSIS
    1437.59128...
    >>> true_solar_time(-3.200000, datetime(2022, 12, 1, 12, 0, 0), 10.977258972213589, tz=+1)  # doctest: +ELLIPSIS
    1377.59128...
    >>> true_solar_time(-3.200000, datetime(2022, 12, 1, 12, 0, 0), 10.977258972213589, tz=-1)  # doctest: +ELLIPSIS
    57.59128...

    Parameters
    ----------
    lon: float
        the longitude of the observer
    date: datetime
        the date and time of interest
    eot: float
        the equation of time
    tz: int
        the timezone (from GMT)

    Returns
    -------
    float
    """
    h = (date.hour + (date.minute + date.second / 60) / 60) / 24
    return (h * 1440 + eot + 4 * np.rad2deg(lon) - 60 * tz) % 1440


def hour_angle(tst):
    """
    The hour angle.

    Examples
    --------
    >>> hour_angle(698.3310557554393)  # doctest: +ELLIPSIS
    -0.09454...
    >>> hour_angle(1437.5912812047598)  # doctest: +ELLIPSIS
    3.13108...

    Parameters
    ----------
    tst: float
        the true solar time

    Returns
    -------
    float
    """
    return np.deg2rad(tst / 4 + 180 if tst < 0 else tst / 4 - 180)
    # return np.deg2rad(tst / 4 + 180) % (2 * np.pi) - np.pi


def solar_zenith_angle(lat, sd, ha):
    """
    The solar zenith angle.

    Examples
    --------
    >>> solar_zenith_angle(55.946388, -0.40507056917407647, -0.0945486056246651)  # doctest: +ELLIPSIS
    0.21378...
    >>> solar_zenith_angle(55.946388, -0.3811597637734667, 3.13108263515689)  # doctest: +ELLIPSIS
    2.15810...

    Parameters
    ----------
    lat: float
        the latitude of the observer
    sd: float
        the declination of the sun
    ha: float
        the hour angle

    Returns
    -------
    float
    """
    return np.arccos(np.sin(lat) * np.sin(sd) + np.cos(lat) * np.cos(sd) * np.cos(ha))


def solar_elevation_angle(sza):
    """
    The solar elevation angle.

    Examples
    --------
    >>> solar_elevation_angle(0.2137866392984689)  # doctest: +ELLIPSIS
    1.35700...
    >>> solar_elevation_angle(2.158102374952806)  # doctest: +ELLIPSIS
    -0.58730...

    Parameters
    ----------
    sza: float
        the solar zenith angle

    Returns
    -------
    float
    """
    return np.pi/2 - sza


def approx_atmospheric_refraction(sea):
    """
    The approximate atmospheric refraction.

    Examples
    --------
    >>> approx_atmospheric_refraction(1.3570096874964277)  # doctest: +ELLIPSIS
    1.04907...e-06
    >>> approx_atmospheric_refraction(-0.5873060481579095)  # doctest: +ELLIPSIS
    0.00015...

    Parameters
    ----------
    sea: float
        the solar elevation angle

    Returns
    -------
    float
    """
    tsea = np.tan(sea) + eps
    if np.rad2deg(sea) > 85:
        return 0
    elif np.rad2deg(sea) > 5:
        return np.deg2rad((58.1 / tsea - 0.07 / np.power(tsea, 3) + 0.000086 / np.power(tsea, 5)) / 3600)
    elif np.rad2deg(sea) > -0.575:
        return np.deg2rad((1735 + sea * (-518.2 + sea * (-518.2 + sea * (103.4 + sea * (-12.79 + sea * 0.711))))) / 3600)
    else:
        return np.deg2rad((-20.772 / tsea) / 3600)


def solar_elevation_corrected_for_atm_refraction(sea, aar):
    """
    The solar elevation corrected for the atmospheric refraction.

    Examples
    --------
    >>> solar_elevation_corrected_for_atm_refraction(1.3570096874964277, 1.049078887873835e-06)  # doctest: +ELLIPSIS
    1.35701...
    >>> solar_elevation_corrected_for_atm_refraction(-0.5873060481579095, 0.00015128646285593412)  # doctest: +ELLIPSIS
    -0.58715...

    Parameters
    ----------
    sea: float
        the solar elevation angle
    aar: float
        the approximate atmospheric refraction

    Returns
    -------
    float
    """
    return sea + aar


def solar_azimuth_angle(lat, ha, sza, sd):
    """
    The solar azimuth angle.

    Examples
    --------
    >>> solar_azimuth_angle(55.946388, -0.0945486056246651, 0.2137866392984689, -0.40507056917407647)  # doctest: +ELLIPSIS
    0.42132...
    >>> solar_azimuth_angle(55.946388, 3.13108263515689, 2.158102374952806, -0.3811597637734667)  # doctest: +ELLIPSIS
    3.15331...

    Parameters
    ----------
    lat: float
        the latitude of the observer
    ha: float
        the hour angle
    sza: float
        the solar zenith angle
    sd: float
        the declination of the sun

    Returns
    -------
    float
    """
    temp = np.arccos(((np.sin(lat) * np.cos(sza)) - np.sin(sd)) / (np.cos(lat) * np.sin(sza) + eps))
    if ha > 0:
        return (temp + np.pi) % (2 * np.pi)
    else:
        return (np.deg2rad(540) - temp) % (2 * np.pi)


def relative_to_absolute_time(obs, time):
    """
    Gets the data and timezone from an observer and overwrites its time based on the given time in days.

    Examples
    --------
    >>> o = Observer(lon=-3.200000, lat=55.946388, city='Edinburgh', degrees=True, date=datetime(2022, 12, 1, 12, 0, 0))
    >>> relative_to_absolute_time(o, 0.)  # doctest: +ELLIPSIS
    datetime.datetime(2022, 12, 1, 0, 0)
    >>> relative_to_absolute_time(o, 0.25)  # doctest: +ELLIPSIS
    datetime.datetime(2022, 12, 1, 6, 0)
    >>> relative_to_absolute_time(o, 0.5)  # doctest: +ELLIPSIS
    datetime.datetime(2022, 12, 1, 12, 0)
    >>> relative_to_absolute_time(o, 10.5)  # doctest: +ELLIPSIS
    datetime.datetime(2022, 12, 1, 12, 0)

    Parameters
    ----------
    obs: Observer
        the observer that we take the date and timezone from
    time: float
        time in days

    Returns
    -------
    datetime
    """
    h = (time % 1) * 24
    m = (h - int(h)) * 60
    s = (m - int(m)) * 60
    return datetime(year=obs.date.year, month=obs.date.month, day=obs.date.day,
                    hour=int(h), minute=int(m), second=int(s), tzinfo=obs.timezone)
