"""
Package that implements the observer on earth. Observers are used in order to place agents on earths coordinates,
calculate the sun position and also how this changes based on the movement of the agent on the earth.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from datetime import datetime, tzinfo
from pytz import timezone
from copy import copy

import numpy as np


class Observer(object):
    def __init__(self, lon=None, lat=None, date=datetime.now(), city=None, degrees=False):
        """
        The observer on Earth holds information about its longitude and latitude, date and time.

        Examples
        --------
        >>> Observer(lon=-3.200000, lat=55.946388, city='Edinburgh', degrees=True)  # doctest: +ELLIPSIS
        Observer(lon='-3.200000', lat='55.946388', city='Edinburgh', date='...', timezone='None')

        Parameters
        ----------
        lon: float, optional
            the longitude of the observer. Default is None
        lat: float, optional
            the latitude of the observer. Default is None
        date: datetime, optional
            the date and time of the event. Default is the current date and time
        city: str, optional
            the name of the city if available. Default is None
        degrees: bool, optional
            True if the longitude and latitude are given in degrees, False otherwise. This will also affect the form
            that they will be returned as attributes. Default is False
        """
        if lon is not None and lat is not None:
            self._lon = float(lon) if not degrees else np.deg2rad(float(lon))
            self._lat = float(lat) if not degrees else np.deg2rad(float(lat))
        else:
            self._lon = lon
            self._lat = lat
        self._date = date
        self._city = city
        h_loc = date.hour
        h_gmt = date.astimezone(timezone("GMT")).hour
        self._tzinfo = date.tzinfo
        self._tz = h_loc - h_gmt
        self.on_change = None
        self.__inrad = not degrees

    @property
    def lon(self):
        """
        The longitude of the observer.

        Returns
        -------
        float
        """
        return self._lon if self.__inrad else np.rad2deg(self._lon)

    @lon.setter
    def lon(self, value):
        """

        Parameters
        ----------
        value: float, int, str
        """
        self._lon = float(value) if self.__inrad else np.deg2rad(float(value))
        if self.on_change is not None:
            self.on_change()

    @property
    def lat(self):
        """
        The latitude of the observer.

        Returns
        -------
        float
        """
        return self._lat if self.__inrad else np.rad2deg(self._lat)

    @lat.setter
    def lat(self, value):
        """
        Parameters
        ----------
        value: float, int, str
        """
        self._lat = float(value) if self.__inrad else np.deg2rad(float(value))
        if self.on_change is not None:
            self.on_change()

    @property
    def tzgmt(self):
        """
        The difference in hours from the GMT timezone.

        Returns
        -------
        int
        """
        return self._tz

    @property
    def timezone(self):
        """
        Information about the timezone.

        Returns
        -------
        tzinfo
        """
        return self._tzinfo

    @property
    def date(self):
        """
        The date and time in the current position.

        Returns
        -------
        datetime
        """
        return self._date

    @date.setter
    def date(self, value):
        """
        Parameters
        ----------
        value: datetime
        """
        self._date = value
        if self.on_change is not None:
            self.on_change()

    @property
    def city(self) -> str:
        """
        The closest city to the current location

        Returns
        -------
        str
        """
        return self._city

    def copy(self):
        """
        Creates a copy of the observer.

        Examples
        --------
        >>> o = get_live_observer()  # doctest: +ELLIPSIS
        >>> c = o.copy()
        >>> c == o
        False
        >>> str(o) == str(c)
        True

        Returns
        -------
        Observer
        """
        return copy(self)

    def __copy__(self):
        return Observer(lon=copy(self.lon), lat=copy(self.lat), degrees=not copy(self.__inrad),
                        date=copy(self.date), city=copy(self.city))

    def __repr__(self):
        return "Observer(lon='%.6f', lat='%.6f', %sdate='%s', timezone='%s')" % (
            self.lon, self.lat, ("city='%s', " % self.city) if self._city is not None else "",
            str(self._date), self.timezone)


def get_seville_observer():
    """
    Creates an observer with the properties of Seville in Spain and with the current date and time.

    - latitude: 37.392509
    - longitude: -5.983877

    Examples
    --------
    >>> get_seville_observer()  # doctest: +ELLIPSIS
    Observer(lon='-5.983877', lat='37.392509', city='Seville', date='...', timezone='None')

    Returns
    -------
    Observer
    """
    sev = Observer()
    sev.lat = '37.392509'
    sev.lon = '-5.983877'
    sev._city = "Seville"

    return sev


def get_edinburgh_observer():
    """
    Creates an observer with the properties of Edinburgh in Scotland and with the current date and time.

    - latitude: 55.946388
    - longitude: -3.200000

    Examples
    --------
    >>> get_edinburgh_observer()  # doctest: +ELLIPSIS
    Observer(lon='-3.200000', lat='55.946388', city='Edinburgh', date='...', timezone='None')

    Returns
    -------
    Observer
    """
    edi = Observer()
    edi.lat = '55.946388'
    edi.lon = '-3.200000'
    edi._city = "Edinburgh"

    return edi


def get_live_observer():
    """
    Creates an observer based on your current location, date and time.

    Examples
    --------
    >>> get_live_observer()  # doctest: +ELLIPSIS
    Observer(lon='...', lat='...', city='...', date='...', timezone='...')

    Returns
    -------
    Observer
    """
    import requests
    import json

    send_url = "http://api.ipstack.com/check?access_key=9d6917440142feeccd73751e2f2124dc"
    geo_req = requests.get(send_url)
    geo_json = json.loads(geo_req.text)

    obs = Observer(lon=geo_json['longitude'], lat=geo_json['latitude'], date=datetime.now())
    obs._city = geo_json['city']

    return obs
