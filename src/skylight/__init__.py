"""
Package that allows computations of the skylight properties.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

from skylight.sky import UniformSky, AnalyticalSky, PragueSky, SkyBase
from skylight.ephemeris import Sun
from skylight.observer import Observer, get_live_observer
from skylight._static import T_L, MODES, SPECTRUM_STEP, SPECTRUM_WAVELENGTHS, SPECTRUM_CHANNELS
from skylight._static import SPECTRAL_RESPONSE, SPECTRAL_RESPONSE_START, SPECTRAL_RESPONSE_STEP, XYZ2RGB
from skylight._static import PLANET_RADIUS, SAFETY_ALTITUDE, SUN_RADIUS, DIST_TO_EDGE, ATMOSPHERE_WIDTH
from skylight._static import SUN_RAD_START, SUN_RAD_STEP, SUN_RAD_TABLE, SUN_RAD_END
from skylight._static import Parameters, AvailableData, SkyInfo
