"""

"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from sky.simulation import UniformSky, Sky, spectrum_influence
from sky.visualise import visualise_luminance, visualise_angle_of_polarisation, visualise_degree_of_polarisation
from sky.ephemeris import Sun
from sky.observer import Observer, get_live_observer
