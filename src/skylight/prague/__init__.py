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

from .prague import PragueSkyModelManager
from .._static import Parameters, AvailableData, Metadata, AngleParameters
