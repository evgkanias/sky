"""
Package that allows reading the binary data files.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2022, Insect Robotics Group," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias", "Petr Vévoda", "Alexander Wilkie"]
__license__ = "GPLv3+"
__version__ = "v1.0-beta"
__maintainer__ = "Evripidis Gkanias"

import numpy as np
import struct


SEEK_BEGIN = 0  # sets the reference point at the beginning of the file
SEEK_CUR = 1  # sets the reference point at the current location
SEEK_END = 2  # sets the reference point at the end of the file

BYTE_ORDER = '<'

USHORT_SIZE = 2
INT_SIZE = 4
FLOAT_SIZE = 4
DOUBLE_SIZE = 8

USHORT_CHAR = 'H'
INT_CHAR = 'i'
FLOAT_CHAR = 'f'
DOUBLE_CHAR = 'd'

USHORT_TYPE = BYTE_ORDER + 'u2'
INT_TYPE = BYTE_ORDER + 'i4'
FLOAT2_TYPE = BYTE_ORDER + 'f2'
FLOAT4_TYPE = BYTE_ORDER + 'f4'
DOUBLE_TYPE = BYTE_ORDER + 'f8'


def read_list(handle, structure: str):
    nb_bytes = (
            structure.count(USHORT_CHAR) * USHORT_SIZE +
            structure.count(INT_CHAR) * INT_SIZE +
            structure.count(FLOAT_CHAR) * FLOAT_SIZE +
            structure.count(DOUBLE_CHAR) * DOUBLE_SIZE
    )
    bytes_read = handle.read(nb_bytes)
    return list(struct.unpack(BYTE_ORDER + structure, bytes_read))


def read_ushort_list(handle, size=None):
    if size is None:
        size = read_int(handle)
    return np.array(
        list(struct.unpack(BYTE_ORDER + USHORT_CHAR * size, handle.read(USHORT_SIZE * size))), dtype=USHORT_TYPE)


def read_float_list(handle, size=None):
    if size is None:
        size = read_int(handle)
    return np.array(
        list(struct.unpack(BYTE_ORDER + FLOAT_CHAR * size, handle.read(FLOAT_SIZE * size))), dtype=FLOAT4_TYPE)


def read_double_list(handle, size=None):
    if size is None:
        size = read_int(handle)
    return np.array(
        list(struct.unpack(BYTE_ORDER + DOUBLE_CHAR * size, handle.read(DOUBLE_SIZE * size))), dtype=DOUBLE_TYPE)


def read_ushort(handle):
    val, = struct.unpack(BYTE_ORDER + USHORT_CHAR, handle.read(USHORT_SIZE))
    return val


def read_int(handle):
    val, = struct.unpack(BYTE_ORDER + INT_CHAR, handle.read(INT_SIZE))
    return val


def read_float(handle):
    val, = struct.unpack(BYTE_ORDER + FLOAT_CHAR, handle.read(FLOAT_SIZE))
    return val


def read_double(handle):
    val, = struct.unpack(BYTE_ORDER + DOUBLE_CHAR, handle.read(DOUBLE_SIZE))
    return val
