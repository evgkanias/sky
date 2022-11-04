from sky.prague import PragueSkyModel
from sky.exceptions import NoPolarisationWarning

import numpy as np


def main(*args):
    sky = PragueSkyModel()
    sky.reset(r"D:\odin\Projects\sky\src\data\PragueSkyModelDatasetGroundInfra.dat")

    print("Radiance")
    print(f"Visibilities: {len(sky.visibilities_rad)}")
    print(f"Albedos: {len(sky.albedos_rad)}")
    print(f"Altitutes: {len(sky.altitudes_rad)}")
    print(f"Elevations: {len(sky.elevations_rad)}")
    print(f"Number of channels: {sky.nb_channels}, "
          f"start: {sky.channel_start:.2f}, "
          f"width: {sky.channel_width:.2f}")
    print("Meta:")
    print(sky.metadata_rad)
    print(f"Dataset: min = {np.min(sky.data_rad):.1f}, max = {np.max(sky.data_rad):.1f}")
    print("")

    print("Transmittance:")
    print(f"d_dim: {sky.d_dim}, a_dim: {sky.a_dim}, rank: {sky.rank_trans}")
    print(f"Visibilities: {len(sky.visibilities_trans)}")
    print(f"Altitudes: {len(sky.altitudes_trans)}")
    print(f"Dataset U: min = {np.min(sky.data_trans_u):.1f}, max = {np.max(sky.data_trans_u):.1f}")
    print(f"Dataset V: min = {np.min(sky.data_trans_v):.1f}, max = {np.max(sky.data_trans_v):.1f}")
    print("")

    pol = sky.data_pol is not None
    print(f"Polarisation: {pol}")
    if pol:
        print("Meta:")
        print(sky.metadata_pol)
        print(f"Dataset: min = {np.min(sky.data_pol):.1f}, max = {np.max(sky.data_pol):.1f}")
        print("")


if __name__ == "__main__":
    import sys

    main(*sys.argv)
