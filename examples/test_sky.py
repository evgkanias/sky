from sky import Sky, visualise_luminance

from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt


flat = True
nb_rows = 90

azimuth, elevation = [], []
for e in np.linspace(0, np.pi/2, nb_rows, endpoint=False):
    i = int(np.round(nb_rows * (1 - 2 * e / np.pi)))
    for a in np.linspace(0, 2 * np.pi, i * 4, endpoint=False) + (i % 2) * 2 * np.pi / (i * 4):
        elevation.append(e)
        azimuth.append(a)

elevation = np.array(elevation)
azimuth = np.array(azimuth)

# ori = R.from_euler('ZY', np.vstack([azimuth, elevation]).T, degrees=False)
ori = R.from_euler('ZY', np.vstack([azimuth, elevation]).T, degrees=False)
sky = Sky(np.deg2rad(40), np.pi)

y, p, a = sky(ori)
# y = np.square(r[..., 1])
# print(r.min(), r.max())
print(y.max(), y.min())

visualise_luminance(y, elevation=elevation, azimuth=azimuth)
