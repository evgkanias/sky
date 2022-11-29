from sky import PragueSky, AnalyticalSky, UniformSky, SkyInstance
from sky.prague.render import apply_exposure

import sky.geometry as geo

import numpy as np
import matplotlib.pyplot as plt

import os

ele_transform = np.cos
# ele_transform = lambda x : x


def visualise(si, title="light-properties"):
    """
    Plots the angle of polarisation in the sky.

    Parameters
    ----------
    si: SkyInstance
        the sky instance.
    title: str
        the title of the figure.
    """

    plt.figure(title, figsize=(13.5, 4.5))

    y = apply_exposure(si.sky_radiance, exposure=2)
    p = si.degree_of_polarisation
    a = si.angle_of_polarisation

    elevation = geo.xyz2elevation(si.view_direction)
    azimuth = geo.xyz2azimuth(si.view_direction)
    sun_e = geo.xyz2elevation(si.sun_direction)
    sun_a = geo.xyz2azimuth(si.sun_direction)

    ax = plt.subplot(131, polar=True)
    plot_disc(y, elevation, azimuth, cmap='Blues_r', ax=ax)
    ax.scatter(sun_a, ele_transform(sun_e), s=100, edgecolor='black', facecolor='yellow')

    ax = plt.subplot(132, polar=True)
    plot_disc(p, elevation, azimuth, cmap='Greys', ax=ax)
    ax.scatter(sun_a, ele_transform(sun_e), s=100, edgecolor='black', facecolor='yellow')

    ax = plt.subplot(133, polar=True)
    plot_disc(a, elevation, azimuth, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2, ax=ax)
    ax.scatter(sun_a, ele_transform(sun_e), s=100, edgecolor='black', facecolor='yellow')

    plt.tight_layout()


def plot_disc(v, elevation, azimuth, cmap="Greys_r", vmin=0, vmax=1, ax=None):
    """
    Plots the sky luminance.

    Parameters
    ----------
    v: np.ndarray[float]
        the value to plot.
    elevation: np.ndarray[float]
        the elevation of the sky elements in rad.
    azimuth: np.ndarray[float]
        the azimuth of the sky elements in rad.
    cmap: str
        the colour map.
    vmin: float
        minimum value.
    vmax: float
        maximum value.
    ax: optional
        the axis to draw the plot.
    """
    if ax is None:
        ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.scatter(azimuth, ele_transform(elevation), s=20, c=v, marker='.', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_ylim([ele_transform(np.pi/2), ele_transform(0)])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])


if __name__ == "__main__":
    nb_rows = 90
    sun_theta, sun_phi = np.deg2rad(30), np.deg2rad(180)

    azimuth, elevation = [], []
    for e in np.linspace(0, np.pi/2, nb_rows, endpoint=False):
        i = int(np.round(nb_rows * (1 - 2 * e / np.pi)))
        for a in np.linspace(0, 2 * np.pi, i * 4, endpoint=False) + (i % 2) * 2 * np.pi / (i * 4):
            elevation.append(e)
            azimuth.append(a)

    elevation = np.array(elevation)
    azimuth = np.array(azimuth)
    ori = geo.sph2ori(elevation, azimuth)

    # PRAGUE SKY MODEL
    sky = PragueSky(sun_theta, sun_phi)
    sky.initialise(os.path.join("..", "data", "PragueSkyModelDatasetGroundInfra.dat"))
    si = sky(ori)

    visualise(si, title="Prague Sky")

    # ANALYTICAL MODEL
    sky = AnalyticalSky(sun_theta, sun_phi)
    si = sky(ori)

    visualise(si, title="Analytical Sky")

    # UNIFORM MODEL
    sky = UniformSky(sun_theta, sun_phi)
    si = sky(ori)

    visualise(si, title="Uniform Sky")

    plt.show()
