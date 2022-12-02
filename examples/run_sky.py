from skylight import PragueSky, AnalyticalSky, UniformSky, SkyInfo
from skylight.render import apply_exposure, spectrum2rgb, SPECTRUM_WAVELENGTHS

import skylight.geometry as geo

from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

import os

ele_transform = np.cos
# ele_transform = lambda x : x


def get_rgb(v):
    if v.shape[0] < 2:
        return np.nanmean(v, axis=0)
    elif v.shape[0] != SPECTRUM_WAVELENGTHS.shape[0]:
        v = interp1d(si.wavelengths, v, axis=0, fill_value='extrapolate')(SPECTRUM_WAVELENGTHS)

    return spectrum2rgb(v).T


def visualise(si, exposure=0.0, title="light-properties"):
    """
    Plots the angle of polarisation in the skylight.

    Parameters
    ----------
    si: SkyInfo
        the skylight instance.
    exposure: float
        the exposure
    title: str
        the title of the figure.
    """

    plt.figure(title, figsize=(13.5, 4.5))

    y = get_rgb(si.sky_radiance) if si.sky_radiance.ndim > 1 else si.sky_radiance
    y = np.clip(apply_exposure(np.maximum(y, 0), exposure=exposure), 0, 1)

    p = get_rgb(si.degree_of_polarisation) if si.degree_of_polarisation.ndim > 1 else si.degree_of_polarisation
    p = np.clip(apply_exposure(np.maximum(p, 0), exposure=exposure), 0, 1)

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
    Plots the skylight luminance.

    Parameters
    ----------
    v: np.ndarray[float]
        the value to plot.
    elevation: np.ndarray[float]
        the elevation of the skylight elements in rad.
    azimuth: np.ndarray[float]
        the azimuth of the skylight elements in rad.
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

    kwargs = {}
    if v.ndim == 1:
        kwargs["cmap"] = cmap
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    ax.scatter(azimuth, ele_transform(elevation), s=20, c=v, marker='.', **kwargs)

    ax.set_ylim([ele_transform(np.pi/2), ele_transform(0)])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])


if __name__ == "__main__":
    nb_rows = 90
    sun_theta, sun_phi = np.deg2rad(30), np.deg2rad(180)

    azimuth, elevation = [], []
    for i, se in enumerate(np.linspace(0, 1, nb_rows, endpoint=False)[::-1]):
        j = nb_rows - i
        for a in np.linspace(0, 2 * np.pi, j * 4, endpoint=False) + (j % 2) * 2 * np.pi / (j * 4):
            elevation.append(np.arccos(se))
            azimuth.append(a)

    elevation = np.array(elevation)
    azimuth = np.array(azimuth)
    ori = geo.sph2ori(elevation, azimuth)

    # PRAGUE SKY MODEL
    sky = PragueSky(sun_theta, sun_phi)
    sky.initialise(os.path.join("..", "data", "PragueSkyModelDatasetGroundInfra.dat"))
    si = sky(ori, wavelengths=SPECTRUM_WAVELENGTHS)

    visualise(si, exposure=-6.0, title="prague skylight")

    # ANALYTICAL MODEL
    sky = AnalyticalSky(sun_theta, sun_phi)
    si = sky(ori, wavelengths=SPECTRUM_WAVELENGTHS)

    visualise(si, exposure=-6.0, title="analytical skylight")

    # UNIFORM MODEL
    sky = UniformSky(sun_theta, sun_phi)
    si = sky(ori, wavelengths=SPECTRUM_WAVELENGTHS)

    visualise(si, exposure=-6.0, title="uniform skylight")

    plt.show()
