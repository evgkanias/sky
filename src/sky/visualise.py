import numpy as np
import matplotlib.pyplot as plt


def visualise_luminance(y, elevation, azimuth):
    """
    Plots the sky luminance.

    Parameters
    ----------
    y: np.ndarray
        the luminance.
    elevation: np.ndarray
        the elevation of the sky elements.
    azimuth: np.ndarray
        the azimuth of the sky elements.
    """
    plt.figure("Luminance", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.scatter(azimuth, np.cos(elevation), s=20, c=y, marker='.', cmap='Blues_r', vmin=0, vmax=6)
    ax.set_ylim([0, 1])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()


def visualise_degree_of_polarisation(sky):
    """
    Plots the degree of polarisation in the sky.

    Parameters
    ----------
    sky: Sky
        the sky model
    """
    import matplotlib.pyplot as plt

    plt.figure("degree-of-polarisation", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = sky.theta_s, sky.phi_s
    ax.scatter(sky.phi, sky.theta, s=10, c=sky.DOP, marker='.', cmap='Greys', vmin=0, vmax=1)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()


def visualise_angle_of_polarisation(sky):
    """
    Plots the angle of polarisation in the sky.

    Parameters
    ----------
    sky: Sky
        the sky model
    """
    plt.figure("angle-of-polarisation", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = sky.theta_s, sky.phi_s
    ax.scatter(sky.phi, sky.theta, s=10, c=sky.AOP, marker='.', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(env.phi_t + np.pi, env.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()
