from skylight.observer import Observer
from skylight.ephemeris import Sun

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pytz import timezone


if __name__ == '__main__':

    obs = Observer(lon=np.deg2rad(0), lat=np.deg2rad(42), date=datetime(2022, 12, 1, 9))
    sun = Sun(obs)

    plt.figure()
    for c, lat in [['r', np.deg2rad(0)],
                   ['y', np.deg2rad(22.5)],
                   ['g', np.deg2rad(45)],
                   ['b', np.deg2rad(67.5)],
                   ['c', np.deg2rad(89)]]:
        sun.obs.lat = lat
        e, a = [], []
        for h in range(24):
            sun.obs.date = datetime(2020, 9, 21, h, tzinfo=timezone('GMT'))
            sun.update()
            e.append(sun.alt)
            a.append(sun.az)

        e, a = np.array(e), np.array(a)

        plt.plot(a, e, '%s.-' % c)
    plt.xlim([0, 2 * np.pi])
    plt.ylim([0, np.pi/2])

    plt.show()
