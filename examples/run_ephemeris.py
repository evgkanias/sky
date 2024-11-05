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
    for c, month in [['r', 1],
                     ['y', 3],
                     ['g', 6],
                     ['b', 9],
                     ['c', 12]]:
        d, e, a = [], [], []
        for h in range(24):
            sun.obs.date = datetime(2024, month, 21, h, tzinfo=timezone('GMT'))
            sun.update()
            d.append(sun.obs.date.hour)
            e.append(sun.alt)
            a.append(sun.az)
            print((sun.sunset - sun.sunrise).total_seconds() / 3600)

        d, e, a = np.array(d), np.array(e), np.array(a)

        plt.plot(d, e, '%s.-' % c)
    plt.xlim([0, 24])
    plt.ylim([0, np.pi/2])

    plt.show()
