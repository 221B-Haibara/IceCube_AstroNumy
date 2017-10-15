import pandas as pd
import numpy as np
from kde import gaussian_kde
import matplotlib.pyplot as plt


class NuDist:
    def __init__(self, ehefile="../Resources/EHE/EHE_effective.csv"):
        pass_sr = pd.read_csv(ehefile)
        bins = np.linspace(-1, 1, 180)
        m = np.histogram(pass_sr["cos(ImpLF_zen)"], weights=np.array(pass_sr["wE3"]) / 0.457845099495, bins=bins,
                         normed=True)

        self.density_nocos = gaussian_kde(np.rad2deg(np.arccos(np.array(pass_sr["cos(ImpLF_zen)"]))), weights=np.array(pass_sr["wE3"]))
        self.density_nocos.set_bandwidth(0.04)
        self.x_nocos = m[1]

        bins = np.linspace(0, 180, 180)
        m = np.histogram(np.rad2deg(np.arccos(pass_sr["cos(ImpLF_zen)"])),
                         weights=np.array(pass_sr["wE3"]) / 0.457845099495,
                         bins=bins, normed=True)

        self.density = gaussian_kde(np.array(pass_sr["cos(ImpLF_zen)"]),
                                    weights=np.array(pass_sr["wE3"]))
        self.density.set_bandwidth(0.05)
        self.x = m[1]

    def show_fits(self):
        plt.figure()
        plt.plot(self.x_nocos, self.density_nocos(self.x_nocos), color="red", label="kde(bright)")
        plt.show()

        plt.figure()
        plt.plot(self.x, self.density(self.x), color="red", label="kde(bright)")
        plt.show()

    def get_random(self, size):
        """
        Draw random neutrino
        :param size: how many neutrinos
        :return: [(right_ascension, declination), (...),  ...]
        """
        theta = self.density.resample(size=int(size*1.1))[0] #this is a bit hacky because smooth dist has inifinite tails
        theta = (theta[(theta>=-1) & (theta<=1)])[:size]
        if len(theta) != size:
            raise RuntimeError("Please rerun, random number generation failed (can happen due to tails of smoothed dist)")

        dec = self.zenith2declination(theta)
        ra = np.random.uniform(0, 360, size=size)
        data = np.array([ra, dec])
        return data.T

    @staticmethod
    def zenith2declination(theta):
        return np.rad2deg(np.arccos(theta)) - 90.


if __name__ == "__main__":
    nudist = NuDist()
    print nudist.get_random(10000)
