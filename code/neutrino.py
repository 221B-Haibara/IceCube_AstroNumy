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

        self.density_nocos = gaussian_kde(np.array(pass_sr["cos(ImpLF_zen)"]), weights=np.array(pass_sr["wE3"]))
        self.density_nocos.set_bandwidth(0.04)
        self.x_nocos = m[1]

        bins = np.linspace(0, 180, 180)
        m = np.histogram(np.rad2deg(np.arccos(pass_sr["cos(ImpLF_zen)"])),
                         weights=np.array(pass_sr["wE3"]) / 0.457845099495,
                         bins=bins, normed=True)

        self.density = gaussian_kde(np.rad2deg(np.arccos(np.array(pass_sr["cos(ImpLF_zen)"]))),
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
        :return: [(declination, right_ascension), (...),  ...]
        """
        theta = self.zenith2declination(self.density.resample(size=size)[0])
        phi = np.random.uniform(0, 2 * np.pi, size=size)
        data = np.array([theta, phi])
        return data.T

    @staticmethod
    def zenith2declination(theta):
        return theta - 90.


if __name__ == "__main__":
    nudist = NuDist()
