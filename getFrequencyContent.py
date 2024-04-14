import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def getFrequencyContent(signal, samples, sampleRate):
    samplePeriod = 1 / sampleRate

    freq = fftpack.fft(signal)
    xf = np.linspace(0, 1 / (2 * samplePeriod), samples // 2)

    fig, ax = plt.subplots(2)
    ax[0].plot(signal)
    ax[1].plot(xf, 2 / samples * np.abs(freq[:samples // 2]))

    plt.show(block=False)