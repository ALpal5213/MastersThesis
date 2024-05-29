import numpy as np
from scipy import fftpack, signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def getFrequencyContent(signalData, sampleRate, plot=False):
    f, Pxx_den = signal.periodogram(signalData, sampleRate)
    freq = f[np.argmax(Pxx_den)]

    if plot == True:
        fig, ax = plt.subplots(2)

        ax[0].plot(signalData)
        ax[1].plot(f, Pxx_den)
        
        plt.show(block=False)

    return freq

