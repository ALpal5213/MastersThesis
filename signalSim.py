import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def generateSignals(f_b, samples, sampleRate):
    f_b = f_b.reshape(-1,1)
    t = (np.arange(samples) / sampleRate).reshape(1,-1)
    
    s = np.exp(2j * np.pi * f_b * t)

    return s

def generateSteeringMatrix(doas, d, numElements):
    k = np.arange(numElements).reshape(-1,1)
    sinTheta = np.sin(doas)

    A = np.exp(-2j * np.pi * d * k * sinTheta)

    return A

def plotSignal(signal, samples, sampleRate, numElements, title=None, plot=True):
    samplePeriod = 1 / sampleRate

    freq = fftpack.fft(signal[0])
    xf = np.linspace(0, 1 / (2 * samplePeriod), samples // 2)

    res = (1 / samples * np.abs(freq[:samples // 2]))

    if plot == True:
        fig, ax = plt.subplots(2)
        for i in range(numElements):
            ax[0].plot(np.asarray(signal[i,:]).squeeze().real[0:200])
        ax[1].plot(xf[0:500], res[0:500])

        if title == None:
            ax[0].set_title("Signal Received by 4 Antenna Elements")
        else:
            ax[0].set_title(title)
        
        ax[0].set_xlabel("Samples")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend(["n=0", "n=1", "n=2", "n=3"])

        ax[1].set_title("Frequency of Received Signal")
        ax[1].set_xlabel("Frequency (Hz)")

        fig.tight_layout(pad=1)
        plt.show(block=False)

    return res, xf