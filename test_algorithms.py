import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import importlib
import signalSim
import music
import mvdr
import plots
import time
import getFrequencyContent
import calibration

importlib.reload(signalSim)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)
importlib.reload(getFrequencyContent)
importlib.reload(calibration)


#################################################################

guessNumSignals = 2
decimalPrecision = 0 # can't go very high

precision = 180 * 10**decimalPrecision + 1

t0 = time.time()

doas = np.array([0, 15]) * np.pi / 180
f_b = np.array([40000, 50000])
samples = 10000
sampleRate = 30720000
d = 0.5
numElements = 4
sigma = 0.5#1 / np.sqrt(2)


s = signalSim.generateSignals(f_b, samples, sampleRate)
A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
    1j * np.random.normal(0, sigma, size=[numElements, samples])

y = 1 * ((A @ s))

# signalSim.plotSignal(y, samples, sampleRate, numElements)

R = (y @ y.conj().T) / samples
Rinv = np.linalg.pinv(R) 

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

#################################################################

spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)


peaks, _ = signal.find_peaks(spectrum, height=5) # Need to modify
doas = thetaScan[peaks]

print("DoA (degrees):", doas * 180 / np.pi)
plots.plot_polar(thetaScan, spectrum, peaks=peaks, title="MuSiC Scan")

# print(doas)
weights = mvdr.calc_weights(doas[0], Rinv, numElements, d)
# print(weights)

#################################################################
# Apply weights for Receiving
weights_ones = np.array([0.25, 0.25, 0.25, 0.25]).reshape(-1, 1)
rx_new = weights_ones.conj().T @ y
rx_new = rx_new.flatten()


weightst = np.exp(-2j * np.pi * d * np.arange(numElements) * np.sin(doas[0]))
rx_summedAndWeighted = weights.conj().T @ y
rx_summedAndWeighted = rx_summedAndWeighted.flatten()

#################################################################

# getFrequencyContent.getFrequencyContent(y[0], sampleRate, plot=True)
# freq = getFrequencyContent.getFrequencyContent(rx_summedAndWeighted, sampleRate, plot=True)
# print(freq)

plt.pause(20)