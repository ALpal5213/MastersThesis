import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import importlib
import signal_simulation
import music
import mvdr
import plots
import time
import getFrequencyContent

importlib.reload(signal_simulation)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)
importlib.reload(getFrequencyContent)

#################################################################

guessNumSignals = 2
decimalPrecision = 0 # can't go very high

numElements = signal_simulation.numElements
d = signal_simulation.d
rx = signal_simulation.rx
samples = signal_simulation.samples
sampleRate = signal_simulation.sample_rate
sigma = signal_simulation.sigma
noise = signal_simulation.noise

precision = 180 * 10**decimalPrecision + 1

t0 = time.time()

R = rx @ rx.conj().T 
Rinv = np.linalg.pinv(R) 

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

#################################################################

results = music.scan(thetaScan, R, numElements, d, guessNumSignals)

peaks, _ = signal.find_peaks(results, prominence=1) # Need to modify
doas = thetaScan[peaks]
print("DoA (degrees):", doas * 180 / np.pi)
weights = mvdr.calc_weights(doas, Rinv, numElements, d)

t1 = time.time()

#################################################################
# Apply weights for Receiving

print(rx.shape)
print(weights.shape)
print(weights[:,0].reshape(-1, 1).shape)

weights_ones = np.array([1, 1, 1, 1]).reshape(-1, 1)
rx_new = weights_ones.conj().T @ rx
rx_new = rx_new.flatten()
print(weights_ones.shape)

rx_summedAndWeighted = weights[:,0].conj().T @ rx
rx_summedAndWeighted = rx_summedAndWeighted.flatten()

#################################################################

getFrequencyContent.getFrequencyContent(rx_new[0:500], 500, sampleRate)
getFrequencyContent.getFrequencyContent(rx_summedAndWeighted[0:500], 500, sampleRate)

plt.pause(100)