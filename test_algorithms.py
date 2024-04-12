import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import importlib
import signal_simulation
import music
import mvdr
import plots
import time

importlib.reload(signal_simulation)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)

#################################################################

guessNumSignals = 2
decimalPrecision = 0 # can't go very high

numElements = signal_simulation.numElements
d = signal_simulation.d
rx = signal_simulation.rx
samples = signal_simulation.samples
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

weights = mvdr.calc_weights(doas, Rinv, numElements, d)

t1 = time.time()

#################################################################

plots.plot_polar(thetaScan, results, peaks=peaks, title="MuSiC Scan")
plots.plot_regular(thetaScan, results, peaks=peaks, title="MuSiC Scan")

print("DoA (degrees):", doas * 180 / np.pi)

print(weights)

print(t1 - t0)