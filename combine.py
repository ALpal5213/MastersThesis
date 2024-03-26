import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import importlib
import signal_simulation
import music
import mvdr
import plots

importlib.reload(signal_simulation)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)

##############################################

guessNumSignals = 2
decimalPrecision = 1 # can't go very high

numElements = signal_simulation.numElements
d = signal_simulation.d
rx = signal_simulation.rx
samples = signal_simulation.samples
sigma = signal_simulation.sigma
noise = signal_simulation.noise

precision = 180 * 10**decimalPrecision + 1

R = rx @ rx.conj().T 
Rinv = np.linalg.pinv(R) 

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

##############################################

# Run MUSIC and MVDR algorithms
results = music.scan(thetaScan, R, numElements, d, guessNumSignals)

peaks, _ = signal.find_peaks(results, height=-2)
doas = thetaScan[peaks]

weights = mvdr.calc_weights(doas, Rinv, numElements, d)

##############################################

results2 = mvdr.scan(thetaScan, Rinv, numElements, d, rx)

##############################################

# sample_rate = 1e6
# samples = signal_simulation.samples
# t = np.arange(samples) / sample_rate
# t = t.reshape(1,-1) # turn into row vector
# f_tone = 0.03e6

# tx = np.exp(2j * np.pi * f_tone * t)

# results3 = []
# results4 = []
# for theta_i in thetaScan:
#    k = np.arange(numElements).reshape(-1, 1)
#    sin_theta = np.sin(theta_i)
#    A = np.exp(-2j * np.pi * d * k * sin_theta) 

#    rx = (A @ tx) 
#    noise = signal_simulation.noise
#    rx = rx + noise

#    r_weighted = weights[:, 0].conj().T @ rx 
#    power_dB = 10*np.log10(np.var(r_weighted))
#    results3.append(power_dB)

#    # r_weighted = weights[:, 1].conj().T @ rx 
#    # power_dB = 10*np.log10(np.var(r_weighted))
#    # results4.append(power_dB)
# results3 -= np.max(results3) # normalize
# # results4 -= np.max(results4) # normalize

##############################################

sample_rate = 1e6
# samples = signal_simulation.samples
t = np.arange(samples) / sample_rate
t = t.reshape(1,-1) # turn into row vector
f_tone = 0.03e6

tx = np.exp(2j * np.pi * f_tone * t)

# print(weights[:, 0].reshape(-1, 1).shape)
# print(weights.shape)

t_out = weights[:, 0].reshape(-1, 1) @ tx 

# noise = signal_simulation.noise

# sigma = 0.5
# noise = sigma * (np.random.randn(numElements, samples) + 1j * np.random.randn(numElements, samples)) / np.sqrt(2)
t_out = t_out + noise
# print("t_out:", t_out.shape)

R = t_out @ t_out.conj().T 
Rinv = np.linalg.pinv(R) 

# results5 = music.scan(thetaScan, R, numElements, d, 1)
results6 = mvdr.scan(thetaScan, Rinv, numElements, d, t_out)

##############################################

plots.plot_polar(thetaScan, results, peaks=peaks, title="MuSiC Scan")
# plots.plot_regular(thetaScan, results, peaks=peaks, title="MuSiC Scan")
plots.plot_polar(thetaScan, results2, title="MVDR Scan")
# plots.plot_polar(thetaScan, results3, peaks=peaks, title="MVDR First Direction")
# plots.plot_polar(thetaScan, results4, peaks=peaks, title="MVDR Second Direction")
# plots.plot_polar(thetaScan, results5, title="transmit using weights (check with MUSIC)")
# plots.plot_polar(thetaScan, results6, title="transmit using weights (check with MVDR)")
print("DoA (degrees):", doas * 180 / np.pi)