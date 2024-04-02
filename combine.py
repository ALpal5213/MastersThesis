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
# results = music.scan(thetaScan, R, numElements, d, guessNumSignals)
results = mvdr.scan(thetaScan, Rinv, numElements, d, rx)

peaks, _ = signal.find_peaks(results, height=-2)
doas = thetaScan[peaks]

weights = mvdr.calc_weights(doas, Rinv, numElements, d)

##############################################

print(doas[0] * 180 / np.pi)
print(doas[1] * 180 / np.pi)
# print(weights.shape)
print(doas  * 180 / np.pi)

weights_good = mvdr.calc_weights(doas[0] , Rinv, numElements, d)
weights_i = mvdr.calc_weights(doas[1] , Rinv, numElements, d)

print("weights_good", weights_good.shape)
print("weights_i", weights_i.shape)

tx = signal_simulation.tx
print(tx[1].reshape(1,-1).shape)

a = np.exp(-2j * np.pi * d * np.arange(numElements) * np.sin(doas[1] )).reshape(-1, 1)

rx_i = a @ tx[1].reshape(1,-1)
print(rx_i.shape)

# rx_i = rx_i.conj()

print("num:", (rx_i @ rx_i.conj().T).shape)
print("denom:", (rx_i.conj().T @ rx_i).shape)
# print(().shape)

weights_new = weights_good - rx_i @ np.linalg.pinv(rx_i.conj().T @ rx_i) @ rx_i.conj().T @ weights_good
print("new weights", weights_new.shape)

##############################################

results2 = mvdr.scan(thetaScan, Rinv, numElements, d, rx)

##############################################

sample_rate = 1e6
samples = signal_simulation.samples
t = np.arange(samples) / sample_rate
t = t.reshape(1,-1) # turn into row vector
f_tone = 0.03e6 +200

tx = np.exp(2j * np.pi * f_tone * t)

results3 = []
results4 = []
for theta_i in thetaScan:
   k = np.arange(numElements).reshape(-1, 1)
   sin_theta = np.sin(theta_i)
   A = np.exp(-2j * np.pi * d * k * sin_theta) 

   rx = (A @ tx) 
   noise = signal_simulation.noise
   rx = rx + noise

   r_weighted = weights[:, 0].conj().T @ rx 
   power_dB = 10*np.log10(np.var(r_weighted))
   results3.append(power_dB)

   r_weighted = weights_new.conj().T @ rx 
   power_dB = 10*np.log10(np.var(r_weighted))
   results4.append(power_dB)
results3 -= np.max(results3) # normalize
results4 -= np.max(results4) # normalize

##############################################

plots.plot_polar(thetaScan, results, peaks=peaks, title="MuSiC Scan")
# plots.plot_regular(thetaScan, results, peaks=peaks, title="MuSiC Scan")
plots.plot_polar(thetaScan, results2, title="MVDR Scan")
# plots.plot_regular(thetaScan, results2, title="MVDR Scan")
plots.plot_regular(thetaScan, results3, peaks=peaks, title="MVDR Without Nulling")
plots.plot_regular(thetaScan, results4, peaks=peaks, title="MVDR With Nulling")
plots.plot_polar(thetaScan, results3, peaks=peaks, title="MVDR Without Nulling")
plots.plot_polar(thetaScan, results4, peaks=peaks, title="MVDR With Nulling")

print("DoA (degrees):", doas * 180 / np.pi)