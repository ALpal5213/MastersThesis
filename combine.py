import importlib
import signal_simulation
import plots
import music
import mvdr
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(signal_simulation)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)

##############################################

guessNumSignals = 2
decimalPrecision = 0 # can't go very high

numElements = signal_simulation.numElements
d = signal_simulation.d
rx = signal_simulation.rx

precision = 180 * 10**decimalPrecision + 1

R = rx @ rx.conj().T 
Rinv = np.linalg.pinv(R) 

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

##############################################

# Run MUSIC and MVDR algorithms
results = music.scan(thetaScan, guessNumSignals, R, numElements, d)

doas, peaks = music.doa(results, thetaScan)

weights = mvdr.w_mvdr(doas, Rinv, numElements, d)

##############################################

# w = mvdr.w_mvdr(45 * np.pi / 180, R, numElements, d)

# print(doas)
print("weights", weights.shape)
# print("weights[0]", weights[:, 0].shape)
# print("weights2", weights2.shape)
# print(weights)
# print(w)


##############################################

results2 = []
for theta_i in thetaScan:
   w = mvdr.w_mvdr(theta_i, Rinv, numElements, d)
   r_weighted = w.conj().T @ rx # apply weights
   power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results2.append(power_dB)
results2 -= np.max(results2) # normalize

##############################################

sample_rate = 1e6
samples = 10000
t = np.arange(samples) / sample_rate
t = t.reshape(1,-1) # turn into row vector
f_tone = 0.02e6

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

   # r_weighted = weights[:, 1].conj().T @ rx 
   # power_dB = 10*np.log10(np.var(r_weighted))
   # results4.append(power_dB)
results3 -= np.max(results3) # normalize
# results4 -= np.max(results4) # normalize

##############################################

sample_rate = 1e6
samples = 10000
t = np.arange(samples) / sample_rate
t = t.reshape(1,-1) # turn into row vector
f_tone = 0.02e6

tx = np.exp(2j * np.pi * f_tone * t)

print(weights[:, 0].reshape(-1, 1).shape)
print(tx.shape)

t_out = weights[:, 0].reshape(-1, 1) @ tx 

# noise = signal_simulation.noise

sigma = 0.5
noise = sigma * (np.random.randn(numElements, samples) + 1j * np.random.randn(numElements, samples))
t_out = t_out + noise
print(t_out.shape)

R = t_out @ t_out.conj().T 
results5 = music.scan_music(thetaScan, 1, R, numElements, d)


# results5 = []
# for theta_i in thetaScan:
   

#    t_weighted = weights[:, 0].reshape(-1, 1) @ tx 
#    power_dB = 10*np.log10(np.var(t_weighted))
#    results5.append(power_dB)

# results5 -= np.max(results3) # normalize

##############################################

plots.plot_polar(thetaScan, results, peaks=peaks, title="MuSiC")
plots.plot_regular(thetaScan, results, peaks=peaks, title="MuSiC")
# plots.plot_polar(thetaScan, results2)
plots.plot_polar(thetaScan, results3, peaks=peaks, title="MVDR First Direction")
# plots.plot_polar(thetaScan, results4, peaks=peaks, title="MVDR Second Direction")
plots.plot_polar(thetaScan, results5, title="transmit using weights")
print("DoA:", doas * 180 / np.pi)

# for i in range(numElements):
#     plt.plot(np.asarray(t_out[i,:]).squeeze().real[0:200])
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()