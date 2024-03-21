import importlib
import signal_simulation
import music
import mvdr
# from music import scan_music, doa_music
# from mvdr import w_mvdr, w_mvdr_test

import matplotlib.pyplot as plt
import numpy as np

importlib.reload(signal_simulation)
importlib.reload(music)
importlib.reload(mvdr)

guessNumSignals = 2
decimalPrecision = 0 # can't go very high

numElements = signal_simulation.numElements
d = signal_simulation.d
rx = signal_simulation.rx
precision = 180 * 10**decimalPrecision + 1

R = rx @ rx.conj().T
Rinv = np.linalg.pinv(R) 

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

# Run MUSIC and MVDR algorithms
results = music.scan_music(thetaScan, guessNumSignals, precision, R, numElements, d)

doas, peaks = music.doa_music(results, thetaScan)

weights = mvdr.w_mvdr(doas, R, numElements, d)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetaScan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.plot(thetaScan[peaks], results[peaks], 'x')
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
plt.show()

# print(doas)
# print("weights", weights.shape)
# print(rx.shape)

results2 = []
for theta_i in thetaScan:
#    w = w_mvdr_test(theta_i, R, numElements, d, guessNumSignals) # 3x1
   w = mvdr.w_mvdr(theta_i, R, numElements, d)
   r_weighted = w.conj().T @ rx # apply weights
   power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results2.append(power_dB)
results2 -= np.max(results2) # normalize

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetaScan, results2) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
plt.show()

# print(thetaScan[np.argmax(results2)] * 180 / np.pi) # 19.99999999999998

# plt.plot(thetaScan*180/np.pi, results2) # lets plot angle in degrees
# plt.xlabel("Theta [Degrees]")
# plt.ylabel("DOA Metric")
# plt.grid()
# plt.show()