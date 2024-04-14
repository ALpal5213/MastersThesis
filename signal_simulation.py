import numpy as np
import matplotlib.pyplot as plt

# create directions of arrival
tx_doa = np.array([20, 45]) # in degrees

# create antenna array properties
d = 0.5 # array spacing in terms of wavelength
numElements = 4
sigma = 1

theta = tx_doa * np.pi / 180 # in radians

sample_rate = 1e6
samples = 10000
t = np.arange(samples) / sample_rate
t = t.reshape(1,-1) # turn into row vector

f_base = 0.03e6
f_tone = np.array([])

for i in range(len(tx_doa)):
    arr = np.array([f_base + i * 10000])
    f_tone = np.append(f_tone, arr)

f_tone = f_tone.reshape(-1, 1) # turn into column vector

tx = np.exp(2j * np.pi * f_tone * t)

# Steering vector matrix
k = np.arange(numElements).reshape(-1, 1)
sin_theta = np.sin(theta)

A = np.exp(-2j * np.pi * d * k * sin_theta)

# Get received signal
rx = (A @ tx) 
# print(rx.shape)


noise = sigma * (np.random.randn(numElements, samples) + 1j * np.random.randn(numElements, samples)) / np.sqrt(2)
print(np.var(noise))

rx = rx + noise
# print(rx.shape)

# for i in range(numElements):
#     plt.plot(np.asarray(rx[i,:]).squeeze().real[0:200])
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()