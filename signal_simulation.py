import numpy as np
import matplotlib.pyplot as plt

# create a transmitted signal
tx_doa = np.array([20, 30]) # in degrees
theta = tx_doa * np.pi / 180 # in radians

sample_rate = 1e6
samples = 10000

t = np.arange(samples) / sample_rate
f_tone = 0.02e6
tx_base = np.exp(2j * np.pi * f_tone * t).reshape(-1, 1)
tx = np.repeat(tx_base, len(tx_doa), axis=1).T

# plt.plot(np.asarray(tx_base).squeeze().real[0:200])
# plt.title('Transmitted Signal')
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.show()

# create antenna array
d = 0.5 # array spacing in terms of wavelength
N = 8 # number of elements for ULA

# Steering vector matrix
k = np.arange(N).reshape(-1, 1)
sin_theta = np.sin(theta).reshape(-1, 1).T

A = np.exp(-2j * np.pi * d * k * sin_theta)

# Get received signal
noise = np.random.randn(N, samples) + 1j * np.random.randn(N, samples)
sigma = 0.05 # noise standard deviation (depends on antenna)

rx = (A @ tx) # matrix multiply the steering vector matrix and tx
rx = rx + sigma * noise # received signal with noise

# print(k.shape)
# print(sin_theta.shape)
# print(A.shape)
# print("transmitted signals:", tx.shape)
# print("received signals", rx.shape)

# for i in range(N):
#     plt.plot(np.asarray(rx[i,:]).squeeze().real[0:200])

# plt.title('Received Signals for each Element')
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.show()