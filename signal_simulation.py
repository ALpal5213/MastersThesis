import numpy as np
import matplotlib.pyplot as plt

# create a transmitted signal
tx_doa = np.array([20]) # in degrees
theta = tx_doa * np.pi / 180 # in radians

sample_rate = 1e6
samples = 10000

t = np.arange(samples) / sample_rate
f_tone = 0.02e6
tx_base = np.exp(2j * np.pi * f_tone * t).reshape(-1, 1)
tx = np.repeat(tx_base, len(tx_doa), axis=1).T

# print(tx.shape)
# print(np.average(tx) - np.average(tx_base))

plt.plot(tx[0][:200]) # lets plot angle in degrees
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# create antenna array
d = 0.5 # array spacing in terms of wavelength
N = 3 # number of elements for ULA

# Steering vector matrix
k = np.arange(N).reshape(-1, 1)
sin_theta = np.sin(theta)

A = np.exp(-2j * np.pi * d * k * sin_theta)

# print(k)
# print(sin_theta)
print(A.shape)

# Get received signal
noise = np.random.randn(N, samples) + 1j * np.random.randn(N, samples)
sigma = 0.05 # noise standard deviation (depends on antenna)

rx = (A @ tx) # matrix multiply the steering vector matrix and tx
# rx = rx + sigma * noise # received signal with noise

print(rx.shape)

for i in range(N):
    plt.plot(rx[i][:200])
    # plt.plot(np.asarray(rx[i,:]).squeeze().imag[0:200]) 
    # plt.plot(np.asarray(rx[i,:]).squeeze().real[0:200]) 
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()