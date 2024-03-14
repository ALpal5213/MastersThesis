import numpy as np
import matplotlib.pyplot as plt

# create a transmitted signal
tx_doa = np.array([20, -20, 50, 40]) # in degrees
theta = tx_doa * np.pi / 180 # in radians

sample_rate = 1e6
samples = 10000
t = np.arange(samples) / sample_rate
t = t.reshape(1,-1) # turn into row vector

f_base = 2e9
f_tone = np.array([])

for i in range(len(tx_doa)):
    arr = np.array([f_base + i * 1])
    f_tone = np.append(f_tone, arr)

f_tone = f_tone.reshape(-1, 1) # turn into column vector

tx = np.exp(2j * np.pi * f_tone @ t)

print(t.reshape(-1,1).T.shape)
print(f_tone.shape)
print((f_tone @ t.reshape(-1, 1).T).shape)
print(tx.shape)

plt.plot(tx[0][:200]) # lets plot angle in degrees
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# create antenna array
d = 0.5 # array spacing in terms of wavelength
N = 10 # number of elements for ULA

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
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()