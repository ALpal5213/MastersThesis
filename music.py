# import libraries
import matplotlib.pyplot as plt
import numpy as np
import imp
import signal_simulation
imp.reload(signal_simulation)

numSignals = len(signal_simulation.tx_doa) # expected signals 
numElements = signal_simulation.N
d = signal_simulation.d
rx = signal_simulation.rx

R = rx @ rx.conj().T
eigenValues, eigenVectors = np.linalg.eig(R)

indexesOfSortedEigenValues = np.argsort(np.abs(eigenValues))
sortedEigenVectors = eigenVectors[:, indexesOfSortedEigenValues]

noiseMatrix = np.zeros((numElements, numElements - numSignals), dtype=np.complex64)

for i in range(numElements - numSignals):
   noiseMatrix[:, i] = sortedEigenVectors[:, i]

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # -180 to +180 degrees
results = []

for theta_i in theta_scan:
    a = np.exp(-2j * np.pi * d * np.arange(numElements) * np.sin(theta_i)) # array factor
    a = a.reshape(-1,1)
    metric = 1 / (a.conj().T @ noiseMatrix @ noiseMatrix.conj().T @ a) # The main MUSIC equation
    metric = np.abs(metric.squeeze()) # take magnitude
    metric = 10*np.log10(metric) # convert to dB
    results.append(metric)

results /= np.max(results)

sortedEigenValues = np.flip(eigenValues[indexesOfSortedEigenValues])
plt.plot(10 * np.log10(np.abs(sortedEigenValues)),'.-')
plt.show()

print(eigenValues.shape)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
plt.show()