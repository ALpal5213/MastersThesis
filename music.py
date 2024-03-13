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

print(numSignals)
print(rx.shape)

# for i in range(numElements):
#     plt.plot(rx[i][:200]) # lets plot angle in degrees
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()

R = rx @ rx.conj().T
eigenValues, eigenVectors = np.linalg.eig(R)

print(R.shape)
print(eigenValues.shape)
print(np.abs(eigenValues))

plt.plot(10 * np.log10(np.abs(eigenValues)),'.-')
plt.show()

indexesOfSortedEigenValues = np.argsort(np.abs(eigenValues))

print(indexesOfSortedEigenValues)

sortedEigenVectors = eigenVectors[:, indexesOfSortedEigenValues]
# sortedEVals = eVals[index]

noiseMatrix = np.zeros((numElements, numElements - numSignals), dtype=np.complex64)

for i in range(numElements - numSignals):
   noiseMatrix[:, i] = eigenVectors[:, i]


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

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
plt.show()

# plt.plot(theta_scan * 180 / np.pi, results) # lets plot angle in degrees
# plt.xlabel("Theta [Degrees]")
# plt.ylabel("DOA Metric")
# plt.grid()
# plt.show()

# sigEVecs = sortedEVecs[:, :numSignals]
# noiseEVecs = sortedEVecs[:, numSignals:numElements]

# angles = np.arange(-90, 90.1, 0.01)

# a1 = np.exp(-1j * 2 * np.pi * d * \
#             np.arange(numElements)[:, None] * \
#             np.sin(np.radians(angles)))

# # print(angles.shape)
# # print(a1.shape)

# music_spectrum = np.zeros(a1.shape, dtype='complex_').T
# # print(music_spectrum.shape)

# for k in range(len(angles)):
#     a1_k = a1[:, k].reshape(-1, 1)
#     Qn = noiseEVecs
#     Qnt = np.transpose(noiseEVecs)
#     music_spectrum[k] = (np.transpose(a1_k).dot(a1_k)) / \
#         (np.transpose(a1_k).dot(Qn.dot(Qnt.dot(a1_k))))

# plt.plot(angles, np.abs(music_spectrum))
# plt.grid(True)
# plt.title('MUSIC Spectrum')
# plt.xlabel('Angle in degrees')
# plt.show()

# maxIndex = np.argsort(music_spectrum)
# print(maxIndex.shape)
# print(maxIndex[50])

# print('Output:', angles[maxIndex[0]])
# print('Output:', np.abs(sortedEVals))

# Set static variables
# N = 8; # number of array elements
# f = 5 * 10**6; # 5 MHz, signal frequency
# d = 0.5; # element spacing in wavelength
# pi = np.pi # pi variable


# # Set dummy values
# doas = np.array([0, 30, 50]) * pi / 180
# numSignals = np.size(doas)
# numSymbols = 10000
# s = np.round(r.rand(numSignals, numSymbols))

# # print(f)
# # print(np.linspace(0, N - 1, num=N))
# # print(doas)
# # print(numSignals)
# # print(s)

# ##### INPUTS #####
# # Steering vector matrix
# A = np.exp(-1j * 2 * pi * d * \
#            np.dot(np.transpose(np.matrix(np.linspace(0, N-1, num=N))), \
#            (np.sin(doas).reshape(-1, 1).T))) 

# # print(A.shape)

# # data matrix
# X = A.dot(s) * 10

# R = X.dot(np.transpose(X)) / numSymbols