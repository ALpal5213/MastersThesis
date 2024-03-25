import numpy as np
from scipy.signal import find_peaks

# Perform MUSIC algorithm and output an array of angles
def scan(thetaScan, numSignals, R, numElements, d):
    eigenValues, eigenVectors = np.linalg.eig(R)

    indexesOfSortedEigenValues = np.argsort(np.abs(eigenValues))
    sortedEigenVectors = eigenVectors[:, indexesOfSortedEigenValues]

    noiseMatrix = np.zeros((numElements, numElements - numSignals), dtype=np.complex64)

    for i in range(numElements - numSignals):
        noiseMatrix[:, i] = sortedEigenVectors[:, i]

    results = []

    for theta_i in thetaScan:
        a = np.exp(-2j * np.pi * d * np.arange(numElements) * np.sin(theta_i)) # array factor
        a = a.reshape(-1,1)
        metric = 1 / (a.conj().T @ noiseMatrix @ noiseMatrix.conj().T @ a) # The main MUSIC equation
        metric = np.abs(metric.squeeze()) # take magnitude
        metric = 10 * np.log10(metric) # convert to dB
        results.append(metric)

    results /= np.max(results)

    return results

def doa(results, thetaScan):
    peaks, _ = find_peaks(results, height=0)
    doas = thetaScan[peaks]

    return doas, peaks