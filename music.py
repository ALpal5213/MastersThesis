import numpy as np

# Perform MUSIC algorithm and output an array of angles
def scan(thetaScan, R, numElements, d, numSignals):
    eigenValues, eigenVectors = np.linalg.eig(R)

    indexesOfSortedEigenValues = np.argsort(np.abs(eigenValues))
    sortedEigenVectors = eigenVectors[:, indexesOfSortedEigenValues]

    noiseMatrix = np.zeros((numElements, numElements - numSignals), dtype=np.complex64)

    for i in range(numElements - numSignals):
        noiseMatrix[:, i] = sortedEigenVectors[:, i]

    spectrum = np.array([])

    for theta_i in thetaScan:
        a = np.exp(-2j * np.pi * d * np.arange(numElements) * np.sin(theta_i)) 
        a = a.reshape(-1,1)

        P = 1 / (a.conj().T @ noiseMatrix @ noiseMatrix.conj().T @ a)
        Pdb = 10 * np.log10(np.abs(P.squeeze())) # convert to dB

        spectrum = np.append(spectrum, Pdb)

    spectrum -= np.min(spectrum)

    return spectrum