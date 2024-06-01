import numpy as np

def scan(thetaScan, R, numElements, d, numSignals):
    eigenvalues, eigenvectors = np.linalg.eig(R)

    indexesOfSortedEigenvalues = np.argsort(np.abs(eigenvalues))
    sortedEigenvectors = eigenvectors[:, indexesOfSortedEigenvalues]

    Vn = np.zeros((numElements, numElements - numSignals),            
        dtype=np.complex64) # noise Eigenvectors matrix

    for i in range(numElements - numSignals):
        Vn[:, i] = sortedEigenvectors[:, i]

    spectrum = np.array([])

    for theta in thetaScan:
        a = np.exp(-2j * np.pi * d * np.arange(numElements) *                      np.sin(theta)) 
        a = a.reshape(-1,1)
        P = 1 / (a.conj().T @ Vn @ Vn.conj().T @ a)
        Pdb = 10 * np.log10(np.abs(P.squeeze())) # convert to dB
        spectrum = np.append(spectrum, Pdb)

    spectrum -= np.min(spectrum)

    return spectrum