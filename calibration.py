import numpy as np
from scipy import signal

def phaseSync(signalData, signalFreq, samples, sampleRate, numElements):
    correctedPhases = np.empty(4, dtype=np.complex64)
    correctedPhases[0] = 1

    for i in range(1, numElements):
        xcorr = signal.correlate(signalData[0], signalData[i])
        diff = samples - np.argmax(xcorr) - 1
        phaseCorrection = np.exp(2j * np.pi * diff * signalFreq / sampleRate)
        correctedPhases[i] = phaseCorrection

    return correctedPhases