import adi
import importlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal

import plots
import music
import mvdr

importlib.reload(plots)
importlib.reload(music)
importlib.reload(mvdr)


# Connect to FMCOMMS5
sdr = adi.FMComms5(uri="ip:analog")

sdr.sample_rate = 1e6
fs = int(sdr.sample_rate)

sdr.rx_buffer_size = 10000 # 1024 is default

# Receiver settings
sdr.rx_lo = 5800000000
sdr.rx_lo_chip_b = 5800000000

# Transmitter settings
sdr.tx_lo = 5800000000
sdr.tx_lo_chip_b = 5800000000

# 0 = Channel TX1A_A
# 1 = Channel TX2A_A
# 2 = Channel TX1A_B
# 3 = Channel TX2A_B
sdr.dds_single_tone(30000, 0.9, 1) 

# Get receiver data
rx_data = sdr.rx() # must be called for new data

print("RX LO %s" % (sdr.rx_lo))

print(rx_data[0].shape)

f0, psd0 = signal.periodogram(rx_data[0], fs)
f1, psd1 = signal.periodogram(rx_data[1], fs)

plt.semilogy(f0, psd0)
plt.semilogy(f1, psd1)
plt.title("Test Plot")
plt.ylim([1e-7, 1e4])
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.draw()
plt.plot()

peaks0, _ = signal.find_peaks(psd0, height=1e1)
peaks1, _ = signal.find_peaks(psd1, height=1e1)
print(peaks0)
print(f0[peaks0])
print(peaks1)
print(f1[peaks1])


###################################################################

guessNumSignals = 1
numElements = 2
d = 0.5

decimalPrecision = 0
precision = 180 * 10**decimalPrecision + 1

rx = np.array(rx_data[0:numElements])
print(rx.shape)

R = rx @ rx.conj().T 
Rinv = np.linalg.pinv(R) 

print(R.shape)

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)


print("DoA:", doas * 180 / np.pi)

