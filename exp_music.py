import adi
import importlib
import matplotlib.pyplot as plt
import numpy as np
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

###################################################################
# Run MUSIC Algorithm

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

results = music.scan(thetaScan, R, numElements, d, guessNumSignals)

peaks, _ = signal.find_peaks(results, height=-2)
doas = thetaScan[peaks]

plots.plot_polar(thetaScan, results, peaks=peaks, title="MuSiC")
plots.plot_regular(thetaScan, results, peaks=peaks, title="MuSiC")

print("DoA:", doas * 180 / np.pi)