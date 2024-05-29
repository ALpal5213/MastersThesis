import adi
import numpy as np
import music
import mvdr
import plots
import time
import matplotlib.pyplot as plt
import matplotlib
import signalSim
import getFrequencyContent
import calibration

import importlib
importlib.reload(signalSim)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)
importlib.reload(getFrequencyContent)
importlib.reload(calibration)

from scipy import signal, fftpack
matplotlib.use('TkAgg')



sdr = adi.FMComms5(uri="ip:analog") # or ip address of device

sdr.rx_destroy_buffer()

sdr.gain_control_mode_chan0 = 'slow_attack'
sdr.gain_control_mode_chan1 = 'slow_attack'
sdr.gain_control_mode_chip_b_chan0 = 'slow_attack'
sdr.gain_control_mode_chip_b_chan1 = 'slow_attack'

fc = 5800000000
samples = 10000
sampleRate = 30720000

sdr.rx_lo = fc
sdr.rx_lo_chip_b = fc

sdr.sample_rate = sampleRate
sdr.rx_buffer_size = samples # 1024 is default

numElements = 4
d = 0.5
numSignals = 1

rx_data = np.array(sdr.rx())[0:numElements]

print(rx_data[0:4].shape)

# Try to fix signals
while (np.all(rx_data[numElements - 2] == rx_data[numElements - 2][0]) or \
       np.all(rx_data[numElements - 1] == rx_data[numElements - 1][0])):

    sdr.sample_rate = sampleRate
    sdr.rx_buffer_size = samples
    fs = int(sdr.sample_rate)

    rx_data = np.array(sdr.rx())[0:numElements]

    print("error:", rx_data[numElements - 2][0], fs)
    print("error:", rx_data[numElements - 1][0], fs)

plt.ion()

# freq = getFrequencyContent.getFrequencyContent(rx_data[0], samples, sampleRate)
samplePeriod = 1 / sampleRate
freq = fftpack.fft(rx_data[0])
freqVals= 2 / samples * np.abs(freq[:samples // 2])
xf = np.linspace(0, 1 / (2 * samplePeriod), samples // 2)

freq = xf[np.argmax(freqVals)]

newPhases = calibration.phaseSync(rx_data, freq, samples, sampleRate, numElements).reshape(-1, 1)
rx_data = np.array(sdr.rx())[0:numElements] * newPhases

R = rx_data @ rx_data.conj().T / samples
Rinv = np.linalg.pinv(R)

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181)
spectrum = music.scan(thetaScan, R, numElements, d, numSignals)

peaks, _ = signal.find_peaks(spectrum, prominence=5)
doas = thetaScan[peaks]

weights = mvdr.calc_weights(doas, Rinv, numElements, d)

selectSignal = 0
rx_summedAndWeighted = weights[:,selectSignal].reshape(-1,1).conj().T @ rx_data
rx_summedAndWeighted = rx_summedAndWeighted.flatten()

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
line, = ax1.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

ax1.set_theta_zero_location('N') # make 0 degrees point up
ax1.set_theta_direction(-1) # increase clockwise
ax1.set_rlabel_position(55)  # Move grid labels away from other labels

fig2, ax2 = plt.subplots()

line0, = ax2.plot(np.asarray(rx_data[0,:]).squeeze().real[0:1000]) 
line1, = ax2.plot(np.asarray(rx_data[1,:]).squeeze().real[0:1000]) 
line2, = ax2.plot(np.asarray(rx_data[2,:]).squeeze().real[0:1000]) 
line3, = ax2.plot(np.asarray(rx_data[3,:]).squeeze().real[0:1000]) 

ax2.set_title("Signal Received by 4 Antenna Elements")
ax2.set_xlabel("Samples")
ax2.set_ylabel("Amplitude")
ax2.legend(["n=0", "n=1", "n=2", "n=3"])

f_w, Pxx_den_w = signal.periodogram(rx_summedAndWeighted, sampleRate)

fig3, ax3 = plt.subplots(2)
line3_0, = ax3[0].plot(rx_summedAndWeighted)
line3_1, = ax3[1].plot(f_w, Pxx_den_w) 

bufferSize = 20

rx0Buffer = rx_data[0]
rx1Buffer = rx_data[1]
rx2Buffer = rx_data[2]
rx3Buffer = rx_data[3]

for i in range(bufferSize - 1):
    rx_data = np.array(sdr.rx())[0:numElements] * newPhases
    rx0Buffer= np.vstack((rx0Buffer, rx_data[0]))
    rx1Buffer= np.vstack((rx1Buffer, rx_data[1]))
    rx2Buffer= np.vstack((rx2Buffer, rx_data[2]))
    rx3Buffer= np.vstack((rx3Buffer, rx_data[3]))

print(rx0Buffer.shape)
print(rx0Buffer[-1].shape)

# print(np.mean(rx_data, axis=0).shape)
avg_rx = rx_data

for i in range(10000):
    rx_data = np.array(sdr.rx())[0:numElements] * newPhases

    np.roll(rx0Buffer, 1, axis=0)
    np.roll(rx1Buffer, 1, axis=0)
    np.roll(rx2Buffer, 1, axis=0)
    np.roll(rx3Buffer, 1, axis=0)

    rx0Buffer[-1] = rx_data[0]
    rx1Buffer[-1] = rx_data[1]
    rx2Buffer[-1] = rx_data[2]
    rx3Buffer[-1] = rx_data[3]

    avg_rx0 = np.sum(rx0Buffer, axis=0) / bufferSize
    avg_rx1 = np.sum(rx1Buffer, axis=0) / bufferSize
    avg_rx2 = np.sum(rx2Buffer, axis=0) / bufferSize
    avg_rx3 = np.sum(rx3Buffer, axis=0) / bufferSize

    avg_rx = np.vstack((avg_rx0, avg_rx1, avg_rx2, avg_rx3))

    R = rx_data @ rx_data.conj().T / samples
    Ravg = avg_rx @ avg_rx.conj().T / samples
    Rinv = np.linalg.pinv(Ravg)
    # print(R)
    # print(Rinv)

    spectrum = music.scan(thetaScan, R, numElements, d, numSignals)

    peaks, _ = signal.find_peaks(spectrum, prominence=5)
    doas = thetaScan[peaks]

    

    # weights = np.empty((numElements, numSignals))
    # print(doas)
    weights = mvdr.calc_weights(doas[0], Rinv, numElements, d)

    # print(weights)
    # print(weights.shape)
    # print(weights[:,0].reshape(-1,1).shape)
    selectSignal = 0
    rx_summedAndWeighted = weights[:,selectSignal].reshape(-1,1).conj().T @ rx_data
    rx_summedAndWeighted = rx_summedAndWeighted.flatten()

    f_w, Pxx_den_w = signal.periodogram(rx_summedAndWeighted, sampleRate)

    line3_0.set_ydata(rx_summedAndWeighted)
    line3_1.set_ydata(Pxx_den_w)

    ax3[0].set_ylim([np.min(rx_summedAndWeighted), np.max(rx_summedAndWeighted)])
    ax3[1].set_ylim([np.min(Pxx_den_w), np.max(Pxx_den_w)])

    fig3.canvas.draw()
    fig3.canvas.flush_events()

    ax1.set_ylim([np.min(spectrum), np.max(spectrum)])
    line.set_ydata(spectrum)
    fig1.canvas.draw()
    fig1.canvas.flush_events()


    line0.set_ydata(np.asarray(rx_data[0,:]).squeeze().real[0:1000])
    line1.set_ydata(np.asarray(rx_data[1,:]).squeeze().real[0:1000])
    line2.set_ydata(np.asarray(rx_data[2,:]).squeeze().real[0:1000])
    line3.set_ydata(np.asarray(rx_data[3,:]).squeeze().real[0:1000])

    fig2.tight_layout(pad=1)
    ax2.set_ylim([np.min(rx_data), np.max(rx_data)])
    fig2.canvas.draw()
    fig2.canvas.flush_events()


    print("iter:", i, "DoA (degrees):", doas * 180 / np.pi)
    print("Freq Weighted:", f_w[np.argmax(Pxx_den_w)])

    
    time.sleep(0.1)