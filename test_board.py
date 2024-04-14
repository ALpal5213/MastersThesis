import adi
import numpy as np
import music
import plots
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

sdr = adi.FMComms5(uri="ip:analog") # or ip address of device

sdr.rx_lo = 2350000000
sdr.rx_lo_chip_b = 2350000000

sdr.sample_rate = 30000000
sdr.rx_buffer_size = 10000 # 1024 is default

numElements = 4
d = 0.5
numSignals = 1

rx_data = sdr.rx()
rx_data = np.array(rx_data)

R = rx_data @ rx_data.conj().T

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181)
spectrum = music.scan(thetaScan, R, numElements, d, numSignals)

plt.ion()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
line1, = ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels

for i in range(100):
    rx_data = sdr.rx()
    rx_data = np.array(rx_data[0:numElements])

    print(rx_data.shape)

    R = rx_data @ rx_data.conj().T

    spectrum = music.scan(thetaScan, R, numElements, d, numSignals)

    ax.set_ylim([np.min(spectrum), np.max(spectrum)])
    line1.set_ydata(spectrum)
    fig.canvas.draw()

    fig.canvas.flush_events()

    print(i)
    time.sleep(1)