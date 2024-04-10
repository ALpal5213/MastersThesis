import json
import paramiko
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import plots

# Establish SSH connection
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('169.254.92.202', username='analog', password='analog')

# Create SFTP client
sftp = ssh.open_sftp()

remote_file_path = '/home/analog/Desktop/code/dict.json'
data_json = sftp.open(remote_file_path).read()

data = json.loads(data_json)

thetaScan = data["thetaScan"]
results = data["results"]

plt.ion()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
line1, = ax.plot(thetaScan, results) # MAKE SURE TO USE RADIAN FOR POLAR

ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels

for i in range(20):
    data_json = sftp.open(remote_file_path).read()

    data = json.loads(data_json)
    results = data["results"]

    # peaks, _ = signal.find_peaks(results, prominence=1)
    # doas = thetaScan[peaks]

    

    # if peaks is not None:
    #     ax.plot(thetaScan[peaks], results[peaks], 'x')

    # if title is not None:
    #     plt.title(title)
    line1.set_xdata(thetaScan)
    line1.set_ydata(results)
    fig.canvas.draw()


    # plt.show()


    fig.canvas.flush_events()

    time.sleep(0.5)

sftp.close()
ssh.close()
# plots.plot_polar(thetaScan, results, title="MuSiC")


# peaks, _ = signal.find_peaks(results, height=-2)
# doas = thetaScan[peaks]

# print("DoA:", doas * 180 / np.pi)