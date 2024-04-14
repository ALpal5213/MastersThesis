import os
import json
import paramiko
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
import numpy as np
from scipy import signal

import plots

load_dotenv()
BOARD_IP = os.getenv('BOARD_IP')
USER_NAME = os.getenv('USER_NAME')
USER_PASSWORD = os.getenv('USER_PASSWORD')

# Establish SSH connection
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(BOARD_IP, username=USER_NAME, password=USER_PASSWORD)

# Create SFTP client
sftp = ssh.open_sftp()

remote_file_path = '/home/analog/Desktop/code/dict.json'
data_json = sftp.open(remote_file_path).read()
print(data_json)
data = json.loads(data_json)

thetaScan = data["thetaScan"]
spectrum = data["spectrum"]

plt.ion()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
line1, = ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels

for i in range(1000):
    try:
        data_json = sftp.open(remote_file_path).read()

        data = json.loads(data_json)
        spectrum = np.array(data["spectrum"])

        # line1.set_xdata(thetaScan)
        ax.set_ylim([np.min(spectrum), np.max(spectrum)])
        line1.set_ydata(spectrum)
        fig.canvas.draw()

        fig.canvas.flush_events()

        print("iteration:", i, "DoAs:", np.array(data["doas"]) * 180 / np.pi)
    except Exception as e:
        print(str(e))
        print("ignoring")
    time.sleep(0.1)

sftp.close()
ssh.close()