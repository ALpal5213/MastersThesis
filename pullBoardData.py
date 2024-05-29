import os
import json
import paramiko
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
import numpy as np
from scipy import signal, fftpack
import pandas as pd
import getFrequencyContent
import sys

import importlib
import mvdr
importlib.reload(mvdr)

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

data = json.loads(data_json)

thetaScan = data["thetaScan"]
spectrum = data["spectrum"]
samples = data["samples"]
sampleRate = data["sampleRate"]

rx_0 = data["rx_0"].strip('][').split(",")
r_0 = np.array(list(map(complex, rx_0)))
f_0, Pxx_den_0 = signal.periodogram(r_0, sampleRate)

rx_weighted = data["rx_weighted"].strip('][').split(",")
r_w = np.array(list(map(complex, rx_weighted)))
f_w, Pxx_den_w = signal.periodogram(r_0, sampleRate)

plt.ion()

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
line1, = ax1.plot(thetaScan, spectrum) 
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_thetamin(-90)
ax1.set_thetamax(90)

fig2, ax2 = plt.subplots(2)
line2_0, = ax2[0].plot(r_0)
line2_1, = ax2[1].plot(f_0[0:500], Pxx_den_0[0:500]) 

fig3, ax3 = plt.subplots(2)
line3_0, = ax3[0].plot(r_w)
line3_1, = ax3[1].plot(f_w[0:500], Pxx_den_w[0:500]) 

keysWanted = ['error', 'doas', 'samples', 'selectSignal', 'numSignals']
df = pd.DataFrame()

for i in range(300):
    try:
        data_json = sftp.open(remote_file_path).read()

        data = json.loads(data_json)

        spectrum = np.array(data["spectrum"])

        ax1.set_ylim([np.min(spectrum), np.max(spectrum)])
        line1.set_ydata(spectrum)
        fig1.canvas.draw()
        fig1.canvas.flush_events()

        rx_0 = data["rx_0"].strip('][').split(",")
        r_0 = np.array(list(map(complex, rx_0)))
        f_0, Pxx_den_0 = signal.periodogram(r_0, sampleRate)

        line2_0.set_ydata(r_0)
        line2_1.set_ydata(Pxx_den_0[0:500])

        ax2[0].set_ylim([np.min(r_0), np.max(r_0)])
        ax2[1].set_ylim([np.min(Pxx_den_0), np.max(Pxx_den_0)])

        fig2.canvas.draw()
        fig2.canvas.flush_events()

        rx_w = data["rx_weighted"].strip('][').split(",")
        r_w = np.array(list(map(complex, rx_w)))
        f_w, Pxx_den_w = signal.periodogram(r_w, sampleRate)

        line3_0.set_ydata(r_w)
        line3_1.set_ydata(Pxx_den_w[0:500])

        ax3[0].set_ylim([np.min(r_w), np.max(r_w)])
        ax3[1].set_ylim([np.min(Pxx_den_w), np.max(Pxx_den_w)])

        fig3.canvas.draw()
        fig3.canvas.flush_events()

        extractedData = dict(filter(lambda item: item[0] in keysWanted, data.items()))

        freqOrigInd = np.flip(np.argsort(Pxx_den_0)[-data["numSignals"]:])
        freqMVDRInd = np.flip(np.argsort(Pxx_den_w)[-data["numSignals"]:])

        freqOrig = f_0[freqOrigInd]
        freqMVDR = f_w[freqMVDRInd]

        freqOrigMag = Pxx_den_0[freqOrigInd]
        freqMVDRMag = Pxx_den_w[freqOrigInd]

        extractedData["freqOrig"] = list(freqOrig)
        extractedData["freqMVDR"] = list(freqMVDR)
        extractedData["freqOrigMag"] = list(freqOrigMag)
        extractedData["freqMVDRMag"] = list(freqMVDRMag)

        df = pd.concat([df, pd.json_normalize(extractedData)], ignore_index=True)

        print("weights:", data["weights"])
        print("Samples:", data["samples"])
        print("iteration:", i, "DoAs:", np.array(data["doas"]) * 180 / np.pi)
        print("Freq Orig:", freqOrig, "Freq Weighted:", freqMVDR)
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print(f" - Error on line {exc_tb.tb_lineno}")
        print(str(e))
        print("ignoring")

    time.sleep(0.2)

badI = [ind for ind, ele in enumerate(list(df.doas)) if ele == [-999]]
badDoaIndices = pd.DataFrame(badI).set_axis(badI).index
badIndices = df[((df.error == True))].index

df = df.drop(badDoaIndices)
df = df.drop(badIndices)

df.to_csv('./data/freqSelectionMVDR.csv', index=False)
print(df)

sftp.close()
ssh.close()