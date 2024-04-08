
# Import libraries
import adi
import importlib
import argparse
import time
import urllib.parse
import json
import os
import numpy as np
from scipy import signal

import music
import mvdr

importlib.reload(music)
importlib.reload(mvdr)

def doaAndBeamforming(numElements, numSignals, freq, samples):
    # Connect to device
    sdr = adi.FMComms5(uri="ip:analog") # or ip address of device

    # Set receiver settings
    sdr.rx_lo = int(freq * 1e9)        # GHz
    sdr.rx_lo_chip_b = int(freq * 1e9) # GHz

    sdr.sample_rate = 1e6
    fs = int(sdr.sample_rate)

    sdr.rx_buffer_size = samples # 1024 is default

    d = 0.5 # distance between elements in units of wavelength

    while(True): # Need to set env variables
        # Get receiver data
        rx_data = sdr.rx() # must be called for new data
        rx_data = np.array(rx_data[0:numElements])

        R = rx_data @ rx_data.conj().T 
        Rinv = np.linalg.pinv(R)

        thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181)

        results = music.scan(thetaScan, R, numElements, d, numSignals)

        peaks, _ = signal.find_peaks(results, height=-2)
        doas = thetaScan[peaks]

        dict = {
            "thetaScan": thetaScan.tolist(), 
            "results": results.tolist(), 
            "doas": doas.tolist()
        }

        dict_json = json.dumps(dict)

        jsonFile = open("./dict.json", "w") # open and write over file
        jsonFile.write(dict_json)
        jsonFile.close()

        f = open("./logs/log.txt", "a")
        f.write("DoA: " + str(doas * 180 / np.pi) + "\n")
        f.close()

        time.sleep(1) # wait 1 seconds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that performs direction of arrival and beamforming"
    )
    parser.add_argument("-n", "--numElements", required=False, type=int, default=4, help="Enter number of array elements")
    parser.add_argument("-k", "--numSignals", required=False, type=int, default=2, help="Enter number of signals")
    parser.add_argument("-f", "--freq", required=False, type=float, default=5.8, help="Enter receiver frequency in GHz")
    parser.add_argument("-s", "--samples", required=False, type=int, default=1024, help="Enter number of sample snapshots")
    args = parser.parse_args()

    numElements = args.numElements
    numSignals = args.numSignals
    freq = args.freq
    samples = args.samples

    doaAndBeamforming(numElements, numSignals, freq, samples)