
# Import libraries
import adi
import importlib
import argparse
import configparser
import time
import json
import numpy as np
from scipy import signal

import music
import mvdr

importlib.reload(music)
importlib.reload(mvdr)

def doaAndBeamforming():
    config = configparser.ConfigParser()
    config.read('config.ini')

    numElements = int(config['Parameters']['numElements'])
    numSignals = int(config['Parameters']['numSignals'])
    d = float(config['Parameters']['elementSpacing'])
    carrierFreq = float(config['Parameters']['carrierFreq'])
    samples = int(config['Parameters']['samples'])
    sampleRate = float(config['Parameters']['sampleRate'])

    run = eval(config['Script']['run'])
    updateInterval = float(config['Script']['updateInterval'])
    
    # Connect to device
    sdr = adi.FMComms5(uri="ip:analog") # or ip address of device

    # Set receiver settings
    sdr.rx_lo = int(carrierFreq * 1e9)        # GHz
    sdr.rx_lo_chip_b = int(carrierFreq * 1e9) # GHz

    sdr.sample_rate = sampleRate
    sdr.rx_buffer_size = samples # 1024 is default

    thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181)

    dict = {
        "numElements": numElements,
        "elementSpacing": d,
        "numSignals": numSignals,
        "carrierFreq": carrierFreq,
        "samples": samples,
        "sampleRate": sampleRate,
        "thetaScan": thetaScan.tolist()
    }

    

    while(run == True):
        # Get receiver data
        rx_data = sdr.rx() # must be called for new data
        rx_data = np.array(rx_data[0:numElements])

        R = rx_data @ rx_data.conj().T 
        # Rinv = np.linalg.pinv(R)

        spectrum = music.scan(thetaScan, R, numElements, d, numSignals)

        peaks, _ = signal.find_peaks(spectrum, prominence=1)
        doas = thetaScan[peaks]

        # Add updated data to dictionary
        dict["spectrum"] = spectrum.tolist()
        dict["doas"] = doas.tolist()

        dict_json = json.dumps(dict)

        jsonFile = open("./dict.json", "w") # open and write over file
        jsonFile.write(dict_json)  
        jsonFile.close()

        logFile = open("./logs/log.txt", "a")
        logFile.write("DoA: " + str(doas * 180 / np.pi) + "\n")
        logFile.close()

        config.read('config.ini')

        numElements = int(config['Parameters']['numElements'])
        numSignals = int(config['Parameters']['numSignals'])
        d = float(config['Parameters']['elementSpacing'])
        carrierFreq = float(config['Parameters']['carrierFreq'])
        samples = int(config['Parameters']['samples'])
        sampleRate = float(config['Parameters']['sampleRate'])

        run = eval(config['Script']['run'])
        updateInterval = float(config['Script']['updateInterval'])
        
        time.sleep(updateInterval) # wait 1 seconds

if __name__ == "__main__":
    doaAndBeamforming()    