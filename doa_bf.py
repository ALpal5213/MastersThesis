
# Import libraries
import adi
import sys
import importlib
import configparser
import time
import datetime
import json
import numpy as np
from scipy import signal, fftpack

import music
import mvdr
import calibration

importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(calibration)

def doaAndBeamforming():
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')

        numElements = int(config['Parameters']['numElements'])
        numSignals = int(config['Parameters']['numSignals'])
        selectSignal = int(config['Parameters']['selectSignal'])
        d = float(config['Parameters']['elementSpacing'])
        carrierFreq = float(config['Parameters']['carrierFreq'])
        samples = int(config['Parameters']['samples'])
        sampleRate = int(config['Parameters']['sampleRate'])
        updateInterval = float(config['Script']['updateInterval'])

        thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181)

        dict = {
            "error": False,
            "numElements": numElements,
            "elementSpacing": d,
            "numSignals": numSignals,
            "selectSignal": selectSignal,
            "carrierFreq": carrierFreq,
            "samples": samples,
            "sampleRate": sampleRate,
            "thetaScan": thetaScan.tolist()
        }
        
        # Connect to device
        sdr = adi.FMComms5(uri="ip:analog") # or ip address of device

        # Set receiver settings
        sdr.rx_lo = int(carrierFreq * 1e9)        # GHz
        sdr.rx_lo_chip_b = int(carrierFreq * 1e9) # GHz

        sdr.sample_rate = sampleRate
        sdr.rx_buffer_size = samples # 1024 is default

        rx_data = np.array(sdr.rx())[0:numElements]

        while (np.all(rx_data[numElements - 2] == rx_data[numElements - 2][0]) or \
               np.all(rx_data[numElements - 1] == rx_data[numElements - 1][0])):

            sdr.sample_rate = sampleRate
            sdr.rx_buffer_size = samples

            rx_data = np.array(sdr.rx())[0:numElements]

            _, _, exc_tb = sys.exc_info()
            logFile =  open("./logs/script_error.log", "a")
            logFile.write("ERROR: Init Block\n")
            output = str(datetime.datetime.now()) + f"No signal\n"
            logFile.write(output)
            logFile.close()

        bufferSize = 20

        rx0Buffer = rx_data[0]
        rx1Buffer = rx_data[1]
        rx2Buffer = rx_data[2]
        rx3Buffer = rx_data[3]

        for i in range(bufferSize - 1):
            rx_data = np.array(sdr.rx())[0:numElements]
            rx0Buffer= np.vstack((rx0Buffer, rx_data[0]))
            rx1Buffer= np.vstack((rx1Buffer, rx_data[1]))
            rx2Buffer= np.vstack((rx2Buffer, rx_data[2]))
            rx3Buffer= np.vstack((rx3Buffer, rx_data[3]))

        # calibrate 0 angle
        f, Pxx_den = signal.periodogram(rx_data[0], sampleRate)
        freq = f[np.argmax(Pxx_den)]

        correctPhases = calibration.phaseSync(rx_data, freq, samples, sampleRate, numElements).reshape(-1, 1)
        dict["correctedPhases"] = str(correctPhases.tolist())

        while(True):
            try:
                config.read('config.ini')
                numSignals = int(config['Parameters']['numSignals'])
                selectSignal = int(config['Parameters']['selectSignal'])
                samples = int(config['Parameters']['samples'])

                dict["numSignals"] = numSignals
                dict["selectSignal"] = selectSignal

                # Get receiver data
                rx_data = np.array(sdr.rx())[0:numElements] * correctPhases
                samples = rx_data.shape[1]
                dict["samples"] = samples

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
                
                spectrum = music.scan(thetaScan, R, numElements, d, numSignals)
                dict["spectrum"] = spectrum.tolist()

                peaks, _ = signal.find_peaks(spectrum, height=5)
                doas = thetaScan[peaks]

                if doas.size == 0:
                    try:
                        doas = np.array(list(map(float, dict["doas"].strip('][').split(', '))))
                    except:
                        doas = np.array([-999]) # indicates error

                dict["doas"] = doas.tolist()
                doaSelected = doas[selectSignal]

                weights = mvdr.calc_weights(doaSelected, Rinv, numElements, d)
                dict["weights"] = str(weights.tolist()) 

                rx_weighted = weights.conj().T @ rx_data
                rx_weighted = rx_weighted.flatten()
                dict["rx_0"] = str(rx_data[0].tolist())
                dict["rx_weighted"] = str(rx_weighted.tolist())

                # Add updated data to dictionary
                dict["error"] = str(False)
                dict_json = json.dumps(dict)

                jsonFile = open("./dict.json", "w") # open and write over file
                jsonFile.write(dict_json)  
                jsonFile.close()

            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                logFile =  open("./logs/script_error.log", "a")
                logFile.write("ERROR: Loop Block\n")
                output = str(datetime.datetime.now()) + f" - Error on line {exc_tb.tb_lineno}: \"" + str(e) + "\"\n"
                logFile.write(output)
                logFile.close()

                dict["error"] = str(True)
                dict_json = json.dumps(dict)

                jsonFile = open("./dict.json", "w") # open and write over file
                jsonFile.write(dict_json)  
                jsonFile.close()
            
            time.sleep(updateInterval)

    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        logFile =  open("./logs/script_error.log", "a")
        logFile.write("ERROR: doaAndBeamforming()\n")
        output = str(datetime.datetime.now()) + f" - Error on line {exc_tb.tb_lineno}: \"" + str(e) + "\"\n"
        logFile.write(output)
        logFile.close()

        dict["error"] = str(True)
        dict_json = json.dumps(dict)

        jsonFile = open("./dict.json", "w") # open and write over file
        jsonFile.write(dict_json)  
        jsonFile.close()

if __name__ == "__main__":
    doaAndBeamforming()    