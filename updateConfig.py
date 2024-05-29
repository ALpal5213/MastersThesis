import configparser
import time

def updateConfig():
    config = configparser.ConfigParser()

    config.read('config.ini')

    config['Script']['updateInterval'] = '0.1'

    config['Parameters']['numElements'] = '4'
    config['Parameters']['numSignals'] = '3'
    config['Parameters']['selectSignal'] = '2'
    config['Parameters']['elementSpacing'] = '0.5'
    config['Parameters']['carrierFreq'] = '5.8'
    config['Parameters']['samples'] = '10000'
    config['Parameters']['sampleRate'] = '30720000'

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    time.sleep(1)


if __name__ == "__main__":
    updateConfig() 