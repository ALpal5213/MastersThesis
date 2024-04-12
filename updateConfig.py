import configparser
import time



def updateConfig():
    config = configparser.ConfigParser()

    config.read('config.ini')

    config['Script']['run'] = 'True'
    config['Script']['updateInterval'] = '1'

    config['Parameters']['numElements'] = '4'
    config['Parameters']['numSignals'] = '1'
    config['Parameters']['elementSpacing'] = '0.5'
    config['Parameters']['carrierFreq'] = '5.8'
    config['Parameters']['samples'] = '10000'
    config['Parameters']['sampleRate'] = '1e6'

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    time.sleep(1)



if __name__ == "__main__":
    updateConfig() 