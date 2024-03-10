import adi
import matplotlib.pyplot as plt
import numpy as np

import iio; 
print(iio.version) # Should be v0.25

# Create device from specific uri address
sdr = adi.ad9361(uri="ip:analog.local")
# Get data from transceiver
data = sdr.rx()