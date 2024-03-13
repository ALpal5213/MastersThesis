import adi
import matplotlib.pyplot as plt
import numpy as np

# Create device from specific uri address
# 169.254.114.13
# sdr = adi.ad9361(uri="ip:169.254.114.13")
sdr = adi.FMComms5(uri="ip:analog")
# # Get data from transceiver
# data = sdr.rx()

sdr.rx_lo = 2000000000
sdr.rx_lo_chip_b = 2411000000