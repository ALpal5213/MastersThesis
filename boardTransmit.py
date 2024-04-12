import adi

# Connect to FMCOMMS5
sdr = adi.FMComms5(uri="ip:analog")

# Transmitter settings
sdr.tx_lo = 5800000000
sdr.tx_lo_chip_b = 5800000000

# 0 = Channel TX1A_A
# 1 = Channel TX2A_A
# 2 = Channel TX1A_B
# 3 = Channel TX2A_B
sdr.dds_single_tone(30000, 0.9, 2) 