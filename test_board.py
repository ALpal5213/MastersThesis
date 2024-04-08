import adi

sdr = adi.FMComms5(uri="ip:analog") # or ip address of device
sdr.rx_lo = 5900000000