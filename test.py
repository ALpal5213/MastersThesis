import adi
import mvdr
import signalSim
import numpy as np
import pandas as pd
import socket


H = 3 #m
r = 0.05 #m
ang_a = 80 * np.pi / 180

f = np.sqrt(H**2 + r**2 - 2*H*r*np.cos(ang_a))
print(f)

ang_p = np.arcsin((H / f) * np.sin(ang_a))
p_deg = ang_p * 180 / np.pi
print(p_deg)

print(180 - p_deg)