import numpy as np
# import matplotlib.pyplot as plt
# import importlib
# import signal_simulation
# importlib.reload(signal_simulation)

# gives two separate weights for each signal
def w_mvdr(theta, Rinv, numElements, d):
   k = np.arange(numElements).reshape(-1, 1)
   sin_theta = np.sin(theta)
   A = np.exp(-2j * np.pi * d * k * sin_theta) 

   W = (Rinv @ A) @ np.linalg.pinv(A.conj().T @ Rinv @ A) 
   return W

# def scan()