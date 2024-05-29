import numpy as np

def calc_weights(theta, Rinv, numElements, d):
   k = np.arange(numElements).reshape(-1, 1)
   sin_theta = np.sin(theta)
   A = np.exp(-2j * np.pi * d * k * sin_theta) 
   W = (Rinv @ A) @ np.linalg.pinv(A.conj().T @ Rinv @ A) 
   return W






