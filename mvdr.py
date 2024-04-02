import numpy as np

# gives two separate weights for each signal
def calc_weights(theta, Rinv, numElements, d):
   k = np.arange(numElements).reshape(-1, 1)
   sin_theta = np.sin(theta)
   A = np.exp(-2j * np.pi * d * k * sin_theta) 
   W = (Rinv @ A) @ np.linalg.pinv(A.conj().T @ Rinv @ A) 
   return W

def scan(thetaScan, Rinv, numElements, d, rx):
   output = []

   for theta_i in thetaScan:
      W = calc_weights(theta_i, Rinv, numElements, d)
      r_weighted = W.conj().T @ rx # apply weights
      power_dB = 10*np.log10(np.var(r_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
      output.append(power_dB)

   output -= np.max(output) # normalize

   return output