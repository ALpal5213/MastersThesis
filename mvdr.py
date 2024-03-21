import numpy as np
# import matplotlib.pyplot as plt
# import importlib
# import signal_simulation
# importlib.reload(signal_simulation)


def w_mvdr(theta, R, numElements, d):
   k = np.arange(numElements).reshape(-1, 1)
   sin_theta = np.sin(theta)
   A = np.exp(-2j * np.pi * d * k * sin_theta) 

   Rinv = np.linalg.pinv(R) 

   w = (Rinv @ A) @ np.linalg.pinv(A.conj().T @ Rinv @ A) 
   return w

def w_mvdr_test(theta, R, numElements, d, numSignals):
   k = np.arange(numElements).reshape(-1, 1)
   K = np.tile(k, (1, numSignals))
   # print(K)
   # print(K.shape)

   sin_theta = np.sin(theta).reshape(-1, 1)
   # print(sin_theta.shape)

   A = np.exp(-2j * np.pi * d * K @ sin_theta) 

   Rinv = np.linalg.pinv(R) 

   w = (Rinv @ A) / (A.conj().T @ Rinv @ A) 
   return w

# def w_mvdr(theta, r, numElements, d):
#    a = np.exp(-2j * np.pi * d * np.arange(numElements) * np.sin(theta)) # steering vector in the desired direction theta
#    a = a.reshape(-1,1) # make into a column vector (size 3x1)
#    R = r @ r.conj().T # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
#    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
#    w = (Rinv @ a)/(a.conj().T @ Rinv @ a) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
#    return w