import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import importlib
import signalSim
import music
import mvdr
import plots
import time
import getFrequencyContent

importlib.reload(signalSim)
importlib.reload(music)
importlib.reload(mvdr)
importlib.reload(plots)
importlib.reload(getFrequencyContent)

# #################################################################
# # Plot 20 degree signal without noise
# #################################################################

# doas = np.array([20]) * np.pi / 180
# f_b = np.array([50000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4

# s = signalSim.generateSignals(f_b, samples, sampleRate)
# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# y = (A @ s)

# signalSim.plotSignal(y, samples, sampleRate, numElements)

# #################################################################
# # Plot 20 and -45 degree signal with noise and Plot MuSiC
# #################################################################

# doas = np.array([20, -45]) * np.pi / 180
# f_b = np.array([50000, 10000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4
# sigma = 1 / np.sqrt(2)
# guessNumSignals = 2
# precision = 181

# s = signalSim.generateSignals(f_b, samples, sampleRate)
# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
# noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#     1j * np.random.normal(0, sigma, size=[numElements, samples])

# y = (A @ s) + noise

# signalSim.plotSignal(y, samples, sampleRate, numElements)

# R = (y @ y.conj().T) / samples 

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)

# spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

# peaks, _ = signal.find_peaks(spectrum, height=5)
# doas = thetaScan[peaks]

# print("DoA (degrees):", doas * 180 / np.pi)
# plots.plot_polar(thetaScan, spectrum, peaks=peaks, title="MuSiC Scan")
# # plt.pause(10)

#################################################################
# Plot Peak Width 4 elements vs 20
#################################################################

# doas = np.array([80]) * np.pi / 180
# f_b = np.array([50000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4
# sigma = 1 / np.sqrt(2)
# guessNumSignals = 1
# precision = 1801

# s = signalSim.generateSignals(f_b, samples, sampleRate)
# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
# noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#     1j * np.random.normal(0, sigma, size=[numElements, samples])

# y = (A @ s) + noise

# # signalSim.plotSignal(y, samples, sampleRate, numElements)

# R = (y @ y.conj().T) / samples 

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)
# spectrum1 = music.scan(thetaScan, R, numElements, d, guessNumSignals)
# peaks, _ = signal.find_peaks(spectrum1, height=5)
# doas = thetaScan[peaks]
# widths = signal.peak_widths(spectrum1, peaks, rel_height=0.5)

# print("DoA (degrees):", doas * 180 / np.pi)
# print(spectrum1[peaks])

# numElements = 20

# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#     1j * np.random.normal(0, sigma, size=[numElements, samples])

# y = (A @ s) + noise
# R = (y @ y.conj().T) / samples 

# spectrum2 = music.scan(thetaScan, R, numElements, d, guessNumSignals)
# peaks, _ = signal.find_peaks(spectrum2, height=5) 
# doas = thetaScan[peaks]
# widths = signal.peak_widths(spectrum2, peaks, rel_height=0.5)

# print("DoA (degrees):", doas * 180 / np.pi)
# print(spectrum2[peaks])
# # plots.plot_polar(thetaScan, spectrum, peaks=peaks, title="MuSiC with 20 Elements")

# fig1, ax1 = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})
# fig1.tight_layout()
# ax1[0].plot(thetaScan, spectrum1)
# ax1[1].plot(thetaScan, spectrum2)

# ax1[0].set_thetamin(-90)
# ax1[0].set_thetamax(90)
# ax1[0].set_theta_zero_location('N') # make 0 degrees point up
# ax1[0].set_theta_direction(-1) # increase clockwise
# ax1[1].set_thetamin(-90)
# ax1[1].set_thetamax(90)
# ax1[1].set_theta_zero_location('N') # make 0 degrees point up
# ax1[1].set_theta_direction(-1) # increase clockwise
# plt.show(block=False)

# plt.pause(200)

# #################################################################
# # Plot Peak Width as a function of number of elements
# #################################################################

# precision = 1801
# guessNumSignals = 1
# f_b = np.array([10000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5

# np.random.seed(4321) # ensure same noise each time

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

# N = np.arange(2, 21, 1)
# widthsArray = []
# doaArray = []
# peaksArray = []

# s = signalSim.generateSignals(f_b, samples, sampleRate)

# for doa in range(0, 90, 20):
#     print(doa)
#     doa = doa * np.pi / 180
#     widthsRow = []
#     doaRow = []
#     peaksRow = []
    
#     for numElements in N:
#         doas = np.array([doa])

#         A = signalSim.generateSteeringMatrix(doas, d, numElements)

#         sigma = 1 / np.sqrt(2)
#         # np.random.seed(4321) # ensure same noise each time
#         noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#             1j * np.random.normal(0, sigma, size=[numElements, samples])

#         y = (A @ s) + noise

#         R = (y @ y.conj().T) / samples 
#         Rinv = np.linalg.pinv(R) 

#         spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

#         peaks, _ = signal.find_peaks(spectrum, height=10)
#         doas = thetaScan[peaks]
#         widths = signal.peak_widths(spectrum, peaks, rel_height=0.5)

#         widthsRow.append(widths[0])
#         doaRow.append(doas[0])
#         peaksRow.append(spectrum[peaks])
    
#     widthsArray.append(widthsRow)
#     doaArray.append(doaRow)
#     peaksArray.append(peaksRow)

# widths = np.array(widthsArray)
# doas = np.array(doaArray) * 180 / np.pi
# peaks = np.array(peaksArray)

# fig, ax = plt.subplots(3, 1, figsize=(6, 7))
# fig.tight_layout()
# for j in range(len(doas)):
#     ax[0].plot(N, doas[j], "-")
#     ax[1].plot(N, widths[j], "-")
#     ax[2].plot(N, peaks[j], "-")
    
# fig.legend(["0" + u"\u00b0", "20" + u"\u00b0", "40" + u"\u00b0", "60" + u"\u00b0", "80" + u"\u00b0"])
# ax[0].set_ylabel("DoA")
# ax[1].set_ylabel("Peak Width")
# ax[2].set_ylabel("Peak Magnitude (dB)")
# ax[2].set_xlabel("Number of Elements")

# ax[0].set_xticks(N)
# ax[1].set_xticks(N)
# ax[2].set_xticks(N)


# plt.show(block=False)

# plt.pause(60)

# #################################################################
# # MuSiC min angle
# #################################################################

# precision = 1801
# guessNumSignals = 2
# f_b = np.array([10000, 50000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4

# np.random.seed(4321) # ensure same noise each time

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

# doaArray = []

# s = signalSim.generateSignals(f_b, samples, sampleRate)

# doaLeft0 = []
# doaRight0 = []
# doaLeft1 = []
# doaRight1 = []
# doaLeft2 = []
# doaRight2 = []
# doaLeft3 = []
# doaRight3 = []
# doaActualLeft = []
# doaActualRight = []

# angles = np.arange(10, 0, -0.1)
# sigmas = [np.sqrt(0 / 2), np.sqrt(1 / 2), np.sqrt(4 / 2), np.sqrt(9 / 2)]

# for j in range(4):
#     sigma = sigmas[j]
#     print(j)

#     for doa in angles:
#         doaLeftCycle = []
#         doaRightCycle = []

#         doas = np.array([-doa, doa]) * np.pi / 180
#         if j == 0:
#             doaActualLeft.append(doas[0])
#             doaActualRight.append(doas[1])

#         A = signalSim.generateSteeringMatrix(doas, d, numElements)

#         for i in range(100):
#             noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#                 1j * np.random.normal(0, sigma, size=[numElements, samples])

#             y = (A @ s) + noise

#             R = (y @ y.conj().T) / samples 
#             Rinv = np.linalg.pinv(R) 

#             spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

#             peaks, _ = signal.find_peaks(spectrum, height=5)
#             doasEst = thetaScan[peaks]

#             if len(doasEst) == 2:
#                 doaLeftCycle.append(doasEst[0])
#                 doaRightCycle.append(doasEst[1])
#             else:
#                 doaLeftCycle.append(doasEst[0])
#                 doaRightCycle.append(doasEst[0])

#         if j == 0:
#             doaLeft0.append(np.mean(doaLeftCycle))
#             doaRight0.append(np.mean(doaRightCycle))
#         elif j == 1:
#             doaLeft1.append(np.mean(doaLeftCycle))
#             doaRight1.append(np.mean(doaRightCycle))
#         elif j == 2:
#             doaLeft2.append(np.mean(doaLeftCycle))
#             doaRight2.append(np.mean(doaRightCycle))
#         elif j == 3:
#             doaLeft3.append(np.mean(doaLeftCycle))
#             doaRight3.append(np.mean(doaRightCycle))

# doaLeft0 = np.array(doaLeft0) * 180 / np.pi
# doaRight0 = np.array(doaRight0) * 180 / np.pi
# doaLeft1 = np.array(doaLeft1) * 180 / np.pi
# doaRight1 = np.array(doaRight1) * 180 / np.pi
# doaLeft2 = np.array(doaLeft2) * 180 / np.pi
# doaRight2 = np.array(doaRight2) * 180 / np.pi
# doaLeft3 = np.array(doaLeft3) * 180 / np.pi
# doaRight3 = np.array(doaRight3) * 180 / np.pi
# doaActualLeft = np.array(doaActualLeft) * 180 / np.pi
# doaActualRight = np.array(doaActualRight) * 180 / np.pi

# fig, ax = plt.subplots(2, 2)
# fig.tight_layout()
    
# ax[0, 0].plot(doaActualRight - doaActualLeft, doaRight0 - doaLeft0, ".")
# ax[0, 1].plot(doaActualRight - doaActualLeft, doaRight1 - doaLeft1, ".")
# ax[1, 0].plot(doaActualRight - doaActualLeft, doaRight2 - doaLeft2, ".")
# ax[1, 1].plot(doaActualRight - doaActualLeft, doaRight3 - doaLeft3, ".")

# fig.supxlabel("Actual DoA Difference")
# fig.supylabel("MuSiC Estimated DoA Difference")

# var = [[0, 1], [4, 9]]

# for i in range(2):
#     for j in range(2):
#         ax[i, j].axline((0, 0), slope=1., color='C0', label='perfect fit')
#         ax[i, j].set_title("Noise Variance: " + str(var[i][j]))
#         ax[i, j].set_xbound(lower=-1, upper=21)
#         ax[i, j].set_ybound(lower=-1, upper=21)
#         ax[i, j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
#         ax[i, j].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# # fig.legend(["0" + u"\u00b0", "20" + u"\u00b0", "40" + u"\u00b0", "60" + u"\u00b0", "80" + u"\u00b0"])

# plt.show(block=False)
# plt.pause(600000000)

# #################################################################
# # MuSiC Actual Doa vs Estimated DoA with noise
# #################################################################

# precision = 1801
# guessNumSignals = 1
# f_b = np.array([10000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4

# np.random.seed(4321) # ensure same noise each time

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision) # -90 to +90 degrees

# doaArray = []

# s = signalSim.generateSignals(f_b, samples, sampleRate)

# doa0 = []
# doa1 = []
# doa2 = []
# doa3 = []
# doaActual = []

# e0 = []
# e1 = []
# e2 = []
# e3 = []

# angles = np.arange(-80, 81, 10)
# sigmas = [np.sqrt(0 / 2), np.sqrt(1 / 2), np.sqrt(4 / 2), np.sqrt(9 / 2)]

# for j in range(4):
#     sigma = sigmas[j]

#     for doa in angles:
#         doaCycle = []

#         doas = np.array([doa]) * np.pi / 180
#         if j == 0:
#             doaActual.append(doas[0])

#         A = signalSim.generateSteeringMatrix(doas, d, numElements)

#         for i in range(100):
#             noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#                 1j * np.random.normal(0, sigma, size=[numElements, samples])
            
#             # print(np.var(noise))

#             y = (A @ s) + noise

#             R = (y @ y.conj().T) / samples 
#             Rinv = np.linalg.pinv(R) 

#             spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

#             peaks, _ = signal.find_peaks(spectrum, height=5)
#             doasEst = thetaScan[peaks]
#             doaCycle.append(doasEst[0])

#         if j == 0:
#             doa0.append(np.mean(doaCycle))
#             e0.append(np.std(doaCycle))
#         elif j == 1:
#             doa1.append(np.mean(doaCycle))
#             e1.append(np.std(doaCycle))
#         elif j == 2:
#             doa2.append(np.mean(doaCycle))
#             e2.append(np.std(doaCycle))
#         elif j == 3:
#             doa3.append(np.mean(doaCycle))
#             e3.append(np.std(doaCycle))

# doa0 = np.array(doa0) * 180 / np.pi
# doa1 = np.array(doa1) * 180 / np.pi
# doa2 = np.array(doa2) * 180 / np.pi
# doa3 = np.array(doa3) * 180 / np.pi
# doaActual = np.array(doaActual) * 180 / np.pi

# e0 = np.array(e0) * 180 / np.pi
# e1 = np.array(e1) * 180 / np.pi
# e2 = np.array(e2) * 180 / np.pi
# e3 = np.array(e3) * 180 / np.pi


# fig, ax = plt.subplots(2, 2)
# fig.tight_layout()
    
# ax[0, 0].errorbar(doaActual, doa0 - doaActual, e0, marker=".")
# ax[0, 1].errorbar(doaActual, doa1 - doaActual, e1, marker=".")
# ax[1, 0].errorbar(doaActual, doa2 - doaActual, e2, marker=".")
# ax[1, 1].errorbar(doaActual, doa3 - doaActual, e3, marker=".")

# fig.supxlabel("Actual DoA")
# fig.supylabel("MuSiC Estimate Minus Actual DoA")

# var = [[0, 1], [4, 9]]

# for i in range(2):
#     for j in range(2):
#         # ax[i, j].axline((0, 0), slope=1., color='C0', label='perfect fit')
#         ax[i, j].set_title("Noise Variance: " + str(var[i][j]))
#         ax[i, j].set_xbound(lower=-85, upper=85)
#         ax[i, j].set_ybound(lower=-5, upper=5)
#         ax[i, j].set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
#         ax[i, j].set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

# # fig.legend(["0" + u"\u00b0", "20" + u"\u00b0", "40" + u"\u00b0", "60" + u"\u00b0", "80" + u"\u00b0"])

# plt.show(block=False)
# plt.pause(600)

# #################################################################
# # Plot 30 and -30 degree signal with noise and Plot signal/freq content
# #################################################################

# doas = np.array([0, 30]) * np.pi / 180
# f_b = np.array([10000, 20000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4
# sigma = 1 / np.sqrt(2)
# guessNumSignals = 2
# precision = 181

# s = signalSim.generateSignals(f_b, samples, sampleRate)
# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
# noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#     1j * np.random.normal(0, sigma, size=[numElements, samples])

# y = (A @ s)
# print(y.shape)

# signalSim.plotSignal(y, samples, sampleRate, numElements)

# R = (y @ y.conj().T) / samples 
# Rinv = np.linalg.pinv(R) 

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)

# spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

# peaks, _ = signal.find_peaks(spectrum, height=5)
# doas = thetaScan[peaks]

# print("DoA (degrees):", doas * 180 / np.pi)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

# if peaks is not None:
#     ax.plot(thetaScan[peaks], spectrum[peaks], 'o')

# # fig.title("test")

# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# ax.set_theta_zero_location('N') # make 0 degrees point up
# ax.set_theta_direction(-1) # increase clockwise
# max = np.max(spectrum)
# ax.set_yticks(np.arange(0, max + 0.1, 100))
# plt.show(block=False)

# selectedDoA = 0 * np.pi / 180

# weights = mvdr.calc_weights(selectedDoA, Rinv, numElements, d)

# y_w = weights.conj().T @ y
# print(y_w.shape)

# signalSim.plotSignal(y_w, samples, sampleRate, 1, title="MVDR Weighted Signal")


# plt.pause(20)

# #################################################################
# # Plot MVDR between 0 and 30
# #################################################################

# doas = np.array([75, 80]) * np.pi / 180
# f_b = np.array([10000, 20000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4
# sigma = 1 / np.sqrt(2)
# guessNumSignals = 2
# precision = 181

# s = signalSim.generateSignals(f_b, samples, sampleRate)
# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
# noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#     1j * np.random.normal(0, sigma, size=[numElements, samples])

# y = (A @ s)
# print(y.shape)

# res, xf = signalSim.plotSignal(y, samples, sampleRate, numElements)

# freqInd = np.flip(np.argsort(res)[-2:])
# print("Freq:", xf[freqInd])

# R = (y @ y.conj().T) / samples 
# Rinv = np.linalg.pinv(R) 

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)

# spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

# peaks, _ = signal.find_peaks(spectrum, height=5)
# doas = thetaScan[peaks]

# print("DoA (degrees):", doas * 180 / np.pi)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

# if peaks is not None:
#     ax.plot(thetaScan[peaks], spectrum[peaks], 'o')

# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# ax.set_theta_zero_location('N') # make 0 degrees point up
# ax.set_theta_direction(-1) # increase clockwise
# max = np.max(spectrum)
# ax.set_yticks(np.arange(0, max + 0.1, 100))
# plt.show(block=False)

# selectedDoA = 50 * np.pi / 180

# weights = mvdr.calc_weights(selectedDoA, Rinv, numElements, d)

# y_w = weights.conj().T @ y
# print(y_w.shape)

# signalSim.plotSignal(y_w, samples, sampleRate, 1, title="MVDR Weighted Signal")

# thetaRange = np.linspace(75 * np.pi / 180, 80 * np.pi / 180, 10)

# freq1 = []
# freq2 = []

# for theta_i in thetaRange:
#     w = mvdr.calc_weights(theta_i, Rinv, numElements, d)
#     y_w = w.conj().T @ y
#     res, xf = signalSim.plotSignal(y_w, samples, sampleRate, 1, plot=False)

#     freq1.append(res[freqInd][0])
#     freq2.append(res[freqInd][1])

# fig1, ax1 = plt.subplots(1)
# ax1.plot(thetaRange * 180 / np.pi, freq1) 
# ax1.plot(thetaRange * 180 / np.pi, freq2) 
# ax1.legend(["10 kHz", "20 kHz"])
# ax1.set_xlabel("Input DoA")
# ax1.set_ylabel("Measured Frequency Magnitude")
# plt.show(block=False)

# plt.pause(60)

# #################################################################
# # Plot MVDR between 0 and 30
# #################################################################

# doas = np.array([0, 30]) * np.pi / 180
# f_b = np.array([10000, 20000])
# samples = 10000
# sampleRate = 1000000
# d = 0.5
# numElements = 4
# sigma = np.sqrt(1 / 2)
# guessNumSignals = 2
# precision = 181

# s = signalSim.generateSignals(f_b, samples, sampleRate)
# A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
# noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
#     1j * np.random.normal(0, sigma, size=[numElements, samples])

# y = 1 * ((A @ s) + noise)
# print(y.shape)

# res, xf = signalSim.plotSignal(y, samples, sampleRate, numElements)

# freqInd = np.flip(np.argsort(res)[-2:])
# print("Freq:", xf[freqInd])

# R = (y @ y.conj().T) / samples 
# Rinv = np.linalg.pinv(R) 

# thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)

# spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

# peaks, _ = signal.find_peaks(spectrum, height=5)
# doas = thetaScan[peaks]

# print("DoA (degrees):", doas * 180 / np.pi)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

# if peaks is not None:
#     ax.plot(thetaScan[peaks], spectrum[peaks], 'o')

# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# ax.set_theta_zero_location('N') # make 0 degrees point up
# ax.set_theta_direction(-1) # increase clockwise
# max = np.max(spectrum)
# ax.set_yticks(np.arange(0, max + 0.1, 10))
# plt.show(block=False)

# selectedDoA = 0 * np.pi / 180

# weights = mvdr.calc_weights(selectedDoA, Rinv, numElements, d)

# y_w = weights.conj().T @ y
# print(y_w.shape)

# signalSim.plotSignal(y_w, samples, sampleRate, 1, title="MVDR Weighted Signal")

# thetaRange = np.linspace(0 * np.pi / 180, 10 * np.pi / 180, 10)

# freq1 = []
# freq2 = []

# for theta_i in thetaRange:
#     w = mvdr.calc_weights(theta_i, Rinv, numElements, d)
#     y_w = w.conj().T @ y
#     res, xf = signalSim.plotSignal(y_w, samples, sampleRate, 1, plot=False)

#     freq1.append(res[freqInd][0])
#     freq2.append(res[freqInd][1])

# fig1, ax1 = plt.subplots(1)
# ax1.plot(thetaRange * 180 / np.pi, freq1) 
# ax1.plot(thetaRange * 180 / np.pi, freq2) 
# ax1.legend(["10 kHz", "20 kHz"])
# ax1.set_xlabel("Input DoA")
# ax1.set_ylabel("Measured Frequency Magnitude")
# plt.show(block=False)

# plt.pause(20)

#################################################################
# Plot MVDR impact by 
#################################################################

doas = np.array([0, 10]) * np.pi / 180
f_b = np.array([10000, 20000])
samples = 10000
sampleRate = 1000000
d = 0.5
numElements = 4
sigma = np.sqrt(1 / 2)
guessNumSignals = 2
precision = 181

s = signalSim.generateSignals(f_b, samples, sampleRate)
A = signalSim.generateSteeringMatrix(doas, d, numElements)

# np.random.seed(4321) # ensure same noise each time
noise = np.random.normal(0, sigma, size=[numElements, samples]) + \
    1j * np.random.normal(0, sigma, size=[numElements, samples])

p = np.array([[10, 0],[0, 1]])

y = 1 * ((A @ p @ s) + noise)
print(y.shape)

res, xf = signalSim.plotSignal(y, samples, sampleRate, numElements)

freqInd = np.flip(np.argsort(res)[-2:])
print("Freq:", xf[freqInd])

R = (y @ y.conj().T) / samples 
Rinv = np.linalg.pinv(R) 

thetaScan = np.linspace(-0.5 * np.pi, 0.5 * np.pi, precision)

spectrum = music.scan(thetaScan, R, numElements, d, guessNumSignals)

peaks, _ = signal.find_peaks(spectrum, height=5)
doas = thetaScan[peaks]

print("DoA (degrees):", doas * 180 / np.pi)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

if peaks is not None:
    ax.plot(thetaScan[peaks], spectrum[peaks], 'o')

ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
max = np.max(spectrum)
ax.set_yticks(np.arange(0, max + 0.1, 10))
plt.show(block=False)

selectedDoA = 10 * np.pi / 180

weights = mvdr.calc_weights(selectedDoA, Rinv, numElements, d)

y_w = weights.conj().T @ y
print(y_w.shape)

signalSim.plotSignal(y_w, samples, sampleRate, 1, title="MVDR Weighted Signal")

thetaRange = np.linspace(0 * np.pi / 180, 10 * np.pi / 180, 100)

freq1 = []
freq2 = []

for theta_i in thetaRange:
    w = mvdr.calc_weights(theta_i, Rinv, numElements, d)
    y_w = w.conj().T @ y
    res, xf = signalSim.plotSignal(y_w, samples, sampleRate, 1, plot=False)

    freq1.append(res[freqInd][0])
    freq2.append(res[freqInd][1])

fig1, ax1 = plt.subplots(1)
ax1.plot(thetaRange * 180 / np.pi, freq1) 
ax1.plot(thetaRange * 180 / np.pi, freq2) 
ax1.legend(["10 kHz", "20 kHz"])
ax1.set_xlabel("Input DoA")
ax1.set_ylabel("Measured Frequency Magnitude")
plt.show(block=False)

plt.pause(20)