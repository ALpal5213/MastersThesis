import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/freqSelectionMVDR.csv')

print("DF Rows:", len(df))

freqOrig1 = np.array(pd.eval(df['freqOrig'][0:100]))
freqOrig2 = np.array(pd.eval(df['freqOrig'][100:200]))
freqOrig3 = np.array(pd.eval(df['freqOrig'][200:]))
freqOrig = np.concatenate((freqOrig1, freqOrig2, freqOrig3))

freqMVDR1 = np.array(pd.eval(df['freqMVDR'][0:100]))
freqMVDR2 = np.array(pd.eval(df['freqMVDR'][100:200]))
freqMVDR3 = np.array(pd.eval(df['freqMVDR'][200:]))
freqMVDR = np.concatenate((freqMVDR1, freqMVDR2, freqMVDR3))

fOMag1 = np.array(pd.eval(df['freqOrigMag'][0:100]))
fOMag2 = np.array(pd.eval(df['freqOrigMag'][100:200]))
fOMag3 = np.array(pd.eval(df['freqOrigMag'][200:]))
fOMag = np.concatenate((fOMag1, fOMag2, fOMag3))

fMVDRMag1 = np.array(pd.eval(df['freqMVDRMag'][0:100]))
fMVDRMag2 = np.array(pd.eval(df['freqMVDRMag'][100:200]))
fMVDRMag3 = np.array(pd.eval(df['freqMVDRMag'][200:]))
fMVDRMag = np.concatenate((fMVDRMag1, fMVDRMag2, fMVDRMag3))

x = np.arange(len(df))
freq1 = np.array([])
freq2 = np.array([])
freq3 = np.array([])
freqSeen = np.array([])

fMag1 = np.array([])
fMag2 = np.array([])
fMag3 = np.array([])
fMagMVDR1 = np.array([])
fMagMVDR2 = np.array([])
fMagMVDR3 = np.array([])

for i in x:
    try:
        freq1 = np.append(freq1, freqOrig[0][0])
        freq2 = np.append(freq2, freqOrig[0][1])
        freq3 = np.append(freq3, freqOrig[0][2])
        freqSeen = np.append(freqSeen, freqMVDR[i][0])

        fMag1 = np.append(fMag1, fOMag[i][0])
        fMag2 = np.append(fMag2, fOMag[i][1])
        fMag3 = np.append(fMag3, fOMag[i][2])
        fMagMVDR1 = np.append(fMagMVDR1, fMVDRMag[i][0])
        fMagMVDR2 = np.append(fMagMVDR2, fMVDRMag[i][1])
        fMagMVDR3 = np.append(fMagMVDR3, fMVDRMag[i][2])
    except Exception as e:
        print("ignoring:", e)

plt.plot(x, freq1)
plt.plot(x, freq2)
plt.plot(x, freq3)
plt.plot(x, freqSeen)
plt.show(block=False)
plt.xlabel("Time")
plt.ylabel("Frequency (MHz)")
plt.legend(["Right Source", "Center Source", "Left Source"])


print("fOMag:", len(fOMag))

fig2, ax2 = plt.subplots(3, 1, figsize=(10, 10))
ax2[0].plot(x, fMag3)
ax2[1].plot(x, fMag2)
ax2[2].plot(x, fMag1)

ax2[0].plot(x, fMagMVDR3)
ax2[1].plot(x, fMagMVDR2)
ax2[2].plot(x, fMagMVDR1)

ax2[0].set_title("Left Source")
ax2[0].set_ylabel("PSD")
ax2[1].set_title("Center Source")
ax2[1].set_ylabel("PSD")
ax2[2].set_title("Right Source")
ax2[2].set_ylabel("PSD")
ax2[2].set_xlabel("Time")

fig2.tight_layout()

plt.pause(60)