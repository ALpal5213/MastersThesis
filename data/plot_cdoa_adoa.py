import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('./data/cdoa_adoa_-80.csv', converters={'doas': pd.eval})
df2 = pd.read_csv('./data/cdoa_adoa_-70.csv', converters={'doas': pd.eval})
df3 = pd.read_csv('./data/cdoa_adoa_-60.csv', converters={'doas': pd.eval})
df4 = pd.read_csv('./data/cdoa_adoa_-50.csv', converters={'doas': pd.eval})
df5 = pd.read_csv('./data/cdoa_adoa_-40.csv', converters={'doas': pd.eval})
df6 = pd.read_csv('./data/cdoa_adoa_-30.csv', converters={'doas': pd.eval})
df7 = pd.read_csv('./data/cdoa_adoa_-20.csv', converters={'doas': pd.eval})
df8 = pd.read_csv('./data/cdoa_adoa_-10.csv', converters={'doas': pd.eval})
df9 = pd.read_csv('./data/cdoa_adoa_0.csv', converters={'doas': pd.eval})
df10 = pd.read_csv('./data/cdoa_adoa_10.csv', converters={'doas': pd.eval})
df11 = pd.read_csv('./data/cdoa_adoa_20.csv', converters={'doas': pd.eval})
df12 = pd.read_csv('./data/cdoa_adoa_30.csv', converters={'doas': pd.eval})
df13 = pd.read_csv('./data/cdoa_adoa_40.csv', converters={'doas': pd.eval})
df14 = pd.read_csv('./data/cdoa_adoa_50.csv', converters={'doas': pd.eval})
df15 = pd.read_csv('./data/cdoa_adoa_60.csv', converters={'doas': pd.eval})
df16 = pd.read_csv('./data/cdoa_adoa_70.csv', converters={'doas': pd.eval})
df17 = pd.read_csv('./data/cdoa_adoa_80.csv', converters={'doas': pd.eval})

doa1 = np.array([])
doa2 = np.array([])
doa3 = np.array([])
doa4 = np.array([])
doa5 = np.array([])
doa6 = np.array([])
doa7 = np.array([])
doa8 = np.array([])
doa9 = np.array([])
doa10 = np.array([])
doa11 = np.array([])
doa12 = np.array([])
doa13 = np.array([])
doa14 = np.array([])
doa15 = np.array([])
doa16 = np.array([])
doa17 = np.array([])

for doa in df1["doas"]:
    doa1 = np.append(doa1, doa[0])

for doa in df2["doas"]:
    doa2 = np.append(doa2, doa[0])

for doa in df3["doas"]:
    doa3 = np.append(doa3, doa[0])

for doa in df4["doas"]:
    doa4 = np.append(doa4, doa[0])

for doa in df5["doas"]:
    doa5 = np.append(doa5, doa[0])

for doa in df6["doas"]:
    doa6 = np.append(doa6, doa[0])

for doa in df7["doas"]:
    doa7 = np.append(doa7, doa[0])

for doa in df8["doas"]:
    doa8 = np.append(doa8, doa[0])

for doa in df9["doas"]:
    doa9 = np.append(doa9, doa[0])

for doa in df10["doas"]:
    doa10 = np.append(doa10, doa[0])

for doa in df11["doas"]:
    doa11 = np.append(doa11, doa[0])

for doa in df12["doas"]:
    doa12 = np.append(doa12, doa[0])

for doa in df13["doas"]:
    doa13 = np.append(doa13, doa[0])

for doa in df14["doas"]:
    doa14 = np.append(doa14, doa[0])

for doa in df15["doas"]:
    doa15 = np.append(doa15, doa[0])

for doa in df16["doas"]:
    doa16 = np.append(doa16, doa[0])

for doa in df17["doas"]:
    doa17 = np.append(doa17, doa[0])

# doa1 = doa1[0:len(doa7)]
print([len(doa1), len(doa2), len(doa3), len(doa4), len(doa5), \
       len(doa6), len(doa7), len(doa8), len(doa9), len(doa10), \
       len(doa11), len(doa12), len(doa13), len(doa14), len(doa15), \
       len(doa16), len(doa17)])

d1m = doa1.mean()
d2m = doa2.mean()
d3m = doa3.mean()
d4m = doa4.mean()
d5m = doa5.mean()
d6m = doa6.mean()
d7m = doa7.mean()
d8m = doa8.mean()
d9m = doa9.mean()
d10m = doa10.mean()
d11m = doa11.mean()
d12m = doa12.mean()
d13m = doa13.mean()
d14m = doa14.mean()
d15m = doa15.mean()
d16m = doa16.mean()
d17m = doa17.mean()

d1s = doa1.std()
d2s = doa2.std()
d3s = doa3.std()
d4s = doa4.std()
d5s = doa5.std()
d6s = doa6.std()
d7s = doa7.std()
d8s = doa8.std()
d9s = doa9.std()
d10s = doa10.std()
d11s = doa11.std()
d12s = doa12.std()
d13s = doa13.std()
d14s = doa14.std()
d15s = doa15.std()
d16s = doa16.std()
d17s = doa17.std()

x = np.array([-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80])
y = np.array([d1m, d2m, d3m, d4m, d5m, d6m, d7m, d8m, d9m, d10m, d11m, d12m, d13m, d14m, d15m, d16m, d17m]) * 180 / np.pi

y2 = np.abs(x) - np.abs(y)
e = np.array([d1s, d2s, d3s, d4s, d5s, d6s, d7s, d8s, d9s, d10s, d11s, d12s, d13s, d14s, d15s, d16s, d17s]) * 180 / np.pi

print(y)
print(y2)
print(e)

fig, ax = plt.subplots(2)
line0, = ax[0].plot(x, y)
line1, = ax[1].plot(x, y2)

ax[0].errorbar(x, y, e, linestyle='None', marker='.')
ax[1].plot(x, y2, linestyle='None', marker='.')

ax[0].set_ylim(-90, 90)
ax[0].set_xlim(-90, 90)
ax[1].set_xlim(-90, 90)

ax[0].set_ylabel("MuSiC Estimated DoA (degrees)")
ax[1].set_ylabel("Difference (degrees)")
ax[1].set_xlabel("Actual Direction of Arrival (degrees)")

plt.show(block=False)
plt.pause(100)

