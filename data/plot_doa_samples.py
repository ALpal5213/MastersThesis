import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df1 = pd.read_csv('./data/doa_samples_10-2.csv', converters={'doas': pd.eval})
df2 = pd.read_csv('./data/doa_samples_10-2_5.csv', converters={'doas': pd.eval})
df3 = pd.read_csv('./data/doa_samples_10-3.csv', converters={'doas': pd.eval})
df4 = pd.read_csv('./data/doa_samples_10-3_5.csv', converters={'doas': pd.eval})
df5 = pd.read_csv('./data/doa_samples_10-4.csv', converters={'doas': pd.eval})
df6 = pd.read_csv('./data/doa_samples_10-4_5.csv', converters={'doas': pd.eval})
df7 = pd.read_csv('./data/doa_samples_10-5.csv', converters={'doas': pd.eval})

doa1 = np.array([])
doa2 = np.array([])
doa3 = np.array([])
doa4 = np.array([])
doa5 = np.array([])
doa6 = np.array([])
doa7 = np.array([])

for doa in df1["doas"]:
    if len(doa) == 1:
        doa1 = np.append(doa1, doa)

for doa in df2["doas"]:
    if len(doa) == 1:
        doa2 = np.append(doa2, doa)

for doa in df3["doas"]:
    if len(doa) == 1:
        doa3 = np.append(doa3, doa)

for doa in df4["doas"]:
    if len(doa) == 1:
        doa4 = np.append(doa4, doa)

for doa in df5["doas"]:
    if len(doa) == 1:
        doa5 = np.append(doa5, doa)
                         
for doa in df6["doas"]:
    if len(doa) == 1:
        doa6 = np.append(doa6, doa)

for doa in df7["doas"]:
    if len(doa) == 1:
        doa7 = np.append(doa7, doa)

print(len(doa1), len(doa2), len(doa3), len(doa4), len(doa5), len(doa6), len(doa7))
# print(stats.t.sf(abs(-.77), df=15))
# doa1 = doa1[0:len(doa7)]
# doa2 = doa2[0:len(doa7)]
# doa3 = doa3[0:len(doa7)]
# doa4 = doa4[0:len(doa7)]
# doa5 = doa5[0:len(doa7)]
# doa6 = doa6[0:len(doa7)]
# doa7 = doa7[0:len(doa7)]

d1s = doa1.std()
d2s = doa2.std()
d3s = doa3.std()
d4s = doa4.std()
d5s = doa5.std()
d6s = doa6.std()
d7s = doa7.std()

x = np.array([100, 316, 1000, 3162, 10000, 31623, 100000])
y = np.array([0, 0, 0, 0, 0, 0, 0]) * 180 / np.pi
y2 = np.array([doa1.mean(), doa2.mean(), doa3.mean(), doa4.mean(), doa5.mean(), doa6.mean(), doa7.mean()]) * 180 / np.pi
e = np.array([d1s, d2s, d3s, d4s, d5s, d6s, d7s]) * 180 / np.pi

print(e)
print(y2)

plt.errorbar(x, y2, e, linestyle='None', marker='.')
plt.xscale("log")
plt.show(block=False)
plt.xlabel("Samples")
plt.ylabel("Direction of Arrival (degrees)")

plt.pause(10)