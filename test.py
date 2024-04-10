# importing libraries
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

print('matplotlib: {}'.format(matplotlib.__version__))
# creating initial data values
# of x and y
x = np.linspace(0, 10, 100)
y = np.sin(x)
 
# to run GUI event loop
plt.ion()
 
# here we are creating sub plots
figure = plt.figure()
ax = figure.add_subplot(111)
line1, = ax.plot(x, y)
 
# setting title
plt.title("Geeks For Geeks", fontsize=20)
 
# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# plt.show()
 
# Loop
for i in range(50):
    # creating new Y values
    new_y = np.sin(x-0.5*i)
 
    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(new_y)
    # plt.plot()
    # drawing updated values
    figure.canvas.draw()
    figure.canvas.flush_events()
    time.sleep(0.1)