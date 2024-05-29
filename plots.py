import matplotlib.pyplot as plt
import numpy as np

def plot_polar(thetaScan, spectrum, peaks=None, title=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(thetaScan, spectrum) # MAKE SURE TO USE RADIAN FOR POLAR

    if peaks is not None:
        ax.plot(thetaScan[peaks], spectrum[peaks], 'o')

    if title is not None:
        plt.title(title)

    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    max = np.max(spectrum)
    ax.set_yticks(np.arange(0, max + 0.1, 10))
    plt.show(block=False)

def plot_regular(thetaScan, results, peaks=None, title=None):
    thetaScan = thetaScan * 180 / np.pi
    plt.plot(thetaScan, results) # lets plot angle in degrees

    if peaks is not None:
        plt.plot(thetaScan[peaks], results[peaks], 'o')

    if title is not None:
        plt.title(title)

    plt.xlabel("Theta [Degrees]")
    plt.ylabel("DOA Metric")
    plt.grid()
    plt.show(block=False)