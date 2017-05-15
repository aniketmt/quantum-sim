import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def func(x, c):
    y = np.exp(-c*x)
    return y


# # This function extracts the maximas, for easy fitting
# def get_peaks(xdata, ydata):
#     peaks = []
#     times = []
#     for i in range(len(ydata)-2):
#         if ydata[i]< ydata[i+1] < ydata[i+2]:
#             peaks.append(ydata[i+1])
#             times.append(xdata[i+1])
#     times = np.array(times)
#     peaks = np.array(peaks)
#     return times, peaks


# Fit the decay of the peaks to get the exponential constant
def exp_decay(times, peaks):
    popt, pcov = curve_fit(func, times, peaks)
    return popt[0]


# Plot both and return the exponential component
def exp_const(xdata, ydata):
    times, peaks = xdata, ydata
    const = exp_decay(times, peaks)
    yfinal = func(xdata, const)
    plt.plot(xdata, ydata, 'b-', xdata, yfinal, 'r-')
    plt.show()
    return const
