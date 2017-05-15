# -*- coding: utf-8 -*-
"""
DO NOT CHANGE
Created on Tue May 24 17:12:20 2016

@author: Duncan, Aniket
"""

import numpy as np
import math
from matplotlib import pyplot as plt

failed = False


def fit(x, y):
    print '\nFitting..'
    amp, ic = heightguess(y)
    l = periodguess(y, x, ic)

    a, b, c, d, e = gradmin(amp, l, ic, x, y, 0)

    # print "\n\nFINAL VALUES: \n a:", a, "  b:", b, "  c:", c, "  d:", d, "  e:", e

    fity = (a * np.sin(d * x) + b * np.cos(d * x)) * np.exp(-c * x) + e
    # fity2 = np.sqrt(a**2 + b**2) * np.exp(-c * x) + e  # * np.sin(d*0 + np.arctan(b/a))
    plt.plot(x, y, 'b-')
    plt.plot(x, fity, 'r-')
    plt.show()

    return c, d  # DECAY CONSTANT, FREQUENCY


def func(a, b, c, d, e, x, y):
    return y - e - (a * np.sin(d * x) + b * np.cos(d * x)) * np.exp(-(c * x))


def gradmin(a, d, e, x, y, trycount):
    # gradient based minimization on e**-bx to find most likely value

    ia, id, ie = a, d, e

    stepsize = 0.0001
    threshold = 0.0000001
    a /= 2.0
    b = a
    c = 0

    new_ss = np.sum(np.power(func(a, b, c, d, e, x, y), 2))
    old_ss = new_ss + 1
    stepnum = 0

    # print "INITIAL VALUES: \n a:", a, "  b:", b, "  c:", c, "  d:", d, "  e:", e, "\n"
    finished = False

    while math.fabs(old_ss - new_ss) > threshold:
        old_ss = new_ss
        stepnum += 1

        agrad = np.sum(func(a, b, c, d, e, x, y) * (np.sin(d * x) * np.exp(-(c * x))))
        bgrad = np.sum(func(a, b, c, d, e, x, y) * (np.cos(d * x) * np.exp(-(c * x))))
        cgrad = np.sum(func(a, b, c, d, e, x, y) * (a*np.sin(d * x) + b*np.cos(d * x)) * x * np.exp(-(c * x)))
        dgrad = np.sum(func(a, b, c, d, e, x, y) * ((a*np.cos(d * x) - b*np.sin(d * x)) * x * np.exp(-(c * x))))
        egrad = np.sum(func(a, b, c, d, e, x, y))

        a += agrad * stepsize
        b += bgrad * stepsize
        c -= cgrad * stepsize * 0.000001
        d += dgrad * stepsize * 0.00001
        e += egrad * stepsize

        new_ss = np.sum(np.power(func(a, b, c, d, e, x, y), 2))

        # print progress every 1500 steps
        # if stepnum % 1500 == 0:
            # print "  Var: {0}\r".format(new_ss/len(x))

        # restart the minimization with higher period if minimizing to wrong well
        if stepnum/1500 > 20 and new_ss/len(x) > 2500:
            if trycount < 5:
                finished = False
                trycount += 1
                print "\n***Reset, try {}***\n\n" .format(trycount)

            else:
                finished = True
                print "\n***COULD NOT CONVERGE***"

            break

        # set an upper limit on steps
        if new_ss == old_ss or stepnum > 1500*400:
            finished = True
            print 'equal'
            break

    if math.fabs(old_ss - new_ss) < threshold:
        finished = True

    if not finished:
        new_d = id - (np.pi / max(x))
        a, b, c, d, e = gradmin(ia, new_d, ie, x, y, trycount)
        finished = True

    if finished:
        print "Var: {0}\n".format(new_ss/len(x))
        return a, b, c, d, e


def heightguess(y):
    # returns "a" and "e"
    intercept = np.mean(y)
    amplitude = max(y) - intercept

    return amplitude, intercept


def periodguess(y, x, intercept):
    # counts number of times the curve traverses its average
    crossing_index = []
    for i in range(len(y) - 1):
        if y[i + 1] != intercept and (y[i] - intercept) / (y[i + 1] - intercept) <= 0:
            crossing_index.append(i)

    traverse_count = len(crossing_index)
    l = (np.pi * traverse_count) / max(x)

    return l
