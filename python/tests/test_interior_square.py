import numpy as np
import matplotlib.pyplot as plt

from ohbemn import Region, wave, Solver

def test_pool():
    de = 10.0 # meters
    di = 150.0 # meters

    r = Region.square()

    f, ax = plt.subplots()
    r.plot(ax)

    print("segments:", r.len())

def test_acousticbem():
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    pass
