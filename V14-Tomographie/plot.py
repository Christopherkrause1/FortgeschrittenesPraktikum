import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp


N = np.genfromtxt('verlauf.txt', unpack=True)
x = np.linspace(0.5,146.5,147)

plt.plot(x,N,'k-', label='Messwerte', linewidth=1)
plt.grid()
plt.legend()
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.xlim(0,147)
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/verlauf.pdf')
