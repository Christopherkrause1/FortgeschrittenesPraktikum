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

###################################
print('Berechnungen Wuerfel 2:')
I = ufloat(161.61,1.52)
N = ufloat(25.56,0.36)
d = 3
mu1 = 1/d*unp.log(I/N)
print('mu1 = ', mu1 )

I = ufloat(161.65,1.51)
N = ufloat(24.04,0.36)
mu2 = 1/d*unp.log(I/N)
print('mu2 = ', mu2 )

I = ufloat(156.62,1.51)
N = ufloat(17.13,0.31)
d = 3*np.sqrt(2)
mu5 = 1/d*unp.log(I/N)
print('mu5 = ', mu5)

I = ufloat(158.99,1.52)
N = ufloat(18.81,0.32)
d= 2*np.sqrt(2)
mu6 = 1/d*unp.log(I/N)
print('mu6 = ', mu6 )

print('Mittelwert:', (mu1+mu2+mu5+mu6)/4)


###################################
print('---------------------------')
print('Berechnungen Wuerfel 3:')
I = ufloat(161.61,1.52)
N = ufloat(109.66,0.75)
d = 3
mu1 = 1/d*unp.log(I/N)
print('mu1 = ', mu1 )

I = ufloat(161.65,1.51)
N = ufloat(107.91,0.75)
mu2 = 1/d*unp.log(I/N)
print('mu2 = ', mu2 )

I = ufloat(156.62,1.51)
N = ufloat(102.99,0.73)
d = 3*np.sqrt(2)
mu5 = 1/d*unp.log(I/N)
print('mu5 = ', mu5)

I = ufloat(158.99,1.52)
N = ufloat(97.85,0.71)
d= 2*np.sqrt(2)
mu6 = 1/d*unp.log(I/N)
print('mu6 = ', mu6 )

print('Mittelwert:', (mu1+mu2+mu5+mu6)/4)
