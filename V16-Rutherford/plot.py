import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp

N, W, t = np.genfromtxt('winkel.txt', unpack=True)


def f_1(W):
   return 1/(4*np.pi*np.epsilon_0)**2 *(2*79*(1.6*10**(-19))**2 /(4*1))**2 * 1/(np.sine(W/2))**4

x_plot = np.linspace(0, 22)
#params, covariance_matrix = curve_fit(f_1, W)
#errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot(x_plot, f_1(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
#print(params)
#print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(W, N/t, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/ \mathrm{cm^2}$')
plt.xlabel(r'$\Theta / $°')
plt.savefig('build/winkel.pdf')
plt.clf()
