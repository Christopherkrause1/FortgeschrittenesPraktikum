import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat

L_1, I_1 = np.genfromtxt('konkav.txt', unpack=True)
L_2, I_2 = np.genfromtxt('planar.txt', unpack=True)

def f_1(L, a, b, c):
   return a * L**2 + b*L + c

x_plot = np.linspace(55, 140)
params, covariance_matrix = curve_fit(f_1, L_1, I_1)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f_1(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print('a')
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(L_1, I_1, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim((56, 138))
plt.ylabel(r'$I / \mathrm{\mu A}$')
plt.xlabel(r'$L / $cm')
plt.savefig('build/stabilitaet.pdf')
plt.clf()

def f_2(L, d, e):
   return d * L + e

x_plot = np.linspace(55, 100)
params, covariance_matrix = curve_fit(f_2, L_2, I_2)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f_2(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print('b')
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(L_2, I_2, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim((56, 95))
plt.ylabel(r'$I / \mathrm{\mu A}$')
plt.xlabel(r'$L / $cm')
plt.savefig('build/stabilitaet_2.pdf')
