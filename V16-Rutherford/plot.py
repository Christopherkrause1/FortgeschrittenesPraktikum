import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp

N, W, t = np.genfromtxt('winkel.txt', unpack=True)
N0 = 6249/300
#print(N0, '+-', np.sqrt(6249)/300)
#print(N/t)
err = np.sqrt(N)/t

#effektive Detektorfläche:
#Breite
a = 2
c = 41
d = 45
b = a*d/c
#print(b)

#Höhe
l = 10
h = l*d/c
#print(h)

#Pyramidenartiger Raumwinkel:
omega = 4 * np.arctan((b*h)/(2*d*np.sqrt(4*d**2 + b**2 + h**2)))
#print(omega)

#Foliendicke
dicke = 2 * 10**(-6)

#Atomdichte
dichte = 5.895*10**(28)

#diff Wirkungsquerschnitt:
dWq = N/(t*N0*dichte*dicke*omega)
print(dWq)

def f_1(y):
   return 1/(4*np.pi*8.854*10**(-12))**2 *((2*79*1.602*10**(-19)) /(4*5.638*10**6))**2 * 1/(np.sin(y/2))**4

x_plot = np.linspace(0.01, 22,200)
#params, covariance_matrix = curve_fit(f_1, W)
#errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot(x_plot, f_1(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
#print(params)
#print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(x_plot, f_1(x_plot/360*2*np.pi), 'k-', label='Theoriekurve')
plt.plot(W, dWq, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.yscale('log')
plt.xlim(-0.2, 21)
plt.ylim(10**(-26), 10**(-12))
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/ \mathrm{cm^2}$')
plt.xlabel(r'$\Theta / $°')
plt.savefig('build/winkel.pdf')
plt.clf()
