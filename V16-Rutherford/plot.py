import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp


p, U, F = np.genfromtxt('pulshoehemit.txt', unpack=True)
p_2, U_2, F_2 = np.genfromtxt('pulshoeheohne.txt', unpack=True)

def f(x, a, b):
    return a*x + b

x_plot = np.linspace(0.05, 230)
params, covariance_matrix = curve_fit(f, p, U)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(p, U, 'k.', label='Mit Folie')
plt.errorbar(p, U, yerr=F, fmt = 'o',color='r', markersize=2, capsize=2, ecolor='b', elinewidth=0.5, markeredgewidth=0.5)


params, covariance_matrix = curve_fit(f, p_2, U_2)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'r-', label='Anpassungsfunktion', linewidth=0.5)
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18, left  = 0.14)
plt.plot(p_2, U_2, 'r.', label='ohne Folie')
plt.errorbar(p_2, U_2, yerr=F_2, fmt = 'o',color='r', markersize=2, capsize=2, ecolor='b', elinewidth=0.5, markeredgewidth=0.5)
plt.xlim(0.07, 220)
plt.legend()
plt.grid()


#plt.xlim(-0.2, 21)
#plt.ylim(10**(-26), 10**(-12))
plt.ylabel(r'$U\:/\:V$')
plt.xlabel(r'$p \: / \:$mbar')
plt.savefig('build/pulsmit.pdf')
plt.clf()








N, W, t = np.genfromtxt('winkel.txt', unpack=True)
N0 = 6249/300
#print(N0, '+-', np.sqrt(6249)/300)
#print('------------------------------------------')
#print('Zaehlraten:')
#print(N/t)
#print(np.sqrt(N)/t)
#print('------------------------------------------')
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
dichte = 5.907*10**(28)

#diff Wirkungsquerschnitt:
dWq = N/(t*N0*dichte*dicke*omega)
y_err = np.sqrt((np.sqrt(N)/(t*N0*dichte*dicke*omega))**2 + ((0.26* N)/(t*N0**2*dichte*dicke*omega))**2)
print(dWq)
print(y_err)
def f_1(y):
   return 1/(4*np.pi*8.854*10**(-12))**2 *((2*79*1.602*10**(-19)) /(4*5.638*10**6))**2 * 1/(np.sin(y/2))**4

x_num = np.linspace(0,20,21)
print('---------------------------')
print('Werte für Tabelle:')
print(f_1(x_num/360*2*np.pi))
print('----------------------------')
x_plot = np.linspace(0.01, 22,200)
#params, covariance_matrix = curve_fit(f_1, W)
#errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot(x_plot, f_1(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
#print(params)
#print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(x_plot, f_1(x_plot/360*2*np.pi), 'k-', label='Theoriekurve')
plt.errorbar(W, dWq, yerr=y_err, fmt = 'x',color='r', markersize=4, capsize=3, ecolor='b', elinewidth=0.5, markeredgewidth=0.75, label='berechnete Werte')
plt.legend()
plt.grid()
plt.yscale('log')
plt.xlim(-0.2, 21)
plt.ylim(10**(-26), 10**(-12))
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/ (\mathrm{m^2})$')
plt.xlabel(r'$\Theta / $°')
plt.savefig('build/winkel.pdf')
plt.clf()


#Mehrfachstreuung
print('----------------------------------------------')
dicke = 4 * 10**(-6)
N = 1033
t = 300
dWq = N/(t*N0*dichte*dicke*omega)
err = np.sqrt((np.sqrt(N)/(t*N0*dichte*dicke*omega))**2 + ((N*0.26)/(t*N0**2*dichte*dicke*omega))**2)
print(dWq)
print(err)

#Z-Abhängigkeit
print('--------------------------------------------')
I = np.array([3.44, 14.16, 12.14])
Ierr = np.sqrt(I*300)/300
print(Ierr)
rho = np.array([19.32, 2.70, 9.80]) * 10**6 #für in m
M = np.array([196.97, 26.98, 208.98])
x = np.array([4,3,1])*10**(-6)
NA = 6.022*10**23

N = NA * rho/M
print(N)
ziel = I/(N*x)
ziel_err = Ierr/(N*x)
print(ziel)
print(ziel_err)

z = np.array([79,13,83])

plt.errorbar(z, ziel, yerr=ziel_err, fmt = 'x',color='r', markersize=4, capsize=3, ecolor='b', elinewidth=0.5, markeredgewidth=0.75, label='berechnete Werte')
plt.legend()
plt.grid()
plt.ylabel(r'$\frac{I}{N\cdot \Delta x} / \frac{\mathrm{m^2}}{\mathrm{s}}$')
plt.xlabel('Z')
plt.savefig('build/z.pdf')

#Raumwinkel Quelle
a = 2
c = 97
d = 101
b = a*d/c
#print(b)

#Höhe
l = 10
h = l*d/c
#print(h)

#Pyramidenartiger Raumwinkel:
omega = 4 * np.arctan((b*h)/(2*d*np.sqrt(4*d**2 + b**2 + h**2)))
#print(omega)

#heutige Aktivität
I0 = ufloat(6249/300, np.sqrt(6249)/300)
A = I0 * (4*np.pi)/omega
print(A)
