import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat

L_1, I_1 = np.genfromtxt('konkav.txt', unpack=True)
L_2, I_2 = np.genfromtxt('planar.txt', unpack=True)


def f_s(L, r_1, r_2):
    return 1- L/r_1 - L/r_2 + L**2 /(r_1*r_2)
def f_p(L, r_1):
    return (1-L/r_1)
x = np.linspace(0, 3)
plt.plot(x, f_s(x, 1.4, 1.4), 'r-', label = 'konkav/konkav')
plt.plot(x, f_p(x, 1.4), 'b-', label = 'konkav/planar')
a = np.ones(50)
y = np.linspace(-0.5, 3)
plt.plot(y, a, 'k--')
#plt.plot(x, f_s(x, 1400, 1400), 'b-')
plt.legend()
plt.grid()
plt.xlim((-0.5, 3))
plt.ylim((0, 1.2))
plt.ylabel(r'$g_{\mathrm{1}} g_{\mathrm{2}}$')
plt.xlabel(r'$L / $m')
plt.savefig('build/parameter.pdf')
plt.clf()


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
plt.clf()

print('--------------------------------------')
print('Polarisation')

def f(P, I0, phi0):
    return I0 * np.cos(P-phi0)**2


P, I = np.genfromtxt('polarisation.txt', unpack=True)
P = (P/360)*2*np.pi

x_plot = np.linspace(-0.2,2*np.pi,200)
params, covariance_matrix = curve_fit(f, P, I)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(P, I, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim((-0.2, 2*np.pi))
plt.ylabel(r'$I / \mathrm{\mu A}$')
plt.xlabel(r'$\phi_{P}$')
plt.savefig('build/polarisation.pdf')
plt.clf()


print('-------------------------------------------')
print('Moden')
print('Grundmode:')
def g1(L, I0, L0, w):
    return I0 * np.exp(-2*((L-L0)/w)**2)

L, I = np.genfromtxt('Grundmode.txt', unpack=True)

x_plot = np.linspace(-16,13)
params, covariance_matrix = curve_fit(g1, L, I)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, g1(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(L, I, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim((-16, 13))
plt.ylabel(r'$I / \mathrm{\mu A}$')
plt.xlabel('$x$ / mm')
plt.savefig('build/grundmode.pdf')
plt.clf()

print('erst Mode:')
def g2(L, I0, L0, w):
    return I0 * ((L-L0)/w)**2 * np.exp(-2*((L-L0)/w)**2)

L, I = np.genfromtxt('ErsteMode.txt', unpack=True)

x_plot = np.linspace(-16,26,200)
params, covariance_matrix = curve_fit(g2, L, I)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, g2(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(L, I, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim((-16, 26))
plt.ylabel(r'$I / \mathrm{\mu A}$')
plt.xlabel('$x$ / mm')
plt.savefig('build/erstemode.pdf')
plt.clf()
