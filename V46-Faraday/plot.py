import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sp
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp

#Einlesen Daten reine Probe
gradl_rein, minutenl_rein, gradr_rein, minutenr_rein, lamb = np.genfromtxt('rein.txt', unpack=True)
L = 0.0051
print('Wellenlaengen:')
print(lamb)
print('--------------------------------')
#Berechnung der Winkel reine Probe
thetal_rein = gradl_rein + minutenl_rein/60
thetar_rein = gradr_rein + minutenr_rein/60
#Winkel in rad umrechnen
thetal_rein = thetal_rein/360 * 2*np.pi
thetar_rein = thetar_rein/360 * 2*np.pi
#Berechnung des Drehwinkels
theta_rein = 1/2*(thetar_rein - thetal_rein)
#Drehwinkel pro Länge
theta_frei_rein = theta_rein/L
#Ausgabe der Winkel
print('Reine Probe:')
print('Theta links in rad:')
print(thetal_rein)
print('Theta rechts in rad:')
print(thetar_rein)
print('Drehwinkel in rad:')
print(theta_rein)
print('Drehwinkel pro Laenge:')
print(theta_frei_rein)
print('------------------------------------------------')

plt.plot(lamb**2, theta_frei_rein, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/rein.pdf')
plt.clf()

gradl_dotiert136, minutenl_dotiert136, gradr_dotiert136, minutenr_dotiert136 = np.genfromtxt('dotiert136.txt', unpack=True)
L = 0.00136
#Berechnung der Winkel reine Probe
thetal_dotiert136 = gradl_dotiert136 + minutenl_dotiert136/60
thetar_dotiert136 = gradr_dotiert136 + minutenr_dotiert136/60
#Winkel in rad umrechnen
thetal_dotiert136 = thetal_dotiert136/360 * 2*np.pi
thetar_dotiert136 = thetar_dotiert136/360 * 2*np.pi
#Berechnung des Drehwinkels
theta_dotiert136 = 1/2*(thetar_dotiert136 - thetal_dotiert136)
#Drehwinkel pro Länge
theta_frei_dotiert136 = theta_dotiert136/L
#Ausgabe der Winkel
print('Dotierte Probe der Dicke 1,36mm:')
print('Theta links in rad:')
print(thetal_dotiert136)
print('Theta rechts in rad:')
print(thetar_dotiert136)
print('Drehwinkel in rad:')
print(theta_dotiert136)
print('Drehwinkel pro Laenge:')
print(theta_frei_dotiert136)
print('------------------------------------------------')

plt.plot(lamb**2, theta_frei_dotiert136, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/dotiert136.pdf')
plt.clf()

gradl_dotiert1296, minutenl_dotiert1296, gradr_dotiert1296, minutenr_dotiert1296 = np.genfromtxt('dotiert1296.txt', unpack=True)
L = 0.001296
#Berechnung der Winkel reine Probe
thetal_dotiert1296 = gradl_dotiert1296 + minutenl_dotiert1296/60
thetar_dotiert1296 = gradr_dotiert1296 + minutenr_dotiert1296/60
#Winkel in rad umrechnen
thetal_dotiert1296 = thetal_dotiert1296/360 * 2*np.pi
thetar_dotiert1296 = thetar_dotiert1296/360 * 2*np.pi
#Berechnung des Drehwinkels
theta_dotiert1296 = 1/2*(thetar_dotiert1296 - thetal_dotiert1296)
#Drehwinkel pro Länge
theta_frei_dotiert1296 = theta_dotiert1296/L
#Ausgabe der Winkel
print('Dotierte Probe der Dicke 1,296mm:')
print('Theta links in rad:')
print(thetal_dotiert1296)
print('Theta rechts in rad:')
print(thetar_dotiert1296)
print('Drehwinkel in rad:')
print(theta_dotiert1296)
print('Drehwinkel pro Laenge:')
print(theta_frei_dotiert1296)

plt.plot(lamb**2, theta_frei_dotiert1296, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/dotiert1296.pdf')
plt.clf()

#Abziehen der Dotierung Probe der Dicke 1,36mm
theta = theta_frei_dotiert136 - theta_frei_rein
def f(x, a):
    return a*x

x_plot = np.linspace(1, 8)
params, covariance_matrix = curve_fit(f, lamb**2, theta)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print('-------------------------------------------')
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
print('-------------------------------------------')

plt.plot(lamb**2, theta, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim(1,8)
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/differenz136.pdf')
plt.clf()

a_136 = ufloat(6.38309048, 1.09034248) * 10**(12)
n = np.array([3.338, 3.354, 3.374, 3.397, 3.423, 3.455, 3.492])
print('Mittelwert:',np.mean(n))
print('Fehler:', np.std(n))
n = ufloat(np.mean(n),np.std(n))
#Berechnung Masse
N = 1.2 *10**(24)
B = 421 *10**(-3)
m = unp.sqrt((sp.e**3*N*B)/(8*np.pi**2*sp.epsilon_0*sp.c**3*a_136*n))
print(m)
#Abziehen der Dotierung Probe der Dicke 1,296mm
theta = theta_frei_dotiert1296 - theta_frei_rein

x_plot = np.linspace(1, 8)
params, covariance_matrix = curve_fit(f, lamb**2, theta)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
print('-------------------------------------------')
print(params)
print(np.sqrt(np.diag(covariance_matrix)))
print('-------------------------------------------')

plt.plot(lamb**2, theta, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.xlim(1,8)
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/differenz1296.pdf')
plt.clf()

a_1296 = ufloat(11.74260897, 1.6360627) * 10**(12)
N = 2.8 *10**(24)
m = unp.sqrt((sp.e**3*N*B)/(8*np.pi**2*sp.epsilon_0*sp.c**3*a_1296*n))
print(m)
