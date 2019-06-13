import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp

#Einlesen Daten reine Probe
gradl, minutenl, gradr, minutenr, lamb = np.genfromtxt('rein.txt', unpack=True)
L = 0.0051
print('Wellenlaengen:')
print(lamb)
print('--------------------------------')
#Berechnung der Winkel reine Probe
thetal = gradl + minutenl/60
thetar = gradr + minutenr/60
#Winkel in rad umrechnen
thetal = thetal/360 * 2*np.pi
thetar = thetar/360 * 2*np.pi
#Berechnung des Drehwinkels
theta = 1/2*(thetar - thetal)
#Drehwinkel pro Länge
theta_frei = theta/L
#Ausgabe der Winkel
print('Reine Probe:')
print('Theta links in rad:')
print(thetal)
print('Theta rechts in rad:')
print(thetar)
print('Drehwinkel in rad:')
print(theta)
print('Drehwinkel pro Laenge:')
print(theta_frei)
print('------------------------------------------------')

plt.plot(lamb**2, theta_frei, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/rein.pdf')
plt.clf()

gradl, minutenl, gradr, minutenr = np.genfromtxt('dotiert136.txt', unpack=True)
L = 0.00136
#Berechnung der Winkel reine Probe
thetal = gradl + minutenl/60
thetar = gradr + minutenr/60
#Winkel in rad umrechnen
thetal = thetal/360 * 2*np.pi
thetar = thetar/360 * 2*np.pi
#Berechnung des Drehwinkels
theta = 1/2*(thetar - thetal)
#Drehwinkel pro Länge
theta_frei = theta/L
#Ausgabe der Winkel
print('Dotierte Probe der Dicke 1,36mm:')
print('Theta links in rad:')
print(thetal)
print('Theta rechts in rad:')
print(thetar)
print('Drehwinkel in rad:')
print(theta)
print('Drehwinkel pro Laenge:')
print(theta_frei)
print('------------------------------------------------')

plt.plot(lamb**2, theta_frei, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/dotiert136.pdf')
plt.clf()

gradl, minutenl, gradr, minutenr = np.genfromtxt('dotiert1296.txt', unpack=True)
L = 0.001296
#Berechnung der Winkel reine Probe
thetal = gradl + minutenl/60
thetar = gradr + minutenr/60
#Winkel in rad umrechnen
thetal = thetal/360 * 2*np.pi
thetar = thetar/360 * 2*np.pi
#Berechnung des Drehwinkels
theta = 1/2*(thetar - thetal)
#Drehwinkel pro Länge
theta_frei = theta/L
#Ausgabe der Winkel
print('Dotierte Probe der Dicke 1,296mm:')
print('Theta links in rad:')
print(thetal)
print('Theta rechts in rad:')
print(thetar)
print('Drehwinkel in rad:')
print(theta)
print('Drehwinkel pro Laenge:')
print(theta_frei)

plt.plot(lamb**2, theta_frei, 'r.', label='Messwerte', Markersize=4)
plt.legend()
plt.grid()
plt.gcf().subplots_adjust(bottom=0.18)
plt.xlabel(r'$\lambda^2 / \mu m^2$')
plt.ylabel(r'$\frac{\Theta}{L} / \frac{\mathrm{rad}}{\mathrm{m}}$')
plt.savefig('build/dotiert1296.pdf')
plt.clf()
