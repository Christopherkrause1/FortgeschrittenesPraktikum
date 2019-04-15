import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

U, A, U2, A2, U3, A3 = np.genfromtxt('spannungen.txt', unpack=True)
#
def f(x, a, b, c):
    return a*x**2 + b*x + c

x_plot = np.linspace(70, 250, 500)

params, covariance_matrix = curve_fit(f, U, A)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
#print(params)
#print(np.sqrt(np.diag(covariance_matrix)))
#print('U_0 gleich ', params[0], ' +- ', errors[0])
#plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(U , A, 'r.', label='Messwerte', Markersize=4)


params, covariance_matrix = curve_fit(f, U2, A2)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', linewidth=0.5)
plt.plot(U2 , A2, 'r.', Markersize=4)

params, covariance_matrix = curve_fit(f, U3, A3)
errors = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, f(x_plot, *params), 'k-', linewidth=0.5)
plt.plot(U3 , A3, 'r.', Markersize=4)

plt.title('Parabeln')
plt.legend()
plt.grid()
plt.ylim(-5, 160)
#plt.xlim((0, 400))
plt.xlabel(r'$U/$V')
plt.ylabel(r'$U/$mV')
plt.savefig('build/plot1.pdf')
