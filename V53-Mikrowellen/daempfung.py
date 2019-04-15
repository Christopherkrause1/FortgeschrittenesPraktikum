import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

A, D1, D2 = np.genfromtxt('daempfung.txt', unpack=True)


#x_plot = np.linspace(70, 250, 500)

#params, covariance_matrix = curve_fit(f, U, A)
#errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot(x_plot, f(x_plot, *params), 'k-', label='Anpassungsfunktion', linewidth=0.5)
#print(params)
#print(np.sqrt(np.diag(covariance_matrix)))
#print('U_0 gleich ', params[0], ' +- ', errors[0])
#plt.gcf().subplots_adjust(bottom=0.18)
plt.plot(A, D1, 'r.', label='Messwerte', Markersize=4)
plt.plot(A, D2, 'b.', label='Eichkurve', Markersize=4)
plt.title('Daempfung')
plt.legend()
plt.grid()
#plt.ylim(-5, 160)
#plt.xlim((0, 400))
plt.xlabel(r'Tiefe$/$mm')
plt.ylabel(r'Dämpfung$/$dB')
plt.savefig('build/daempfung.pdf')
