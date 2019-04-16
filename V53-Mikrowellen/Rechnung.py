import numpy as np
import numpy as np
from scipy.constants import mu_0
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp

dl = 83
dr = 71
lg = ufloat(47.3, 1.3)

print(unp.sqrt(1 + 1/unp.sin((np.pi*(dl-dr)/lg))**2))

Av = 20
An = 38.5

print(10**((An-Av)/20))
