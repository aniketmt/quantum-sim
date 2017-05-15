from matplotlib import pyplot as plt
from qutip import *
import numpy as np

w = 0.1
kappa = 0.1

rho_0 = fock_dm(10, 5)
a = destroy(10)
H = w * a.dag() * a
coll = [np.sqrt(kappa)*a]
oper = [a.dag()*a]

tlist = np.linspace(0, 60, 60)
result = mesolve(H, rho_0, tlist, coll, oper)

plt.plot(tlist, result.expect[0])
plt.show()
