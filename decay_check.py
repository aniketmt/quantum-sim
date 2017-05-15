"""
Check decay of a particular qubit in a two qubit system
Both free and forced oscillations possible

Author: Aniket Maiti
"""

from qutip import *
import numpy as np
from matplotlib import pyplot as plt
from exp_decay_2 import fit


#
# Operators
#

sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())
szz = tensor(sigmaz(), sigmaz())
# Driving terms
sx1 = tensor(sigmax(), qeye(2))
sx2 = tensor(qeye(2), sigmax())
sxx = tensor(sigmax(), sigmax())
# Collapse terms
sm1 = tensor(destroy(2), qeye(2))
sm2 = tensor(qeye(2), destroy(2))
# Visualization
sy1 = tensor(sigmay(), qeye(2))
sy2 = tensor(qeye(2), sigmay())


#
# Constants
#

w1 = 5.685 * 2*np.pi
w2 = 6.290 * 2*np.pi
J = 0.200 * 2*np.pi
g1 = 0.09 * 2*np.pi
g2 = 0.09 * 2*np.pi

gamma1 = 1/(20.6 * 10**3)
kappa1 = 1/(2 * 20.6 * 10**3) + 1/(51.4 * 10**3)
gamma2 = 1/(39.7 * 10**3)
kappa2 = 1/(2 * 39.7 * 10**3) + 1/(64.8 * 10**3)
n_th = 0.0


#
# Collapse operators
#
cops = []

# qubit1
rate = (n_th+1) * gamma1
if rate > 0:
    cops.append(np.sqrt(rate)*sm1)
# qubit excitation by thermal photons
rate = n_th * gamma1
if rate > 0:
    cops.append(np.sqrt(rate)*sm1.dag())
# qubit dephasing
rate = kappa1
if rate > 0:
    cops.append(np.sqrt(rate)*sz1)

# qubit2
rate = (n_th+1) * gamma2
if rate > 0:
    cops.append(np.sqrt(rate)*sm2)
# qubit excitation by thermal photons
rate = n_th * gamma2
if rate > 0:
    cops.append(np.sqrt(rate)*sm2.dag())
# qubit dephasing
rate = kappa2
if rate > 0:
    cops.append(np.sqrt(rate)*sz2)


#
# Hamiltonian
#

H = w1/2.0 * sz1 + w2/2.0 * sz2 + J/2.0 * szz


# Oscillations forced with a signal of A*sin(wt), for time t [rotation]
def forced_oscillations(psi0, A, w, time):
    H1 = g1*sx1 + g2*sx2
    args = {'A': A, 'w': w}
    Ht = [H, [H1, "A*sin(w*t)"]]
    t_list = np.linspace(0, time, 500)
    result = mesolve(Ht, psi0, t_list, cops, [sm2.dag()*sm2], args)

    return np.array(result.expect[0]), t_list


# Hamiltonian simply allowed to evolve [rest]
def free_oscillations(psi0):
    t_list = np.linspace(0, 25 * 10**3, 500)
    result = mesolve(H, psi0, t_list, cops, [sm2.dag()*sm2])
    return np.array(result.expect[0]), t_list


#
# Display
#

psi0 = tensor(fock(2, 1), fock(2, 1))

# # Free Oscillations
# sz2_list, t_list = free_oscillations(psi0)

# Forced Oscillations
A = 0.25
sz2_list, t_list = forced_oscillations(psi0, A, w2-J, 2.0*22.213)

# Plot
fig, axes = plt.subplots(1, 1)
axes.plot(t_list, sz2_list)
axes.set_ylim([0.0, 1.1])
axes.set_xlabel('Time [ns]')
axes.set_ylabel('Occupancy')
axes.set_title('State of qubit')
plt.show()

# # Fit if possible
# c, f = fit(t_list, sz2_list*100)
# print 'The exponential decay constant is : ', c
