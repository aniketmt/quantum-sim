"""
Rabi oscillations of a single qubit, with decoherence

Author: Aniket Maiti
"""


from qutip import *
from matplotlib import pyplot as plt
from numpy import pi, linspace, sqrt, arctan, cos, sin, multiply
from exp_decay_2 import fit


#
# Problem parameters:
#
delta = 0.0 * 2 * pi  # qubit sigma_x coefficient
eps0 = 1.0 * 2 * pi   # qubit sigma_z coefficient
A = 0.25 * 2 * pi     # drive amplitude (reducing -> RWA more accurate)
w = 1.0 * 2 * pi      # drive frequency

gamma = 0.10 * 4/3    # relaxation rate =  1/T1
kappa = 0.10          # dephasing rate  =  1/T_phi
n_th = 0.0            # average number of excitations ("temperature")

theta = arctan(delta/eps0)                                # Effective angle of initial state wrt z-axis
psi0 = cos(theta/2)*fock(2, 1) + sin(theta/2)*fock(2, 0)  # initial state

use_rwa = False      # Whether to use Rotating Wave Approx


#
# Operators
#
sx = sigmax()
sy = sigmay()
sz = sigmaz()
sm = destroy(2)


#
# Collapse operators
#
cops = []

# qubit relaxation
rate = (n_th+1) * gamma
if rate > 0:
    cops.append(sqrt(rate)*sm)
# qubit excitation by thermal photons
rate = n_th * gamma
if rate > 0:
    cops.append(sqrt(rate)*sm.dag())
# qubit dephasing
rate = kappa
if rate > 0:
    cops.append(sqrt(rate)*sz)

# time space
tlist = linspace(0, 5.0 * 2 * pi / A, 500)


#
# Hamiltonian
#

# # For interaction picture
# def H_func(t, args):
#     Ht = sin(w*t)
#
#     H_0_exp_m = (-1j * t * H0).expm().data
#     H_0_exp_p = (1j * t * H0).expm().data
#
#     H_MW_int = H_0_exp_m * Ht * H_0_exp_p
#     return H_MW_int


H0 = - delta / 2.0 * sx - eps0 / 2.0 * sz
H1 = - A * sx

# define the time-dependence of the hamiltonian using the list-string format
args = {'w': w}
Ht = [H0, [H1, "sin(w*t)"]]

if not use_rwa:
    output = mesolve(Ht, psi0, tlist, cops, [sm.dag() * sm], args)

else:
    # Rotating Wave Approx
    H_rwa = - delta / 2.0 * sx - A * sx / 2
    output = mesolve(H_rwa, psi0, tlist, cops, [sx, sy, sz, sm.dag() * sm])


#
# Plots
#

# Plot appropriate expectation values
if use_rwa:
    sxlist, sylist, szlist, n_q = output.expect
else:
    n_q = output.expect[0]

fig, axes = plt.subplots(1, 1)

axes.plot(tlist, n_q)
axes.set_ylim([0.0, 1.1])
axes.set_xlabel('Time [ns]')
axes.set_ylabel('Occupation probability')
axes.set_title('Excitation probability of qubit')

plt.show()

# Plot Bloch Sphere
if use_rwa:
    sphere = Bloch()
    sphere.add_points([sxlist, sylist, szlist], meth='l')
    sphere.vector_color = ['r']
    sphere.add_vectors([sin(theta), 0, -cos(theta)])  # direction of eigenvector
    sphere.show()

# Get Rabi decay constant by fitting to a decaying sinusoid
c, f = fit(tlist, multiply(n_q, 100))  # returns decay constant and frequency
print '\nRabi decay constant : ', c
