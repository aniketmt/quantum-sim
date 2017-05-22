"""
This simulates the relaxation and dephasing of a single qubit in free space
The Hamiltonian is given by assuming the qubit is equivalent
 to a dipole situated at an angle theta to a unit magnetic field

Author: Aniket Maiti
"""

from qutip import *
from matplotlib import pyplot as plt
from numpy import pi, linspace, sqrt, cos, sin


# Define a Hamiltonian and integrate the system over a a certain time (t_list)
# w*cos(theta) is the weight of the Hamiltonian in the z-direction, and w*sin(theta) in the x
# gamma1 is the relaxation rate, while gamma2 is the de-phasing
def qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    H = w * sz  # (cos(theta) * sz + sin(theta) * sx)

    # collapse operators
    c_op_list = []

    rate = gamma1 * (n_th + 1)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)

    # evolve and calculate expectation values
    output = mesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])
    return output.expect


# CONSTANTS

w = 1.0 * 2 * pi  # qubit angular frequency
theta = 0.2 * pi  # qubit angle from sigma_z axis (toward sigma_x axis)
gamma1 = 0.1      # qubit relaxation rate
gamma2 = 0.0      # qubit dephasing rate
n_th = 0          # zero temperature (average rate of thermal excitation)


# PROCESS

# Initial state
a = 0.75
psi0 = (a * basis(2, 0) + (1 - a) * basis(2, 1))/sqrt(a**2 + (1-a)**2)

# Integrate over time
tlist = linspace(0, 30, 3000)
sx, sy, sz = qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist)


# VISUALISATION

# Plot Bloch Sphere
sphere = Bloch()
sphere.add_points([sx, sy, sz], meth='l')
sphere.point_color = ['r']
sphere.vector_color = ['b']
sphere.size = [4, 4]
sphere.font_size = 14
sphere.add_vectors([sin(theta), 0, cos(theta)])  # direction of the eigenvector
sphere.show()

# Plot Sz-value graph
fig, axes = plt.subplots(1, 1)

axes.plot(tlist[:500], sz[:500])
axes.set_xlabel('Time [ns]')
axes.set_ylabel('Sz value')
axes.set_title('State of qubit')

plt.show()
