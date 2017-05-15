from qutip import *
import numpy as np

w = 1 * 2*np.pi
gamma1 = 0.05
gamma2 = 0.02
theta = 0.0 * 2*np.pi

# initial state
a = 0.75
psi0 = (a * basis(2, 0) + (1-a)*basis(2, 1))/np.sqrt(a**2 + (1-a)**2)
# energy-eigenvector; State will still have non-trivial evolution due to cop:
# psi0 = np.cos(theta/2)*basis(2, 0) + np.sin(theta/2)*basis(2, 1)
tlist = np.linspace(0, 30, 1000)

# Assume the qubit is at an angle theta from the z-axis in the xz plane
H = np.cos(theta) * sigmaz()  # + np.sin(theta) * sigmax())

cop = []
n_th = 0.5  # thermal bath zero temp
cop.append(np.sqrt(gamma1*(n_th+1)) * destroy(2))  # relaxation
# cop.append(np.sqrt(gamma1*n_th) * destroy(2).dag())
cop.append(np.sqrt(gamma2) * sigmaz())  # dephasing

op = [sigmax(), sigmay(), sigmaz()]

result = mesolve(H, psi0, tlist, cop, op)
sxlist, sylist, szlist = result.expect
# sx = sxlist[:int(len(sxlist)/2)]
# sy = sylist[:int(len(sylist)/2)]
# sz = szlist[:int(len(szlist)/2)]

sphere = Bloch()
sphere.add_points([sxlist, sylist, szlist], meth='l')
sphere.vector_color = ['r']
sphere.add_vectors([np.sin(theta), 0, np.cos(theta)])  # direction of eigenvector
sphere.show()
