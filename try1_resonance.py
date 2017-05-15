from qutip import *
import numpy as np
from matplotlib import pyplot as plt


def Hamiltonian(w1, w2, w3):
    # Defining coupling strengths
    g12 = 0.03 * 2 * np.pi    # qubit 1-2 coupling
    g13 = 0.0 * 2 * np.pi    # No qubit 1-3 coupling

    # Pre-compute operators for the Hamiltonian
    sz = sigmaz()
    sx = sigmax()

    # qubit 1 operators
    sz1 = tensor(sz, qeye(2), qeye(2))
    sx1 = tensor(sx, qeye(2), qeye(2))

    # qubit 2 operators
    sz2 = tensor(qeye(2), sz, qeye(2))
    sx2 = tensor(qeye(2), sx, qeye(2))

    # qubit 3 operators
    sz3 = tensor(qeye(2), qeye(2), sz)
    sx3 = tensor(qeye(2), qeye(2), sx)

    H = w1*sz1 + w2*sz2 + w3*sz3 + g12*sx1*sx2 + g13*sx1*sx3  # sx-sx coupling

    return H

# Defining frequencies
w1list = np.linspace(0.8, 1.0, 50) * 2 * np.pi
w2 = 0.9 * 2 * np.pi
w3 = 0.9 * 2 * np.pi

eigen_values = []
for w1 in w1list:
    H = Hamiltonian(w1, w2, w3)
    ee = H.eigenenergies()
    eigen_values.append([ee[1]-ee[0], ee[2]-ee[0], ee[3]-ee[0]])

eigen_values = np.array(eigen_values)
plt.plot(w1list, eigen_values[:, 0], 'r-', w1list, eigen_values[:, 1], 'g-', w1list, eigen_values[:, 2], 'b-')
plt.show()
