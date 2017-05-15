from qutip import *

# First qubit - excited
qubit1 = basis(2, 1)  # equivalent to transpose([0, 1])

# Second qubit - not excited
qubit2 = basis(2, 0)  # equivalent to transpose([1, 0])

# Total state
psi = tensor(qubit1, qubit2)  # direct product

sz1 = tensor(sigmaz(), qeye(2))  # Sz x 1
sz2 = tensor(qeye(2), sigmaz())  # 1 x Sz
