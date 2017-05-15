from qutip import *
import numpy as np
from scipy.linalg import sqrtm


# Our definition of fidelity
def fid(psi1, psi2):
    dm1 = np.asmatrix(ket2dm(psi1).full())
    dm2 = np.asmatrix(ket2dm(psi2).full())

    f = np.trace(sqrtm( (np.matmul(sqrtm(dm1), np.matmul(dm2, sqrtm(dm1)))) ))
    return f

# Sticking to the x-z plane
delta_theta = 0.05 * np.pi
theta_list = np.linspace(0, np.pi-delta_theta, 10)

# Print our fidelity according to our definition Tr( rt( rt(rho1)*rho2*rt(rho1) ) )
print fid(basis(2, 0), np.cos(delta_theta)*basis(2, 0) + np.sin(delta_theta)*basis(2, 1)), ' -> Our fidelity'

# Rotate around the plane
for theta in theta_list:
    psi1 = np.cos(theta)*basis(2, 0) + np.sin(theta)*basis(2, 1)
    psi2 = np.cos(theta+delta_theta)*basis(2, 0) + np.sin(theta+delta_theta)*basis(2, 1)

    sphere = Bloch()
    sphere.add_states(psi1)
    sphere.add_states(psi2)
    sphere.show()

    print fidelity(ket2dm(psi1), ket2dm(psi2))

