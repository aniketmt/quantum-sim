"""
Two qubits excited by waves sent to the cavity
Cavity photons have been ignored for now (2x2 Hilbert space)
Checks individual rotations for fidelity

Author: Aniket Maiti
"""


from qutip import *
from matplotlib import pyplot as plt
import numpy as np
from exp_decay_2 import fit


#
# CONSTANTS
#

wq1 = 5.685 * 2*np.pi    # qubit-1 sigma_z coefficient
wq2 = 6.290 * 2*np.pi    # qubit-2 sigma_z coefficient
gq1 = 0.09 * 2*np.pi     # qubit-1 coupling to cavity
factor = 1.0/20.0
gq2 = factor*gq1         # qubit-2 coupling to cavity
J12 = 0.10 * 2*np.pi     # sigma-z sigma-z coupling

# Oscillation amplitude and time
A = 0.25
t_pi1, t_pi2 = 22.213, 22.213/factor


#
# Operators
#

# Qubit energy and coupling
sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())
szz = tensor(sigmaz(), sigmaz())
# Driving terms
sx1 = tensor(sigmax(), qeye(2))
sx2 = tensor(qeye(2), sigmax())
# Visualization
sy1 = tensor(sigmay(), qeye(2))
sy2 = tensor(qeye(2), sigmay())


#
# Hamiltonian
#

# Rotate qubit 1 using a pulse A*cos(wt + p)
def excite1(psi0, tlist, A, w, p):
    H0 = wq1/2.0 * sz1 + wq2/2.0 * sz2 + J12/2.0 * szz
    H1 = gq1 * sx1
    # define the time-dependence of the hamiltonian using the list-string format
    args = {'A': A, 'w': w, 'p':p}
    Ht = [H0, [H1, "A * cos(w*t + p)"]]

    output = mesolve(Ht, psi0, tlist, [], [], args)
    return output


# Rotate qubit 2 using a pulse A*cos(wt + p)
def excite2(psi0, tlist, A, w, p):
    H0 = wq1/2.0 * sz1 + wq2/2.0 * sz2 + J12/2.0 * szz
    H1 = gq2 * sx2
    # define the time-dependence of the hamiltonian using the list-string format
    args = {'A': A, 'w': w, 'p':p}
    Ht = [H0, [H1, "A * cos(w*t + p)"]]

    output = mesolve(Ht, psi0, tlist, [], [], args)
    return output


#
# Operations on System
#

# get t_pi1 and t_pi2 by fitting decay graph and getting frequency -> gives incorrect t_pi
def get_pulse_time():
    global t_pi1, t_pi2

    w = wq1+J12
    psi0 = tensor(basis(2, 1), basis(2, 0))
    t_list = np.linspace(0, 10 * 2 * np.pi / A, 5000)
    result = excite1(psi0, t_list, A, w, 0)
    sz1list = expect(sz1, result.states)
    # t_pi1 = t_list[np.argmax(sz1list)]
    c1, f1 = fit(t_list, sz1list)
    
    t_pi1 = (2 * np.pi / f1) / 2.0
    t_pi2 = t_pi1/factor

    print t_pi1, t_pi2


# Excitation using a pulse
# Rotate qubit 1 using half pulse width
def full_rotation1(rho0, w):
    t_list = np.linspace(0.0, t_pi1, 500)
    result = excite1(rho0, t_list, A, w, 0.0)

    return result.states


# Rotate qubit2 using full pulse width
def full_rotation2(rho0, w):
    t_list = np.linspace(0, t_pi2, 500/factor)
    result = excite2(rho0, t_list, A, w, 0.0)

    return result.states


# Travel pi (ideally) to go to state |01> from state |00>
def check_rotation():
    rho0 = tensor(fock(2, 0), fock(2, 0))    # state |00>

    # rho1 = full_rotation1(rho0, wq1 + J12)[-1]
    # exp_rho = tensor(basis(2, 1), basis(2, 0))

    rho1 = full_rotation2(rho0, wq2+J12)[-1]
    exp_rho = tensor(basis(2, 0), basis(2, 1))  # state |01>

    visualize(rho1)
    
    print 'Fidelity: ', fidelity(rho1, exp_rho)


# Visualize rotation on a Bloch sphere
def visualize(state_final):
    sphere = Bloch()

    sxlist1 = expect(sx1, state_final)
    sylist1 = expect(sy1, state_final)
    szlist1 = expect(sz1, state_final)
    sphere.add_vectors([sxlist1, sylist1, szlist1])

    sxlist2 = expect(sx2, state_final)
    sylist2 = expect(sy2, state_final)
    szlist2 = expect(sz2, state_final)
    sphere.add_vectors([sxlist2, sylist2, szlist2])

    sphere.show()

check_rotation()
