import numpy as np
from qutip import *
from matplotlib import pyplot as plt

wc = 10.0 * 2*np.pi  # cavity frequency
wa = 0.7 * 1.0 * 2*np.pi  # atom frequency
g = 0.05 * 2*np.pi  # coupling strength

kappa = 0.005       # cavity dissipation rate
gamma = 0.05        # atom dissipation rate
delta = 0.00        # atom dephasing rate

N = 15              # number of cavity fock states
n_th_a = 0.0        # avg number of thermal bath excitation
use_rwa = True      # rotating wave approximation: s+*a + s-*a.dag, very high frequency terms are ignored
theta = 0.25 * 2*np.pi

tlist = np.linspace(0, 25, 101)

# Operators in 2xN Hilbert space

sz = tensor(sigmaz(), qeye(N))
sm = tensor(destroy(2), qeye(N))
a = tensor(qeye(2), destroy(N))

# Hamiltonian
if use_rwa:
    H = wa*sz + wc*(a.dag()*a + 1/2) + g*(sm.dag()*a + sm*a.dag())
else:
    H = wa*sz + wc*(a.dag()*a + 1/2) + g*(sm.dag() + sm)*(a + a.dag())

qpsi = basis(2, 0)  # + np.sin(theta/2)*basis(2, 1)
psi0 = tensor(qpsi, basis(N, 1))  # ground state qubit in a cavity with a single photon

# collapse operators
cop = []
rate = (n_th_a+1) * kappa  # decay always happens
if rate > 0:
    cop.append(np.sqrt(rate)*a)
rate = n_th_a * kappa  # if surrounding temperature non-zero, thermal photons excite cavity
if rate > 0:
    cop.append(np.sqrt(rate)*a.dag())
rate = gamma
if rate > 0:
    cop.append(np.sqrt(rate)*sm)
# rate = delta
# if rate > 0:
#     cop.append(np.sqrt(rate)*sz)

# integrating over time
result = mesolve(H, psi0, tlist, [], [sz, a.dag()*a])
qubit, cavity = result.expect

fig, axes = plt.subplots(1, 1)

axes.plot(tlist, cavity, label="Cavity")
axes.plot(tlist, qubit, label="Atom excited state")
axes.legend(loc=0)
axes.set_xlabel('Time')
axes.set_ylabel('Occupation probability')
axes.set_title('Vacuum Rabi oscillations')

plt.show()
