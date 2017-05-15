"""
Two qubits excited by waves sent to the cavity
Cavity photons included (vacuum Rabi possible if w = 0)
2x2xN Hilbert space

Author : Aniket Maiti
"""


from qutip import *
from matplotlib import pyplot as plt
import numpy as np
from exp_decay_2 import fit


#
# CONSTANTS
#

# Coupling
wq1 = 5.685 * 2*np.pi    # qubit-1 sigma_z coefficient
wq2 = 6.290 * 2*np.pi    # qubit-2 sigma_z coefficient
wc  = 7.200 * 2*np.pi    # cavity frequency
gq1 = 0.09 * 2*np.pi     # qubit-1 coupling to cavity
factor = 20.0/20.0
gq2 = factor * gq1       # qubit-2 coupling to cavity
J12 = 0.10 * 2*np.pi     # sigma-z sigma-z coupling

# Decoherence
T1_1 = np.Infinity            # T1 for qubit-1
T2_1 = np.Infinity            # T2(E) for qubit-1
T1_2 = np.Infinity            # T1 for qubit-2
T2_2 = np.Infinity            # T2(E) for qubit-2
T1_c = np.Infinity            # T1 for the cavity

gammaq1 = 1/T1_1              # relaxation rate for qubit 1 =  1/T1(q1)
gammaq2 = 1/T1_2              # relaxation rate for qubit 2 =  1/T1(q2)
dphiq1 = 1/(2*T1_1) + 1/T2_1  # dephasing rate for qubit 1 =  1/T_phi(q1)
dphiq2 = 1/(2*T1_2) + 1/T2_2  # dephasing rate for qubit 2 =  1/T_phi(q2)
kappa = 1/T1_c                # cavity relaxation rate

n_th = 0.0                    # average number of excitations ("temperature")

# Drive oscillations
A = 0.25              # Amplitude of given oscillations
t_pi1 = 22.213        # Found to have best fidelity
t_pi2 = t_pi1/factor  # For second qubit, since coupling strength is different

# Approximations
N = 15                # Number of fock states possible for cavity
use_rwa = False       # Whether to use the Rotating Wave Approximation

# Tomography
T_init, T_final = 0.0, 0.0  # Will keep universal time
process_stack = []          # To retrace active-processes and hence project back onto z-axis


#
# Operators
#

# Qubit
sz1 = tensor(sigmaz(), qeye(2), qeye(N))
sz2 = tensor(qeye(2), sigmaz(), qeye(N))
szz = tensor(sigmaz(), sigmaz(), qeye(N))
# Cavity interaction
adsm1 = tensor(destroy(2), qeye(2), destroy(N).dag())
asm1d = tensor(destroy(2).dag(), qeye(2), destroy(N))
adsm2 = tensor(qeye(2), destroy(2), destroy(N).dag())
asm2d = tensor(qeye(2), destroy(2).dag(), destroy(N))
# Driving terms
sx1 = tensor(sigmax(), qeye(2), qeye(N))
sx2 = tensor(qeye(2), sigmax(), qeye(N))
# Collapse terms
sm1 = tensor(destroy(2), qeye(2), qeye(N))
sm2 = tensor(qeye(2), destroy(2), qeye(N))
# Cavity relaxation
a = tensor(qeye(2), qeye(2), destroy(N))
# Visualise
sy1 = tensor(sigmay(), qeye(2), qeye(N))
sy2 = tensor(qeye(2), sigmay(), qeye(N))

#
# Collapse operators
#
cops = []

# qubit1
# qubit relaxation
rate = (n_th+1) * gammaq1
if rate > 0:
    cops.append(np.sqrt(rate)*sm1)
# qubit excitation by thermal photons
rate = n_th * gammaq1
if rate > 0:
    cops.append(np.sqrt(rate)*sm1.dag())
# qubit dephasing
rate = dphiq1
if rate > 0:
    cops.append(np.sqrt(rate)*sz1)

# qubit2
# qubit relaxation
rate = (n_th+1) * gammaq2
if rate > 0:
    cops.append(np.sqrt(rate)*sm2)
# qubit excitation by thermal photons
rate = n_th * gammaq2
if rate > 0:
    cops.append(np.sqrt(rate)*sm2.dag())
# qubit dephasing
rate = dphiq2
if rate > 0:
    cops.append(np.sqrt(rate)*sz2)

# cavity
# cavity relaxation
rate = (n_th+1) * kappa
if rate > 0:
    cops.append(np.sqrt(rate)*a)
# cavity excitation by thermal photons
rate = n_th * kappa
if rate > 0:
    cops.append(np.sqrt(rate)*a.dag())


#
# Hamiltonian
#

def compute(psi0, tlist, A, w, p, nodecay=False):
    if not use_rwa:
        # Entire James-Cummings Hamiltonian
        H0 = wq1/2.0 * sz1 + wq2/2.0 * sz2 + J12/2.0 * szz + wc/2.0 * (a.dag()*a + 1/2) + gq1/2.0 * (adsm1+asm1d) + gq2/2.0 * (adsm2+asm2d)
        H1 = gq1 * sx1 + gq2 * sx2

        # define the time-dependence of the hamiltonian using the list-string format
        args = {'A': A, 'w': w, 'p':p}
        Ht = [H0, [H1, "A * cos(w*t + p)"]]
        if nodecay:
            output = mesolve(Ht, psi0, tlist, [], [], args)  # , options=Options(nsteps=10000))
        else:
            output = mesolve(Ht, psi0, tlist, cops, [], args)  # , options=Options(nsteps=10000))

    else:
        # Rotating Wave Approx
        H_rwa = J12/2.0 * szz + A * (gq1 * sx1 + gq2 * sx2) + wc/2.0 * (a.dag()*a + 1/2) + gq1/2.0 * (adsm1+asm1d) + gq2/2.0 * (adsm2+asm2d)
        if nodecay:
            output = mesolve(H_rwa, psi0, tlist, [], [])  # , options=Options(nsteps=10000))
        else:
            output = mesolve(H_rwa, psi0, tlist, cops, [])  # , options=Options(nsteps=10000))

    return output


#
# Operations on System
#

# check the shifting of the wu and wl from w_mean due to zz coupling
def check_w1_ul():
    w_list = np.linspace(0, 2*wq1/(2*np.pi), 50)
    psi01 = tensor(basis(2, 0), basis(2, 0), basis(N, 0))
    psi02 = tensor(basis(2, 0), basis(2, 1), basis(N, 0))
    t_list = np.linspace(0, 2*np.pi/A, 100)
    amp_list1, amp_list2 = [], []

    for w in w_list:
        result1 = compute(psi01, t_list,  A, 2*np.pi*w, 0)
        sz1list1 = result1.expect[0]
        amp_list1.append(np.max(sz1list1)-np.min(sz1list1))
        result2 = compute(psi02, t_list, A, 2 * np.pi * w, 0)
        sz1list2 = result2.expect[0]
        amp_list2.append(np.max(sz1list2) - np.min(sz1list2))
    plt.plot(w_list, amp_list1, 'b-', w_list, amp_list2, 'r-')
    plt.show()


# check time evolution of qubit-1 state for a particular w
def check_w(w):
    psi0 = tensor(basis(2, 0), basis(2, 0), basis(N, 0))
    t_list = np.linspace(0, 2*np.pi/A, 500)
    result = compute(psi0, t_list, A, w, 0)
    print len(result.states)
    sz1list = expect(sz1, result.states)

    fig, axes = plt.subplots(1, 1)
    axes.plot(t_list, sz1list)
    axes.set_ylim([-1.0, 1.1])
    plt.show()


# get t_pi1 and t_pi2
def get_pulse_time():
    global t_pi1, t_pi2

    w = wq1+J12
    psi0 = tensor(basis(2, 1), basis(2, 0), basis(N, 0))
    t_list = np.linspace(0, 10 * 2 * np.pi / A, 5000)
    result = compute(psi0, t_list, A, w, 0, True)
    sz1list = expect(sz1, result.states)
    # t_pi1 = t_list[np.argmax(sz1list)]
    c1, f1 = fit(t_list, sz1list)
    t_pi1 = (2 * np.pi / f1) / 2.0
    t_pi2 = t_pi1/factor

    print t_pi1, t_pi2


# Main operations

# Quantum state tomography : z-axis projection
def tomo(initial_dm, final_dm):
    global process_stack, T_init, T_final

    # Store the rhos, for displaying later
    # if len(process_stack) >= 2:
    #     rhos = []
    # else:
    #     rhos = [initial_dm]
    rhos = []
    rho = final_dm    # initialize the ket we are going to work with

    # retrace all steps in process_stack, to go back to the initial state
    while process_stack:
        w, t = process_stack.pop()  # get parameters of the previous active-process
        T_init = T_final            # change universal time
        T_final += t

        t_list = np.linspace(T_init, T_final, 1000)
        rho = compute(rho, t_list, A, w, np.pi, True).states[-1]
        # store states for visualization
        rhos.append(rho)

    return rhos, fidelity(initial_dm, rho)


# Initialize (New process)
def reset():
    global T_init, T_final, process_stack
    T_init, T_final = 0.0, 0.0
    process_stack = []


# Excitation using a pulse
def rotation(rho0, w, pulse_time, p):
    global T_init, T_final, process_stack
    T_init = T_final
    T_final += pulse_time

    t_list = np.linspace(T_init, T_final, 1000)
    result = compute(rho0, t_list, A, w, p)

    process_stack.append([w, pulse_time])
    return result.states[-1]


# Go to state (|0> + |1>)|0>  -> half pulse width w1_u
# Then apply CNOT gate        -> full pulse width w2_l

def measure_bell_state():
    rho0 = tensor(fock(2, 0), fock(2, 0), fock(N, 0))  # state |00>
    rho1 = rotation(rho0, wq1+J12, t_pi1/2.0, 0.0)              # state (|0> + |1>)|0>

    half_ket = tensor((basis(2, 0) + basis(2, 1)).unit(), basis(2, 0), basis(N, 0))
    half_rho = half_ket * half_ket.dag()
    print fidelity(rho1, half_rho)

    rho_final = rotation(rho1, wq2-J12, t_pi2, 0.0)             # state (|00> + |11>)

    visualize_bloch(rho_final, rho1)

    rhos, fid = tomo(rho0, rho_final)
    visualize_bloch(rhos[-2], rhos[-1])
    return fid


#
# Visualisation
#

# To get qubit 1 coordinates
def coords_qubit1(current_state):
    sxlist1 = expect(sx1, current_state)
    sylist1 = expect(sy1, current_state)
    szlist1 = expect(sz1, current_state)

    return [sxlist1, sylist1, szlist1]


# To get qubit  coordinates
def coords_qubit2(current_state):
    sxlist2 = expect(sx2, current_state)
    sylist2 = expect(sy2, current_state)
    szlist2 = expect(sz2, current_state)

    return [sxlist2, sylist2, szlist2]


# Plot two states (eg: final and mid-process) on a Bloch sphere
def visualize_bloch(state_1, state_2):
    sphere = Bloch()

    # state_1 as points
    sphere.add_points(coords_qubit1(state_1))
    sphere.add_points(coords_qubit2(state_1))

    # state_2 as vectors
    sphere.add_vectors(coords_qubit1(state_2))
    sphere.add_vectors(coords_qubit2(state_2))

    sphere.show()

# get_pulse_time()
fid = measure_bell_state()
print 'The fidelity is found to be: ', fid
