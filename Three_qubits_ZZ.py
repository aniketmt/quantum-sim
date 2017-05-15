"""
Three qubits excited by waves sent to the cavity
Cavity photons have been ignored for now (2x2x2 Hilbert space)
Approximation 1: Cavity photons have been ignored for now (2x2 Hilbert space)
Approximation 2: Each qubit has been considered to be individually rotated
                (rotation of one does not affect the other)
Approximation 3: Third qubit not excited by input signal
Excitation strength for each qubit controlled by its cavity coupling
State tomography used for fidelity check
Universal time

Author: Aniket Maiti
"""


from qutip import *
from matplotlib import pyplot as plt
import numpy as np
from exp_decay_2 import fit


#
# CONSTANTS
#

# Qubit parameters
wq1 = 5.685 * 2*np.pi      # qubit-1 sigma_z coefficient
wq2 = 6.290 * 2*np.pi      # qubit-2 sigma_z coefficient
wq3 = 7.018 * 2*np.pi      # qubit-3 sigma_z coefficient

J12 = 0.200 * np.pi        # sigma-z sigma-z coupling bw qubit-1 and qubit-2
J23 = 0.0 * 0.253 * np.pi  # sigma-z sigma-z coupling bw qubit-2 and qubit-3
J31 = 0.0 * 0.232 * np.pi  # sigma-z sigma-z coupling bw qubit-3 and qubit-1
Jx12 = 0.0/100.0 * J12     # sigma-x sigma-x coupling bw qubit-1 and qubit-2
Jxxx = 0.0/100.0 * J12     # sigma-x coupling between all three


gq1 = 0.09 * 2*np.pi     # qubit-1 coupling to cavity
factor = 20.0/20.0
gq2 = factor * gq1       # qubit-2 coupling to cavity
gq3 = 0.0

# Decoherence
T1_1 = np.Infinity            # T1 for qubit-1
T2_1 = np.Infinity            # T2(E) for qubit-1
T1_2 = np.Infinity            # T1 for qubit-2
T2_2 = np.Infinity            # T2(E) for qubit-2

gammaq1 = 1/T1_1              # relaxation rate for qubit 1 =  1/T1(q1)
gammaq2 = 1/T1_2              # relaxation rate for qubit 2 =  1/T1(q2)
dphiq1 = 1/(2*T1_1) + 1/T2_1  # dephasing rate for qubit 1 =  1/T_phi(q1)
dphiq2 = 1/(2*T1_2) + 1/T2_2  # dephasing rate for qubit 2 =  1/T_phi(q2)

n_th = 0.0                    # average number of excitations ("temperature")

# Approximations
use_rwa = False        # Whether to use the Rotating Wave Approximation

# Drive Oscillations
A = 0.25               # Drive amplitude
t_pi1 = 22.213         # Found to have best fidelity
t_pi2 = t_pi1/factor   # For second qubit, since coupling is different

# Tomography
T_init, T_final = 0.0, 0.0  # Will keep universal time
process_stack = []          # To retrace processes via tomography


#
# Operators
#

# Qubit energies and coupling
sz1 = tensor(sigmaz(), qeye(2), qeye(2))
sz2 = tensor(qeye(2), sigmaz(), qeye(2))
sz3 = tensor(qeye(2), qeye(2), sigmaz())

z12 = tensor(sigmaz(), sigmaz(), qeye(2))
z23 = tensor(qeye(2), sigmaz(), sigmaz())
z31 = tensor(sigmaz(), qeye(2), sigmaz())

x12 = tensor(sigmax(), sigmax(), qeye(2))
xxx = tensor(sigmax(), sigmax(), sigmax())

# Driving terms
sx1 = tensor(sigmax(), qeye(2), qeye(2))
sx2 = tensor(qeye(2), sigmax(), qeye(2))

# Collapse terms
sm1 = tensor(destroy(2), qeye(2), qeye(2))
sm2 = tensor(qeye(2), destroy(2), qeye(2))

# Visualization
sy1 = tensor(sigmay(), qeye(2), qeye(2))
sy2 = tensor(qeye(2), sigmay(), qeye(2))


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


#
# Hamiltonian
#

H0_ideal = wq1/2.0 * sz1 + wq2/2.0 * sz2 + wq3/2.0 * sz3 + J12/2.0 * z12 + J23/2.0 * z23 + J31/2.0 * z31
H0 = H0_ideal + Jxxx / 2.0 * xxx + Jx12 / 2.0 * x12


# Hamiltonian to only rotate the first qubit [ideal implies no energy exchange or leakage]
def excite1(psi0, tlist, A, w, p, ideal=False):
    if not use_rwa:
        H1 = gq1 * sx1
        # define the time-dependence of the hamiltonian using the list-string format
        args = {'A': A, 'w': w, 'p':p}
        Ht = [H0, [H1, "A * cos(w*t + p)"]]
        Ht_ideal = [H0_ideal, [H1, "A * cos(w*t + p)"]]
        if ideal:
            output = mesolve(Ht_ideal, psi0, tlist, [], [], args)  # , options=Options(nsteps=10000))
        else:
            output = mesolve(Ht, psi0, tlist, cops, [], args)  # , options=Options(nsteps=10000))

    else:
        # Rotating Wave Approx
        H_rwa_ideal = J12/2.0 * z12 + J23/2.0 * z23 + J31/2.0 * z31 + A * gq1 * sx1
        H_rwa = H_rwa_ideal + Jxxx/2.0 * xxx + Jx12 / 2.0 * x12
        if ideal:
            output = mesolve(H_rwa_ideal, psi0, tlist, [], [])  # , options=Options(nsteps=10000))
        else:
            output = mesolve(H_rwa, psi0, tlist, cops, [])  # , options=Options(nsteps=10000))

    return output


# Hamiltonian to only rotate the second qubit [ideal implies no energy exchange or leakage]
def excite2(psi0, tlist, A, w, p, ideal=False):
    if not use_rwa:
        H1 = gq2 * sx2
        # define the time-dependence of the hamiltonian using the list-string format
        args = {'A': A, 'w': w, 'p':p}
        Ht = [H0, [H1, "A * cos(w*t + p)"]]
        Ht_ideal = [H0_ideal, [H1, "A * cos(w*t + p)"]]
        if ideal:
            output = mesolve(Ht_ideal, psi0, tlist, [], [], args)  # , options=Options(nsteps=10000))
        else:
            output = mesolve(Ht, psi0, tlist, cops, [], args)  # , options=Options(nsteps=10000))

    else:
        # Rotating Wave Approx
        H_rwa_ideal = J12/2.0 * z12 + J23/2.0 * z23 + J31/2.0 * z31 + A * gq2 * sx2
        H_rwa = H_rwa_ideal + Jxxx/2.0 * xxx + Jx12 / 2.0 * x12
        if ideal:
            output = mesolve(H_rwa_ideal, psi0, tlist, [], [])  # , options=Options(nsteps=10000))
        else:
            output = mesolve(H_rwa, psi0, tlist, cops, [])  # , options=Options(nsteps=10000))

    return output


#
# Operations on System
#

# Quantum state tomography : z-axis projection
def tomo(initial_dm, final_dm):
    global process_stack, T_init, T_final

    # Store the rhos, for displaying later
    if len(process_stack) >= 2:
        rhos = []
    else:
        rhos = [initial_dm]
    rho = final_dm  # initialize the ket we are going to work with

    # retrace all steps in process_stack, to go back to the initial state
    while process_stack:
        w, t, q = process_stack.pop()  # get parameters of the previous active-process
        T_init = T_final               # change universal time
        T_final += t
        if q == 1:
            # qubit 1
            t_list = np.linspace(T_init, T_final, 500)
            rho = excite1(rho, t_list, A, w, np.pi, True).states[-1]
        elif q == 2:
            # qubit 2
            t_list = np.linspace(T_init, T_final, 500/factor)
            rho = excite2(rho, t_list, A, w, np.pi, True).states[-1]
        # store states for visualization
        rhos.append(rho)

    return rhos, fidelity(initial_dm, rho)


# Get the graph for amplitudes of oscillation, displaying resonances four resonances
# Takes too long
def check_w1_ul():
    psi0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    psi1 = tensor(basis(2, 0), basis(2, 0), basis(2, 1))
    psi2 = tensor(basis(2, 0), basis(2, 1), basis(2, 0))
    psi3 = tensor(basis(2, 0), basis(2, 1), basis(2, 1))

    w_list = np.linspace(0, 2*wq1/(2*np.pi), 20)
    t_list = np.linspace(0, 2*np.pi/A, 100)

    amps = []

    for psi in [psi0, psi1, psi2, psi3]:
        for w in w_list:
            amp_list = []
            result = excite1(psi, t_list, A, 2*np.pi*w, 0, True)
            sz1list = expect(sz1, result.states)
            amp_list.append(np.max(sz1list)-np.min(sz1list))

            amps.append(amp_list)

    for y_data in amps:
        plt.plot(w_list, y_data)
    plt.show()


# Display the entire time evolution of qubit 1 for a particular frequency w
def check_w(w):
    psi0 = tensor(basis(2, 1), basis(2, 0), basis(2, 0))

    t_list = np.linspace(0, 5*2 * np.pi / A, 500)
    result = excite1(psi0, t_list, A, w, 0)
    sz1list = expect(sz1, result)

    fig, axes = plt.subplots(1, 1)
    axes.plot(t_list, sz1list)
    axes.set_ylim([-1.1, 1.1])
    plt.show()


# get t_pi
def get_pulse_time():
    global t_pi1, t_pi2
    w = wq1+J12
    psi0 = tensor(basis(2, 1), basis(2, 0))
    t_list = np.linspace(0, 10 * 2 * np.pi / A, 5000)
    result = excite1(psi0, t_list, A, w, 0, True)
    sz1list = np.array(expect(sz1, result.states))
    # t_pi = t_list[np.argmax(sz1list)]
    c, f = fit(t_list, sz1list)
    t_pi1 = (2*np.pi/f)/2.0
    t_pi2 = t_pi1/factor


# Main processes

# Initialize (New process)
def reset():
    global T_init, T_final, process_stack
    T_init, T_final = 0.0, 0.0
    process_stack = []


# Rest
def rest(psi0, rest_t):
    global T_init, T_final
    T_init = T_final
    T_final += rest_t
    tlist = np.linspace(T_init, T_final, 500)
    H = H0
    output = mesolve(H, psi0, tlist, cops, [])
    return output.states[-1]


# Rotate qubit 1 using half pulse width
def half_rotation(rho0, w):
    global T_init, T_final, process_stack
    T_init = T_final
    T_final += t_pi1/2.0
    t_list = np.linspace(T_init, T_final, 500)
    result = excite1(rho0, t_list, A, w, 0.0)

    process_stack.append([w, t_pi1 / 2.0, 1])
    return result.states


# Rotate qubit2 using full pulse width
def full_rotation(rho0, w):
    global T_init, T_final, process_stack
    T_init = T_final
    T_final += t_pi2
    t_list = np.linspace(T_init, T_final, 500/factor)
    result = excite2(rho0, t_list, A, w, 0.0)

    process_stack.append([w, t_pi2, 2])
    return result.states


# Go to state (|0> + |1>)|0>  -> half pulse width w1_u
# Then apply CNOT gate        -> full pulse width w2_l

def measure_ghz_state():
    # starting new set of rotations
    reset()

    # Process
    psi0 = tensor(fock(2, 0), fock(2, 0), fock(2, 0))  # state |00>
    psi_half = half_rotation(psi0, wq1+J12)[-1]        # Go to equator
    psi_final = full_rotation(psi_half, wq2-J12)[-1]   # Apply CNOT on second -> GHZ state

    # visualizing the two
    print 'Final states (before tomography)'
    visualize_bloch(psi_final, psi_half)

    # Finding effect of decoherence -> tomography
    rhos, fid = tomo(psi0, psi_final)
    print 'States on projecting onto z-axis'
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


print measure_ghz_state()
