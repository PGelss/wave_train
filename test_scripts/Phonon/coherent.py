import math
import numpy as np

# Set up simulation parameters
n_dim = 8           # size of basis set
red_m = 1           # reduced mass
omega = 1e-3        # harmonic frequency
delta = 20          # Initial displacement
num_steps = 25      # number of main time steps (200 for a full vibrational period)
step_size = np.pi / 100    # size of main time steps
sub_steps = 100     # number of sub steps

# Formulate Hamiltonian in terms of creation/annihilation operators
vec_sqrt = np.sqrt([i for i in range(1, n_dim)])
raising  = np.diag(vec_sqrt, -1)         # raising (creation) operator
lowering = np.diag(vec_sqrt, 1)          # lowering (annihilation) operator
identity = np.eye(n_dim)                 # identity operator
qu_numbr = raising @ lowering            # number operator
position = raising + lowering            # position operator
momentum = raising - lowering            # momentum operator
pos_squa = position @ position           # position squared
mom_squa = momentum @ momentum           # momentum squared
hamilton = qu_numbr + identity/2         # Hamilton operator
operator = -1j * hamilton                # RHS of TDSE multiplied by -1j
print (hamilton)

# Conversion factors for position and momentum
pos_conv = np.sqrt(1 / (2 * red_m * omega))
mom_conv = 1j * np.sqrt(red_m * omega / 2)

# Setup coherent state, displaced by delta
alpha  = 0.5 / pos_conv * delta
prefct = np.exp(-0.5 * alpha ** 2)
coeffs = prefct * np.array([alpha ** n / np.sqrt(math.factorial(n)) for n in range(n_dim)])
# print(str(alpha) + ": " + str(coeffs))

# Propagate in time using explicit Euler
psi = coeffs
for i in range(num_steps+1):

    # Propagate only when k>0
    if i != 0:

        # Time sub-stepping
        for k in range(sub_steps):
            psi = (identity + step_size/sub_steps * operator) @ psi

    # Expectation values from state vectors |psi>
    norm = np.conjugate(psi) @ psi
    nrgy = np.conjugate(psi) @ (hamilton @ psi)
    qu_n = np.conjugate(psi) @ (qu_numbr @ psi)
    pos1 = np.conjugate(psi) @ (position @ psi) * pos_conv
    mom1 = np.conjugate(psi) @ (momentum @ psi) * mom_conv
    pos2 = np.conjugate(psi) @ (pos_squa @ psi) * pos_conv**2
    mom2 = np.conjugate(psi) @ (mom_squa @ psi) * mom_conv**2
    posD = np.sqrt(pos2 - pos1 ** 2)  # uncertainty
    momD = np.sqrt(mom2 - mom1 ** 2)  # uncertainty

    print(42 * '-')
    print('step : ', i, ', time = ', i * step_size)
    print(42 * '-')
    print('norm = ', np.real_if_close(norm))
    print('nrgy = ', np.real_if_close(nrgy))
    print('qu_n = ', np.real_if_close(qu_n))
    print('pos  = ', np.real_if_close(pos1), ' +/- ', np.real_if_close(posD))
    print('mom  = ', np.real_if_close(mom1), ' +/- ', np.real_if_close(momD))
    print('DxDp = ', np.real_if_close(posD) * np.real_if_close(momD))  # uncertainty product
    print (' ')

    # Expectation values from density matrices rho := |psi><psi|
    rho = np.outer(psi, np.conjugate(psi))  # |ket><bra|
    norm = np.trace(rho)
    nrgy = np.trace(rho @ hamilton)
    qu_n = np.trace(rho @ qu_numbr)
    pos1 = np.trace(rho @ position) * pos_conv
    mom1 = np.trace(rho @ momentum) * mom_conv
    pos2 = np.trace(rho @ pos_squa) * pos_conv**2
    mom2 = np.trace(rho @ mom_squa) * mom_conv**2
    Dpos = np.sqrt(pos2 - pos1 ** 2)  # uncertainty
    Dmom = np.sqrt(mom2 - mom1 ** 2)  # uncertainty

    print('norm = ', np.real_if_close(norm))
    print('nrgy = ', np.real_if_close(nrgy))
    print('qu_n = ', np.real_if_close(qu_n))
    print('pos  = ', np.real_if_close(pos1), ' +/- ', np.real_if_close(posD))
    print('mom  = ', np.real_if_close(mom1), ' +/- ', np.real_if_close(momD))
    print('DxDp = ', np.real_if_close(posD) * np.real_if_close(momD))  # uncertainty product
    print (' ')
