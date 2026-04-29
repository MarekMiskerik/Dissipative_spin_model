import mpmath
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sci
import time

'''https://doi.org/10.1103/PhysRevA.106.L010201'''

## Define parameters
j = 10 # total spin
N = 2*j + 1 # number of particles considering j = j_MAX

# define the Liouvillian superoperator L
# a is parametr of interaction strength in Hamiltonian
# parameters C, C0, p are related to the dissipation
# critical point a_c = h/2
def L(p, h, C, C0, a, s = j):
    K1z = spre(jmat(s, 'z'))
    K1x = spre(jmat(s, 'x'))
    K1p = spre(jmat(s, '+'))
    K1m = spre(jmat(s, '-'))

    K2z = spost(jmat(s, 'z').dag())
    K2x = spost(jmat(s, 'x').dag())
    K2p = spost(jmat(s, '+').dag())
    K2m = spost(jmat(s, '-').dag())

    Kzm = K1z - K2z
    Kzp = K1z + K2z
    Kz = K1z * K2z
    Kp = K1p * K2p
    Km = K1m * K2m
    # the term 1j * a / s * (K1x**2 - K2x**2) corresponds to interaction of spins
    return - C * (s + 1) - 1j * h * Kzm + 1j * a / s * (K1x**2 - K2x**2) + C / s * Kz + (C - C0) * 0.5 / s * Kzm**2 - C * 0.5 * p / s * Kzp + C * (1 - p) * 0.5 / s * Kp + C * (1 + p) * 0.5 / s * Km
# care about the type of object it returns. it should be a superoperator

# returns function f as an operator function f(A)
# second argument needs to be specified as a lambda expression
def oper_func(A, f):
    evals, eigvecs = A.eigenstates()
    fA = sum([f(evals[i]) * eigvecs[i] * eigvecs[i].dag() for i in range(len(evals))])
    return fA

# effective non-Hermitian Hamiltonian
# is not a superoperator
def H_eff(p, h, C, C0, a, s=j):
    H = -h * jmat(s, 'z') + a / s * jmat(s, 'x')**2
    L0 = np.sqrt(C0/s) * jmat(s, 'z')
    Lp = np.sqrt(C * (1 - p) / (2 * s)) * jmat(s, '+')
    Lm = np.sqrt(C * (1 + p) / (2 * s)) * jmat(s, '-')
    D = 1j * (Lp * Lp + Lm * Lm + L0 * L0)
    return H - D

#H_eff_evals = H_eff(p = 0.1, h = 1.0, C = 0.1, C0 = 0.1, a = 2).eigenenergies()

'''plt.scatter(H_eff_evals.real / j, H_eff_evals.imag / j, color='black', s=4)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of the effective Hamiltonian')
plt.show()'''

## Compute and visualize the spectrum of the Liouvillian

# defining an operator Km = K1z - K2z to find the block structure of the Liouvillian as [L, Km] = 0
def Km(s=j):
    K1z = spre(jmat(s, 'z'))
    K2z = spost(jmat(s, 'z').dag())
    return K1z - K2z

# operator K^2 = K_1^2 * K_2^2 or tensor product of J^2 and (J^2)^T
# this operator is a strong symmetry because J^2 on Hilbert space commutes with H and all L_i
def Ksq(s=j):
    J2x = jmat(s, 'x')**2
    J2y = jmat(s, 'y')**2
    J2z = jmat(s, 'z')**2
    J2 = J2x + J2y + J2z
    Ksq1 = spre(J2)
    Ksq2 = spost(J2.dag())
    return Ksq1 * Ksq2

# operator K_p^2= K_1^2 + K_2^2
# this operator is a weak symmetry
def Ksqp(s=j):
    J2x = jmat(s, 'x')**2
    J2y = jmat(s, 'y')**2
    J2z = jmat(s, 'z')**2
    J2 = J2x + J2y + J2z
    Ksq1 = spre(J2)
    Ksq2 = spost(J2.dag())
    return Ksq1 + Ksq2

# NOTE: both Ksq and Ksqp useless for block diagonalization of L as we are restrected to j = const.
#       not used below

# parity operator (-1)^(J_z + j) extended to the superoperator space
# commutes with L
def P(s=j):
    Parity = oper_func(jmat(s, 'z'), lambda x: 1 if x % 2 == 0 else -1)
    return spre(Parity) * spost(Parity.dag())

#print(L(j,p))
#print(Ksq(j))
#print(Ksqp(j))

# diagonalize operator L in blocks given by the eigenvalues of operator K assuming [L,K] = 0, not very general yet
# should add something that finds the sizes of the blocks for general K
# possible to calculate in higher presicion using mpmath
def block_eigvals(L, K, precision = 16):
    # verify that L and K commute
    comm_norm = commutator(L, K).norm() # by default qutip takes trace norm Tr(sqrt(A.dag() * A))
    if comm_norm > 1e-10:
        raise ValueError("Operators do not commute")
    evalsK, eigvecsK = K.eigenstates()

    #print(evalsK)
    L_M = L.transform(eigvecsK) # matrix representation of L in the basis of eigenvectors of K
    L_M = L_M.full() # convert to numpy array

    evals = [] # array to store the eigenvalues of L

    # finding the sizes of the blocks by couting the unique eigenvalues of K
    rounded_evalsK = np.round(evalsK, decimals=5) # round to avoid numerical issues with very close eigenvalues
    _, block_sizes = np.unique(rounded_evalsK, return_counts=True) # 2nd argument returns the counts of each unique value
    #print(block_sizes)
    
    # loop to find the eigenvalues for each block
    n = 0
    for size in block_sizes:
        block = L_M[n:n+size, n:n+size]
        #print(block)
        if precision == 16:
            evals_block = sci.eigvals(block)
        else:
            import mpmath
            mpmath.mp.dps = precision
            block = mpmath.matrix(block)
            evals_block, _ = mpmath.eig(block)
        evals.append(evals_block)
        n += size

    all_evals = np.concatenate(evals)

    return np.array([complex(val) for val in all_evals])

start = time.time()
evals = block_eigvals(L(p = 0.99, h = 1.0, C = 1, C0 = 0.0, a = 0.0), Km())
#evals = L(p = 0.99, h = 1.0, C = 1, C0 = 0.0, a = 0.0).eigenenergies()
end = time.time()
print("Time taken to compute the spectrum: ", end - start)

'''plt.scatter(evals.real / j, evals.imag / j, color='black', s=4)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of the Liouvillian')
plt.show()'''

## Separating spectrum into exceptional and normal points to see the Liouvillian spectral phase transition (LSPT)
def spec_separation(evals, threshold): # the threshold should decrease with the system size
    excvals = np.array([])
    normvals = np.array([])

    for a in evals:
        normal = True
        i = 0
        while normal == True and i < len(evals):
            if abs(a - evals[i]) < threshold and a != evals[i]:
                normal = False
            i += 1
        if normal:  
            normvals = np.append(normvals, a)
        else:
            excvals = np.append(excvals, a)
    
    return excvals, normvals

start = time.time()
excvals, normvals = spec_separation(evals, 1e-2)
end = time.time()
print("Time taken to separate the spectrum: ", end - start)

#print(expvals)
#print(normvals)

plt.scatter(excvals.real / j, excvals.imag / j, color='red', s=4)
plt.scatter(normvals.real / j, normvals.imag / j, color='blue', s=4)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of the Liouvillian with the LSPT')
plt.show()


## Spectrum for M = 0 sector and different values of p

# function that finds a projection of an operator O onto a subspace generated by vecs
# NOTE: does not work with '.proj', most likely because '.eigenstates' does not return 'ket' or 'bra' objects for superoperators
def operator_proj(O, vecs):
    if not vecs:
        return 0 * O
    P = sum([v * v.dag() for v in vecs])
    return P * O * P.dag()


ps = np.linspace(0, 1, 20)

# Lists to collect our coordinates for plotting
all_x = []
all_y = []

'''for p in ps:
    eigvals = operator_proj(L(j,p), [eigvecsM[i] for i in Mzero]).eigenenergies()
    
    x_vals = eigvals.real / j
    y_vals = np.full_like(x_vals, p)
    all_x.extend(x_vals)
    all_y.extend(y_vals)

plt.plot(all_x, all_y, 'k.', markersize=2)
plt.ylabel('p', fontsize=14)
plt.xlabel('Real part of eigenvalues', fontsize=14)
plt.show()'''
# NOTE: does seem good, but some points in the plot seems to be closer comparing with the paper


## Spectrum for M = 0 sector and different values of p with separation of normal and exceptional points
norm_x, norm_y = [], []
exc_x, exc_y = [], []

'''for p in ps:
    eigvals = operator_proj(L(j,p), [eigvecsM[i] for i in Mzero]).eigenenergies()
    
    eigexc, eignorm = spec_separation(eigvals, 0.1)

    n_x_vals = eignorm.real / j
    n_y_vals = np.full_like(n_x_vals, p)
    norm_x.extend(n_x_vals)
    norm_y.extend(n_y_vals)

    e_x_vals = eigexc.real / j
    e_y_vals = np.full_like(e_x_vals, p)
    exc_x.extend(e_x_vals)
    exc_y.extend(e_y_vals)

plt.plot(norm_x, norm_y, 'b.', markersize=2, label='Normal')
plt.plot(exc_x, exc_y, 'r.', markersize=2, label='Exceptional')
plt.ylabel('p', fontsize=14)
plt.xlabel('Real part of eigenvalues', fontsize=14)
plt.legend(loc='best')
plt.show()'''
# NOTE: need to adjust the threshold for separation