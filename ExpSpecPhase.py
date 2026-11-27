import mpmath
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sci
import time
import os
from scipy.sparse.linalg import eigs
import scipy.sparse as sps
import sympy as sp
from scipy.sparse.csgraph import connected_components

## Define parameters

# define the Liouvillian superoperator L
# s is total spin of the system
# a is parametr of interaction strength in Hamiltonian
# parameters C, C0, p are related to the dissipation
# critical point a_c = h/2
def Liouvillian(s, h, C, p, C0, a):
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
    # the term 1j * a / s * (K1x**2 - K2x**2) corresponds to the interaction of spins
    return - C * (s + 1) - 1j * h * Kzm + 1j * a / s * (K1x**2 - K2x**2) + C / s * Kz + (C - C0) * 0.5 / s * Kzm**2 - C * 0.5 * p / s * Kzp + C * (1 - p) * 0.5 / s * Kp + C * (1 + p) * 0.5 / s * Km
# care about the type of object it returns. it should be a superoperator

# Liouvillian definiton using a specific function from QuTiP, should be faster
# TODO: use this instead of the above definition, but check if it returns the same object type
def true_Liouvillian(s, p, h, C, C0, a):
    H = -h * jmat(s, 'z') + a / s * jmat(s, 'x')**2
    L0 = np.sqrt(C0/s) * jmat(s, 'z')
    Lp = np.sqrt(C * (1 - p) / (2 * s)) * jmat(s, '+')
    Lm = np.sqrt(C * (1 + p) / (2 * s)) * jmat(s, '-')
    return liouvillian(H, [L0, Lp, Lm])


# returns function f as an operator function f(A)
# second argument needs to be specified as a lambda expression
def oper_func(A, f):
    evals, eigvecs = A.eigenstates()
    fA = sum([f(evals[i]) * eigvecs[i] * eigvecs[i].dag() for i in range(len(evals))])
    return fA

# effective non-Hermitian Hamiltonian
# is not a superoperator
def H_eff(s, p, h, C, C0, a):
    H = -h * jmat(s, 'z') + a / s * jmat(s, 'x')**2
    L0 = np.sqrt(C0/s) * jmat(s, 'z')
    Lp = np.sqrt(C * (1 - p) / (2 * s)) * jmat(s, '+')
    Lm = np.sqrt(C * (1 + p) / (2 * s)) * jmat(s, '-')
    D = 1j * (Lp * Lp + Lm * Lm + L0 * L0)
    return H - D


## Compute the spectrum of the Liouvillian

# defining an operator Km = K1z - K2z to find the block structure, only for a = 0
# [L,Km] = 0 for a = 0
def Km(s):
    K1z = spre(jmat(s, 'z'))
    K2z = spost(jmat(s, 'z').dag())
    return K1z - K2z

# operator K^2 = K_1^2 * K_2^2 or tensor product of J^2 and (J^2)^T
# this operator is a strong symmetry because J^2 on Hilbert space commutes with H and all L_i
# not really useful as we are working in fixed j sector
def Ksq(s):
    J2x = jmat(s, 'x')**2
    J2y = jmat(s, 'y')**2
    J2z = jmat(s, 'z')**2
    J2 = J2x + J2y + J2z
    Ksq1 = spre(J2)
    Ksq2 = spost(J2.dag())
    return Ksq1 * Ksq2

# operator K_p^2= K_1^2 + K_2^2
# this operator is a weak symmetry
# not really useful as we are working in fixed j sector
def Ksqp(s):
    J2x = jmat(s, 'x')**2
    J2y = jmat(s, 'y')**2
    J2z = jmat(s, 'z')**2
    J2 = J2x + J2y + J2z
    Ksq1 = spre(J2)
    Ksq2 = spost(J2.dag())
    return Ksq1 + Ksq2

# NOTE: both Ksq and Ksqp useless for block diagonalization of L as we are restrected to j = const.

# parity operator (-1)^(J_z + j) extended to the superoperator space
# commutes with L
# probably better to use expm()
def P(s):
    Parity = oper_func(jmat(s, 'z'), lambda x: 1 if x % 2 == 0 else -1)
    return spre(Parity) * spost(Parity.dag())

# diagonalize operator L in blocks given by the eigenvalues of operator K assuming [L,K] = 0
# possible to calculate in higher presicion using mpmath
# this version only resturns the eigenvalues of L and does not distinguish between the blocks
def block_diagonalization(L, K, precision = 16):
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

# block diagonalization of L in blocks given by the eigenvalues of K assuming [L,K] = 0
# this version returns a dictionary with the eigenvalues of L for each block given by the eigenvalue of K
def show_block_diagonalization(L, K, precision=16):
    # verify that L and K commute
    comm_norm = commutator(L, K).norm()
    if comm_norm > 1e-10:
        raise ValueError("Operators do not commute")
    evalsK, eigvecsK = K.eigenstates()

    L_M = L.transform(eigvecsK) # matrix representation of L in the basis of K
    L_M = L_M.full() # convert to numpy array

    # dictionary to store the eigenvalues of L mapped to the eigenvalue of K (the block)
    block_evals_dict = {} 

    # find the sizes of the blocks and the unique eigenvalues of K
    rounded_evalsK = np.round(evalsK, decimals=5)
    
    # capture both the unique eigenvalues and their counts
    unique_evalsK, block_sizes = np.unique(rounded_evalsK, return_counts=True) 
    
    # loop to find the eigenvalues for each block
    n = 0
    for k_eval, size in zip(unique_evalsK, block_sizes):
        block = L_M[n:n+size, n:n+size]
        
        if precision == 16:
            evals_block = sci.eigvals(block)
        else:
            import mpmath
            mpmath.mp.dps = precision
            block_mp = mpmath.matrix(block)
            evals_block, _ = mpmath.eig(block_mp)
        
        evals_block = np.array([complex(val) for val in evals_block])

        # store the L eigenvalues under the corresponding K eigenvalue key
        block_evals_dict[k_eval] = evals_block
        n += size

    return block_evals_dict


# the three functions below aim to calculate low-lying spectrum, i.e. the eigenvalues nearest to zero 

# convert a QuTiP superoperator to a scipy sparse matrix (works for QuTiP 4 and 5)
def to_sparse(L):
    try:
        # .to('csr') coverts QuTiP 5 superoperator to a compressed sparse row matrix but leave it as a QuTiP object
        # .data_as('csr_matrix') converts the QuTiP object to a scipy sparse matrix
        # can probably use onlt data_as('csr_matrix') but this is safer
        return L.to('csr').data_as('csr_matrix')  # QuTiP 5
    except AttributeError:
        return L.data  # QuTiP 4, already a scipy csr matrix

# computes only the k eigenvalues of L
# sigma slightly off zero, because L has an exact zero eigenvalue (steady state), so (L - 0) would be singular
def low_lying_spectrum(L, k, sigma=1e-3, eigenvectors=False):
    Ls = to_sparse(L).tocsc()
    # k is the number of eigenvalues to compute, sigma is the shift L - sigma*I, which='LM' means we want the eigenvalues with largest magnitude
    if eigenvectors == False:
        evals = eigs(Ls, k=k, sigma=sigma, which='LM', return_eigenvectors=False)
        return evals[np.argsort(-evals.real)]  # sort: slowest decaying first
    else:
        evals, evecs = eigs(Ls, k=k, sigma=sigma, which='LM', return_eigenvectors=True)
        idx = np.argsort(-evals.real)
        return evals[idx], evecs[:, idx]  # sort: slowest decaying first


def low_lying_block_diagonalization(L, K, k, sigma=1e-3):
    # verify that L and K commute
    '''comm_norm = commutator(L, K).norm()
    if comm_norm > 1e-10:
        raise ValueError("Operators do not commute")'''
    evalsK, eigvecsK = K.eigenstates()

    L_M = L.transform(eigvecsK) # matrix representation of L in the basis of K
    L_M = L_M.full() # convert to numpy array

    # dictionary to store the eigenvalues of L mapped to the eigenvalue of K (the block)
    block_evals_dict = {} 

    # find the sizes of the blocks and the unique eigenvalues of K
    rounded_evalsK = np.round(evalsK, decimals=5)
    
    # capture both the unique eigenvalues and their counts
    unique_evalsK, block_sizes = np.unique(rounded_evalsK, return_counts=True) 
    
    # loop to find the eigenvalues for each block
    n = 0
    for k_eval, size in zip(unique_evalsK, block_sizes):
        block = L_M[n:n+size, n:n+size]
        
        evals_block = eigs(block, k=k, sigma=sigma, which='LM', return_eigenvectors=False)
        
        evals_block = np.array([complex(val) for val in evals_block])

        # store the L eigenvalues under the corresponding K eigenvalue key
        block_evals_dict[k_eval] = evals_block
        n += size

    return block_evals_dict


def parity_block_low_lying_spectrum(L, n, k, sigma=1e-3, check=True):
    def parity_sector_indices(n):
        """Indices of the vectorized density matrix in each parity sector.
        n = 2j+1. Column stacking: entry rho[i, k] has index i + n*k,
        and parity (-1)^(m - m') = (-1)^(i - k)."""
        idx = np.arange(n * n)
        i, k = idx % n, idx // n
        even = idx[(i - k) % 2 == 0]
        odd = idx[(i - k) % 2 == 1]
        return even, odd
    Ls = to_sparse(L)                    # for a QuTiP object, use to_sparse(L)
    even, odd = parity_sector_indices(n)
    if check:                                  # the off-diagonal blocks must vanish
        off = max(abs(Ls[even][:, odd]).max(), abs(Ls[odd][:, even]).max())
        if off > 1e-12:
            raise ValueError(f"L does not commute with parity (largest coupling {off:.2e})")
    out = {}
    for P, idx in ((+1, even), (-1, odd)):
        B = Ls[idx][:, idx].tocsc()
        ev = eigs(B, k=k, sigma=sigma, which='LM', return_eigenvectors=False)
        out[P] = ev[np.argsort(-ev.real)]
    return out


## Searching for coincidental eigenvalues in the spectrum
# coincidental eigenvalues are those that are identical or very close to each other in our case
# there are two types of coincidental eigenvalues: degenerate and coalescent
# degenerate eigenvalues are those that have linearly independent eigenvectors
# coalescent eigenvalues are those that have linearly dependent eigenvectors, i.e. they correspond to nontrivial Jordan blocks

# simple function that just separates the coincidental eigenvalues from the non-coincidental ones
# was used initially, not used in currect version of the code, but kept for reference
def spec_separation(evals, threshold): # the threshold should decrease with the system size
    coincidences = np.array([], dtype=complex)
    non_coincidences = np.array([], dtype=complex)

    for a in evals:
        unique = True
        i = 0
        while unique == True and i < len(evals):
            if abs(a - evals[i]) < threshold and a != evals[i]:
                unique = False
            i += 1
        if unique:  
            non_coincidences = np.append(non_coincidences, a)
        else:
            coincidences = np.append(coincidences, a)
    
    return coincidences, non_coincidences

# a function that finds coincidences and distinguishes coalescences from degeneracies
# does work only for double coincidences
# was written by me, but not used in the current version of the code, kept for reference
def spectrum_double_coincidences(A, coincident_threshold, coalescent_threshold): # the threshold should decrease with the system size
    evals, evecs = A.eigenstates()

    coincidences_index = np.array([], dtype=int) # indeces of coincidential eigenvalues
    non_coincidens = np.array([], dtype=complex) # array to store non-coincidential eigenvalues
    # loop over all eigenvalues to find coincidential ones
    # goes through all pairs twice, should be improved to avoid this 
    for i in range(len(evals)):
        unique = True
        j = 0
        while unique == True and j < len(evals):
            if abs(evals[j] - evals[i]) < coincident_threshold and j != i:
                unique = False
            j += 1
        if not unique:
            coincidences_index = np.append(coincidences_index, i) # only the indeces are stored
        else:
            non_coincidens = np.append(non_coincidens, evals[i])

    coincident_evecs = evecs[coincidences_index] # make a list of coincidential eigenvectors and eigenvalues
    coincident_evals = evals[coincidences_index]
    coalescences = np.array([], dtype=complex) # array to store coalescent eigenvalues
    degenerecies = np.array([], dtype=complex) # array to store degenerate eigenvalues

    # separation of coalescent and degenerate eigenvalues
    # we run through all pairs of coincidential eigenvectors and check their overlap
    # coalescent eigenvectors are those that are linearly dependent, i.e. their overlap is 1 or -1 as QuTiP gives normalized eigenvectors
    for i in range(len(coincident_evecs)): 
        for j in range(i + 1, len(coincident_evecs)):
            overlap = coincident_evecs[i].dag() * coincident_evecs[j]
            # have to compare eigenvectors corresponding to the same eigenvalue as the list of eigenvalues includes pairs of coincidential eigenvalues
            if abs(coincident_evals[i] - coincident_evals[j]) < coincident_threshold: 
                if abs(abs(overlap) - 1) < coalescent_threshold:
                    # storing both close eigenvalues
                    coalescences = np.append(coalescences, coincident_evals[i]) 
                    coalescences = np.append(coalescences, coincident_evals[j])
                else:
                    degenerecies = np.append(degenerecies, coincident_evals[i])
                    degenerecies = np.append(degenerecies, coincident_evals[j])

    return non_coincidens, degenerecies, coalescences
    

def spectrum_coincidences(evals, evecs, coincident_threshold, rank_tol=1e-3):
    # converts to scipy array if the eigenvectors are QuTiP kets, otherwise assumes they are already in a suitable format
    if len(evecs) > 0 and hasattr(evecs[0], "full"):
        evecs = np.column_stack([ket.full().ravel() for ket in evecs])
    
    V = evecs / np.linalg.norm(evecs, axis=0) # normalize eigenvectors
    # evlas[:, None].shape is (1,k) and evlas[None, :].shape is (k,1) 
    dist = np.abs(evals[:, None] - evals[None, :]) # distance matrix of shape (k,k), zeros on diagonal
    # dist < coincident_threshold sepresates a boolean matrix or equivalently a graph
    # connected_components finds a connected elements of this graph
    # n_comp: number of clusters, label is an array of length k: label[i] is an index of the cluster corresponding to eigenvalue i
    n_comp, label = connected_components(dist < coincident_threshold, directed=False)
    #print(dist) 

    non_coincidences, degeneracies, coalescences, missing = [], [], [], []
    singular_values = []
    for comp in range(n_comp):
        members = np.where(label == comp)[0] # indeces of eigenvalues in current cluster comp
        if len(members) == 1: # cluster of size 1 means no coincidences
            non_coincidences.append(evals[members[0]])
            continue
        # returns singular values of eigenvector matrix
        # compute_uv = False returns only sorted singular values
        s = np.linalg.svd(V[:, members], compute_uv=False)
        singular_values.append(s)
        n_missing = int(np.sum(s < rank_tol * s[0])) # counts how many of the singular values are small compared to the largest one
        if n_missing == 0: # with no small singular value, the cluster corresponds to degeneracy
            degeneracies.extend(evals[members])
            if len(members) > 2:
                print("degeneracy of order", len(members), "found for eigenvalue", evals[members[0]])
        else:
            coalescences.extend(evals[members])
            missing.extend([n_missing] * len(members))

    # returns the arrays of non-coincidental eigenvalues, degenerate eigenvalues, coalescent eigenvalues, and the number of missing eigenvectors for each coalescent eigenvalue
    return (np.array(non_coincidences, dtype=complex),
            np.array(degeneracies, dtype=complex),
            np.array(coalescences, dtype=complex),
            np.array(missing, dtype=int),
            np.array(singular_values, dtype=float))

def eig_dif(q_evals, clas_evals, zero_tol=1e-2):
    q = np.asarray(q_evals)
    cl = np.asarray(clas_evals)

    # remove the steady-state zero eigenvalue(s)
    q = q[np.abs(q) > zero_tol]
    cl = cl[np.abs(cl) > zero_tol]

    # Liouvillian gap: slowest-decaying nonzero quantum mode
    q_gap = q[np.argmin(np.abs(q.real))]

    # classical eigenvalue closest to it in the complex plane
    cl_match = cl[np.argmin(np.abs(cl - q_gap))]

    return abs(q_gap - cl_match), q_gap, cl_match


## Visualization and analysis of the spectrum

# a procedure to plot some results, mainly for testing purposes
# should be modified to plot the results of te spectrum_coincidences function
def spectrum_coincidences_plotting(evals, evecs, coincident_threshold, rank_tol=1e-3):
    start = time.time()
    non_coincidens, degenerecies, coelescences, missing = spectrum_coincidences(evals, evecs, coincident_threshold, rank_tol)
    end = time.time()
    print("Time taken to compute and separate the spectrum: ", end - start)

    plt.scatter(non_coincidens.real, non_coincidens.imag, color='blue', s=4)
    plt.scatter(degenerecies.real, degenerecies.imag, color='limegreen', s=4)
    plt.scatter(coelescences[missing == 1].real, coelescences[missing == 1].imag, color='red', s=4)
    plt.scatter(coelescences[missing > 1].real, coelescences[missing > 1].imag, color='orange', s=4)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Spectrum of the Liouvillian')
    plt.show()

# making plots of the full spectrum for different parameters and saving them to a folder
def figure_making(j):
    output_dir = "deg&coa"
    os.makedirs(output_dir, exist_ok=True)

    Cs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    #Cs = np.array([0.1])
    #C0 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    C0s = np.array([0.0])
    ps = np.array([0.1, 0.5, 0.99])
    #ps = np.array([0.0])
    aa = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    #aa = np.array([0.8])

    for C in Cs:
        for C0 in C0s:
            for p in ps:
                for a in aa:
                    L = Liouvillian(s = j, h = 1.0, C = C, p = p, C0 = C0, a = a)
                    non_coincidens, degenerecies, coelescences = spectrum_coincidences(L.eigenstates(), 1e-2, 1e-2)
                    plt.scatter(non_coincidens.real / j, non_coincidens.imag / j, color='blue', s=4)
                    plt.scatter(degenerecies.real / j, degenerecies.imag / j, color='limegreen', s=4)
                    plt.scatter(coelescences.real / j, coelescences.imag / j, color='red', s=4)
                    plt.xlabel('Real Part')
                    plt.ylabel('Imaginary Part')
                    plt.title('j = {}; C = {}, C0 = {}, p = {}, a = {}'.format(j, C, C0, p, a))
                    #plt.show()
                    filename = f"spectrum_C{C}_C0{C0}_p{p}_a{a}.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath)
                    plt.close() # close everytime


# visualizes the block diagonalization of L in blocks given by the eigenvalues of K
def visualize_block_diagonalization(L, K, j):
    evals_dict = show_block_diagonalization(L, K, precision=16)
    for i, (key, evals) in enumerate(evals_dict.items()):
        #evals_c = np.array([complex(val) for val in evals])
    
        marker_size = 4 + i * 4
    
        # z-order decreases with each block, ensuring smaller sizes (lower i) 
        # get a higher z-order and stay on top (Default zorder is around 1 or 2)
        current_zorder = 100 - i 
    
        plt.scatter(
            evals.real / j, 
            evals.imag / j, 
            s=marker_size, 
            zorder=current_zorder,  
            label=f'K = {key}'
        )

    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')

    plt.legend(loc='best', fontsize='small')

    plt.show()


# visualizes the block diagonalization of L into the two parity sectors
def visualize_parity_block_diagonalization(L, j, k, sigma=1e-3):
    n = int(round(2*j + 1))
    evals_dict = parity_block_low_lying_spectrum(L, n, k=k, sigma=sigma)   # {+1: evals, -1: evals}
    for i, (key, evals) in enumerate(evals_dict.items()):

        marker_size = 10 + i * 8

        # z-order decreases with each block, ensuring smaller sizes (lower i)
        # get a higher z-order and stay on top (Default zorder is around 1 or 2)
        current_zorder = 200 - i

        plt.scatter(
            evals.real,
            evals.imag,
            s=marker_size,
            zorder=current_zorder,
            label=f'P = {key:+d}'
        )

    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')

    plt.legend(loc='best', fontsize='small')

    plt.show()


# vizualization of the low-lying spectrum
def plot_low_lying(L, k, sigma=1e-3, compare_full=False):
    evals = low_lying_spectrum(L, k=k, sigma=sigma)
    if compare_full:
        full = L.eigenenergies()
        plt.scatter(full.real , full.imag , color='lightgray', s=4, label='full')
    plt.scatter(evals.real , evals.imag , color='red', s=8, label=f'{k} low-lying')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Low-lying spectrum of the Liouvillian')
    plt.legend()
    plt.show()

    # the Liouvillian gap: the largest nonzero real part
    print("Liouvillian gap:", -evals[1].real)
    return evals


# plots both the low-lying spectrum of the Liouvillian and the semiclassical spectrum obtained from the fixed points of the classical dynamics
# written by Claude
def plot_comparison0(evals_num, fixed_points, a, c, N_val, abs_threshold, j, C, p, save):
    all_L_evals = []  

    # common style for all spectrum plots to compare different specra
    def plot_spectrum(evals, scale=1.0, ax=None, color='blue', s=30, filled=True,
                  alpha=1.0, label=None, title='Spectrum of the Liouvillian'):
        if ax is None:
            ax = plt.gca()
        # open markers let you see points lying on top of each other
        style = dict(color=color) if filled else dict(facecolors='none', edgecolors=color)
        ax.scatter(evals.real / scale, evals.imag / scale, s=s, marker='o',
                   alpha=alpha, label=label, **style)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=0)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1, zorder=0)
        if scale == 1.0:
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
        else:
            ax.set_xlabel(r'Real Part / $j$')
            ax.set_ylabel(r'Imaginary Part / $j$')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='datalim')
        if label is not None:
            ax.legend()
        return ax

    plt.figure(figsize=(8, 8))

    plot_spectrum(evals_num, color='red', s=25, label=f'numerics, j = {j}',
                  title='Low-lying Liouvillian spectrum: numerics vs. semiclassics')

    fp_colors = ['blue', 'green', 'black', 'orange', 'olive', 'magenta', 'brown']

    all_L_evals = [] 

    last = [fixed_points[-1]]
    for i, point in enumerate(fixed_points):
        J_evals = Jacobian_eigenvalues(point, a=a, c=c)
    
        current_L_evals = liouvillian_eigenvalues(J_evals, N_val)
    
        all_L_evals.append(current_L_evals)
    
        current_L_evals_arr = np.array(current_L_evals)
    
        current_L_evals_filtered = current_L_evals_arr[np.abs(current_L_evals_arr) < abs_threshold]
    
        fp_color = fp_colors[i % len(fp_colors)]
    
        plot_spectrum(current_L_evals_filtered, color=fp_color, s=80, filled=False, label=f'semiclassics, FP {i+1}: {point.round(2)}', title=f'Liouvillian spectrum: C = {C:.2f}, p = {p}, a = {a}')

    all_L_evals = np.concatenate(all_L_evals)
    all_L_evals = all_L_evals[np.abs(all_L_evals) < abs_threshold] 
    if save == True:
        output_dir = "FPspec&low_spec"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"j{j}_C{C:.2f}_p{p}_a{a}_k{k}_N{N_val}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close() # close everytime
    plt.show()


def plot_comparison(evals_num, fixed_points, a, c, N_val, abs_threshold, j, C, p, k,
                    save=False, evecs_num=None,
                    coincident_threshold=1e-2, rank_tol=1e-2,
                    output_dir="low_lying_all_inclusive"):

    # common style for all spectrum plots to compare different spectra
    def plot_spectrum(evals, scale=1.0, ax=None, color='blue', s=30, filled=True,
                      alpha=1.0, label=None, title='Spectrum of the Liouvillian'):
        if ax is None:
            ax = plt.gca()
        evals = np.asarray(evals)
        # open markers let you see points lying on top of each other
        style = dict(color=color) if filled else dict(facecolors='none', edgecolors=color)
        ax.scatter(evals.real / scale, evals.imag / scale, s=s, marker='o',
                   alpha=alpha, label=label, **style)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=0)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1, zorder=0)
        if scale == 1.0:
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
        else:
            ax.set_xlabel(r'Real Part / $j$')
            ax.set_ylabel(r'Imaginary Part / $j$')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='datalim')
        if label is not None:
            ax.legend()
        return ax

    title = f'Liouvillian spectrum: j = {j}, C = {C:.2f}, p = {p}, a = {a}'
    plt.figure(figsize=(8, 8))

    # --- numerics -------------------------------------------------------
    if evecs_num is None:
        # no eigenvectors -> plain numerics as before
        plot_spectrum(evals_num, color='red', s=25, label=f'numerics, j = {j}', title=title)
    else:
        non_coinc, degen, coal, missing, _ = spectrum_coincidences(
            evals=evals_num, evecs=evecs_num,
            coincident_threshold=coincident_threshold, rank_tol=rank_tol)

        categories = [
            (non_coinc,             'black',   'non-coincidences'),
            (degen,                 'lime',   'degeneracies'),
            (coal[missing == 1],    'orange',    'coalescences'),
            (coal[missing > 1],     'pink', 'coalescences (higher order)'),
        ]
        for vals, col, lab in categories:
            if len(vals) > 0:   # avoid empty legend entries
                plot_spectrum(vals, color=col, s=25, label=lab, title=title)

    # --- semiclassics ---------------------------------------------------
    # colours chosen so they don't clash with the numerical categories
    fp_colors = ['brown', 'red', 'blue', 'purple', 'gray', 'olive', 'lime']

    all_L_evals = []
    for i, point in enumerate(fixed_points):
        J_evals = Jacobian_eigenvalues(point, a=a, c=c)
        L_evals = np.asarray(liouvillian_eigenvalues(J_evals, N_val))
        all_L_evals.append(L_evals)

        L_evals = L_evals[np.abs(L_evals) < abs_threshold]
        plot_spectrum(L_evals, color=fp_colors[i % len(fp_colors)], s=80, filled=False,
                      label=f'semiclassics, FP {i+1}: {np.round(point, 2)}', title=title)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"j{j}_C{C:.2f}_p{p}_a{a}_k{k}_N{N_val}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    else:
        plt.show()


'''function imported from 'classical_dynamics.py' to compute the eigenvalues of the Liouvillian from the eigenvalues of the Jacobian at a fixed point'''
from itertools import product
def liouvillian_eigenvalues(J_eigenvalues, n_max):
    lam = np.asarray(J_eigenvalues, dtype=complex)

    unstable = lam.real > 0
    # unstable directions get shifted by +1
    shift = unstable.astype(int)
    sign = np.where(unstable, -1.0, 1.0)

    s = []
    for m in product(range(n_max + 1), repeat=len(lam)):
        if sum(m) <= n_max:
            s.append(np.sum(sign * (np.array(m) + shift) * lam))
    return np.array(s)


def Jacobian_eigenvalues(point, a, c, h=1.0):

    def jacobian(point, a, c, h=1.0):
        j1, j2, j3 = point
        return np.array([
            [c*j3,               -h,                 c*j1],
            [h + 2*a*j3,          c*j3,              2*a*j1 + c*j2],
            [-2*a*j2 - 2*c*j1,   -2*a*j1 - 2*c*j2,   0.0]
        ])
    
    J = jacobian(point, a, c, h)

    def tangent_basis(point):
        n = np.asarray(point, dtype=float)
        n = n / np.linalg.norm(n)
        # The last two right-singular vectors of n^T span the plane orthogonal to n
        _, _, Vt = np.linalg.svd(n[None, :])
        return Vt[1:].T          # shape (3, 2), orthonormal columns
    
    P = tangent_basis(point)
    J_tan = P.T @ J @ P      # 2x2 Jacobian restricted to the sphere
    return np.linalg.eigvals(J_tan)

def find_fixed_points(a, c, h = 1, tol=1e-12):
    pts = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]           # always fixed points

    if abs(h) > tol:
        # j3 solves c^2 j3^2 + 2 a h j3 + h^2 = 0 ; then j2 = k j1, k = c j3 / h
        if abs(c) > tol:
            D = a*a - c*c
            j3s = set() if D < -tol else {h*(-a + s)/c**2 for s in (np.sqrt(max(D, 0.0)),
                                                                         -np.sqrt(max(D, 0.0)))}
        elif abs(a) > tol:
            j3s = {-h / (2*a)}
        else:
            j3s = set()
        for j3 in j3s:
            if abs(j3) <= 1 + tol:
                k = c * j3 / h
                j1 = np.sqrt(max(1 - j3*j3, 0.0) / (1 + k*k))
                pts += [(j1, k*j1, j3), (-j1, -k*j1, j3)]
    else:
        # h = 0: j3 = 0 and a*sin(2*phi) = -c on the unit circle
        if abs(a) > tol and abs(c) <= abs(a) + tol:
            t = np.arcsin(np.clip(-c / a, -1, 1))
            for phi in {t/2, (np.pi - t)/2}:
                for s in (1, -1):
                    pts.append((s*np.cos(phi), s*np.sin(phi), 0.0))

    return np.unique(np.round(np.array(pts), 14), axis=0)   # shape (n, 3), real points only

def classical_spectrum(fixed_points, a, c, N_val):
    return np.concatenate([
        liouvillian_eigenvalues(Jacobian_eigenvalues(pt, a=a, c=c), N_val)
        for pt in fixed_points
    ])
''''''


j = 100

a = 0.3
c = 0.5
p = 0.99

C = c / p


k = 50 # number of low-lying eigenvalues to compute from the Liouvillian

N_val = 10 # corresponds to a number of eigenvalues to compute from the semiclassical spectrum
abs_threshold = 10 # threshold for the absolute value of eigenvalues to keep in the semiclassical spectrum

fixed_points = find_fixed_points(a=a, c=c)
#fixed_points = np.delete(fixed_points, 1)

AS = [a] #np.linspace(0.50425, 0.50429, 15)  # range of a values to loop over

for A in AS:
    fixed_points = find_fixed_points(a=A, c=c)   # fixed points change with a
    L = Liouvillian(s=j, h=1.0, C=C, p=p, C0=0.0, a=A)
    evals_num, evecs_num = low_lying_spectrum(L, k=k, eigenvectors=True)

    plot_comparison(evals_num, fixed_points, A, c, N_val, abs_threshold, j, C, p, k,
                    evecs_num=evecs_num, save=False,
                    output_dir="../Plots/low_lying_all_inclusive")
    visualize_parity_block_diagonalization(L, j, k, sigma=1e-3)


'''clas_evals = classical_spectrum(fixed_points, a, c, N_val)

js = [] #[20, 40, 60, 80, 100, 150, 200]
difs = []
for j in js:
    L = Liouvillian(s=j, h=1.0, C=C, p=p, C0=0.0, a=a)
    q_evals = low_lying_spectrum(L, k=k)
    dif, q_gap, cl_match = eig_dif(q_evals, clas_evals)
    difs.append(dif)
    print(j, q_gap, cl_match, dif)

slope, intercept = np.polyfit(np.log(js), np.log(difs), 1)
print(f"slope = {slope:.3f}")   # dif ~ j**slope

js_arr = np.array(js)
plt.loglog(js, difs, 'o', label='data')
plt.loglog(js_arr, np.exp(intercept) * js_arr**slope, '--',
           label=f'fit: slope = {slope:.2f}')
plt.legend()
plt.xlabel('$j$')
plt.ylabel(r'$|\Lambda - \Lambda^{\rm cl}|$')
#plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()'''


'''for a in AS:
    L = Liouvillian(s = j, h = 1.0, C = C, p = p, C0 = 0.0, a = a)
    time_start = time.time()
    evals_num, evecs_num = low_lying_spectrum(L, k=k, eigenvectors=True)
#print("Low-lying eigenvalues of the Liouvillian:", evals_num)
#print("Low-lying eigenvectors of the Liouvillian:", evecs_num)
    time_end = time.time()
    #print(f"Time taken to compute the low-lying spectrum: {time_end - time_start} seconds")

    time_start = time.time()
    non_coincidences, degeneracies, coalescences, missing, singular_values = spectrum_coincidences(evals=evals_num, evecs=evecs_num, coincident_threshold=1e-2, rank_tol=1e-2)
    time_end = time.time()
    #print(f"Time taken to separate the spectrum: {time_end - time_start} seconds")
    #print(singular_values)

    plt.scatter(non_coincidences.real, non_coincidences.imag, c='blue', label='non-coincidences')
    plt.scatter(degeneracies.real, degeneracies.imag, c='lime', label='degeneracies')
    plt.scatter(coalescences[missing == 1].real, coalescences[missing == 1].imag, c='red', label='coalescences')
    plt.scatter(coalescences[missing > 1].real, coalescences[missing > 1].imag, c='orange', label='coalescences (higher order)')
    plt.title(f'Low-lying spectrum: j = {j}, C = {C:.2f}, p = {p}, a = {a}')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    output_dir = "../Plots/low-lying_coincidences"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"j{j}_C{C:.2f}_p{p}_a{a}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close() # close everytime
    plt.show()'''

#L = Liouvillian(s = j, h = 1.0, C = C, p = p, C0 = 0.0, a = a)
#visualize_parity_block_diagonalization(L, j, k, sigma=1e-3)
#evals, evecs = L.eigenstates()
#spectrum_coincidences_plotting(evals, evecs, coincident_threshold=0.05, rank_tol=1e-2)

# NOTE: ansatz L(a) = L0 + a * L1 where L1 corresponds to interaction term in Hamiltonian commutator -1/j*[J_x^2, ...]
# This allows us to constuct L0, L1 once as scipy sparce matrices and speed up the calculation