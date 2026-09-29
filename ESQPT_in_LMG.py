from qutip import *
import numpy as np
import matplotlib.pyplot as plt

j = 10

Jz = jmat(j, 'z')
Jx = jmat(j, 'x')

def H(a, b = 1, chi = 0):
    return  b * Jz - a/j * (Jx + chi * (Jz + j))**2

def h(a, b = 1, chi = 0):
    return H(a, b, chi)/j

def oper_func(A, f):
    evals, eigvecs = A.eigenstates()
    fA = sum([f(evals[i]) * eigvecs[i] * eigvecs[i].dag() for i in range(len(evals))])
    return fA

def P(s=j):
    return (1j * np.pi * (jmat(s, 'z') + j)).expm()

import numpy as np

def find_degenerate_groups(evals, tol=1e-8):
    """Group indices of evals that are equal within tol."""
    order = np.argsort(evals)
    evals_sorted = evals[order]
    
    groups = []
    current_group = [order[0]]
    for i in range(1, len(evals_sorted)):
        if evals_sorted[i] - evals_sorted[i-1] < tol:
            current_group.append(order[i])
        else:
            groups.append(current_group)
            current_group = [order[i]]
    groups.append(current_group)
    return groups

def parity_eigenvalue(P, evec, tol=1e-6):
    exp_val = expect(P, evec)
    if abs(exp_val - 1) < tol:
        return +1
    elif abs(exp_val + 1) < tol:
        return -1
    else:
        return None  # not a clean parity eigenstate — see caveat below

def parity_adapted_basis(P, evecs, group):
    # Stack the degenerate eigenvectors as columns
    V = np.hstack([evecs[i].full() for i in group])  # shape (dim, k)
    
    # Restrict P to this subspace: P_sub = V^dagger P V  (k x k matrix)
    P_mat = P().full()
    P_sub = V.conj().T @ P_mat @ V
    
    # Diagonalize the small k x k Hermitian matrix
    p_evals, p_evecs = np.linalg.eigh(P_sub)
    
    # New parity-adapted eigenvectors, as combinations of the original degenerate set
    new_vecs = V @ p_evecs  # shape (dim, k)
    return p_evals, new_vecs

evals, evecs = h(a=3.0).eigenstates()

tol_degen = 1e-5
groups = find_degenerate_groups(evals, tol=tol_degen)

doublets = []
for group in groups:
    if len(group) < 2:
        continue  # non-degenerate, skip
    p_evals, new_vecs = parity_adapted_basis(P, evecs, group)
    signs = np.sign(np.round(p_evals))
    if +1 in signs and -1 in signs:
        doublets.append({
            'energy': evals[group[0]],
            'group_size': len(group),
            'parity_eigenvalues': p_evals,
            'vectors': new_vecs
        })

for d in doublets:
    print(f"E = {d['energy']:.6f}, size = {d['group_size']}, parities = {np.round(d['parity_eigenvalues'],3)}")

evals = []
for a in np.linspace(0, 3, 100):
    evals.append(h(a).eigenenergies())

plt.plot(np.linspace(0, 3, 100), evals)
plt.xlabel('Interaction strength')
plt.ylabel('Energy')
plt.show()