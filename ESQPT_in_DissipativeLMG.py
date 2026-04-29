from qutip import *
import numpy as np
import matplotlib.pyplot as plt

j = 300
N = 2*j + 1

def H(h = 1.0, a = 0.1):
    return -h * jmat(j, 'z') - a / N * jmat(j, 'x')**2

def H_eff(k, a = 2):
    D = 1j * k / (2*N) * jmat(j, '-') * jmat(j, '+')
    return H(a=a) - D

evals = H_eff(0.05).eigenenergies()

plt.scatter(evals.real / N, -evals.imag / N, color='black', s=4)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of the effective Hamiltonian')
plt.show()

def dens_states(H, E_array):
    evals = H.eigenenergies() / N
    Tr = np.sum([1 / (E_array - eval) for eval in evals], axis = 0)
    return -1 / np.pi * np.imag(Tr)

E = np.linspace(-0.6, 0.4, 1000)
#densities = [dens_states(e) for e in E]
plt.plot(E, dens_states(H_eff(0.05), E), color='black')
plt.xlabel('Energy')
plt.ylabel('Density of States')
plt.title('Density of States of the effective Hamiltonian')
plt.show()