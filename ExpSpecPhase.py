from qutip import *
import numpy as np
import matplotlib.pyplot as plt

'''https://doi.org/10.1103/PhysRevA.106.L010201'''

## Define parameters
j = 10 # total spin
N = 2*j + 1 # number of particles considering j = j_MAX

p = 0.99

def L(j, p, h = 1.0, C = 1.0, C0 = 0.0):
    K1z = spre(jmat(j, 'z'))
    K1p = spre(jmat(j, '+'))
    K1m = spre(jmat(j, '-'))

    K2z = spost(jmat(j, 'z').dag())
    K2p = spost(jmat(j, '+').dag())
    K2m = spost(jmat(j, '-').dag())

    Kzm = K1z - K2z
    Kzp = K1z + K2z
    Kz = K1z * K2z
    Kp = K1p * K2p
    Km = K1m * K2m
    return - C * (j + 1) + 1j * h * Kzm + C / j * Kz + (C - C0) * 0.5 / j * Kzm**2 - C * 0.5 * p / j * Kzp + C * (1 - p) * 0.5 / j * Kp + C * (1 + p) * 0.5 / j * Km

## Compute and visualize the spectrum of the Liouvillian
evals = L(j,p).eigenenergies()

'''plt.scatter(evals.real / j, evals.imag / j, color='black', s=4)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of the Liouvillian')
plt.show()'''

## Separating spectrum into exceptional and normal points to see the Liouvillian spectral phase transition (LSPT)
def spec_separation(evals, threshold = 1e-1): # the threshold should decrease with the system size
    expvals = np.array([])
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
            expvals = np.append(expvals, a)
    
    return expvals, normvals

expvals, normvals = spec_separation(evals)

#print(expvals)
#print(normvals)

plt.scatter(expvals.real / j, expvals.imag / j, color='red', s=4)
plt.scatter(normvals.real / j, normvals.imag / j, color='blue', s=4)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of the Liouvillian with the LSPT')
plt.show()

            