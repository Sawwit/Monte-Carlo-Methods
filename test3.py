import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm
np.random.seed(0)

width = 3
phi = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width))  
newphi = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width))  
lamb = 1.5
kappa = 0.1

def potential_v(x,lamb):
    '''Compute the potential function V(x).'''
    return lamb*(x*x-1)*(x*x-1)+x*x

def scalar_action(phi,lamb, kappa):
    A = potential_v(phi, lamb) - 2 * kappa * (np.roll(phi, 1, axis = 0) + np.roll(phi, -1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, -1, axis = 1) 
         + np.roll(phi, 1, axis = 2) + np.roll(phi, -1, axis = 2) + np.roll(phi, 1, axis = 3) + np.roll(phi, -1, axis = 3)) * phi
    return np.sum(A)
print(scalar_action(newphi,lamb,kappa) - scalar_action(phi,lamb,kappa))

print(np.sum(np.roll(newphi, 1, axis = 0) + np.roll(newphi, -1, axis = 0) + np.roll(newphi, 1, axis = 1) + np.roll(newphi, -1, axis = 1) 
         + np.roll(newphi, 1, axis = 2) + np.roll(newphi, -1, axis = 2) + np.roll(newphi, 1, axis = 3) + np.roll(newphi, -1, axis = 3)))
