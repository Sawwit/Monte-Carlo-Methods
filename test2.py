import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm
np.random.seed(0)

width = 3
phi= np.random.multivariate_normal([0 for _ in range(width)],np.eye(width), size= (width,width,width)) *300
new_phi = np.random.multivariate_normal([0 for _ in range(width)],np.eye(width), size= (width,width,width)) *300
lamb = 1.5
kappa = 0.1

def potential_v(x,lamb):
    '''Compute the potential function V(x).'''
    return lamb*(x*x-1)*(x*x-1)+x*x

def neighbor_sum(phi,s):
    '''Compute the sum of the state phi on all 8 neighbors of the site s.'''
    w = len(phi)

    return (phi[(s[0]+1)%w,s[1],s[2],s[3]] + phi[(s[0]-1)%w,s[1],s[2],s[3]] +
            phi[s[0],(s[1]+1)%w,s[2],s[3]] + phi[s[0],(s[1]-1)%w,s[2],s[3]] +
            phi[s[0],s[1],(s[2]+1)%w,s[3]] + phi[s[0],s[1],(s[2]-1)%w,s[3]] +
            phi[s[0],s[1],s[2],(s[3]+1)%w] + phi[s[0],s[1],s[2],(s[3]-1)%w] )

def scalar_action_diff_site(phi,site,newphi,lamb,kappa):
    '''Compute the change in the hamiltonian when phi is changed to newphi.'''
    return (2 * kappa * neighbor_sum(phi,site) * (phi[site[0],site[1],site[2],site[3]] - newphi[site[0],site[1],site[2],site[3]]) +
            potential_v(newphi[site[0],site[1],site[2],site[3]],lamb) - potential_v(phi[site[0],site[1],site[2],site[3]],lamb) )

def scalar_acttion_diff(phi,newphi,lamb,kappa):
    difference=0
    for i in range(width):
        for j in range(width):
            for k in range(width):
                for l in range(width):
                    difference +=scalar_action_diff_site(phi,[i,j,k,l],newphi,lamb,kappa)
    
    return difference  

def scalar_ham_diff(phi,pi,newphi,newpi,lamb,kappa):
    ham_difference = 0.5* np.sum(newpi**2 - pi**2)

    return ham_difference+scalar_acttion_diff(phi,pi,newphi,newpi,lamb,kappa)            

def force(phi,kappa,lamb):
    F= np.zeros((width,width,width,width))
    for i in range(width):
        for j in range(width):
            for k in range(width):
                for l in range(width):
                    F[i,j,k,l]= 2*phi[i,j,k,l] + 4*lamb*(phi[i,j,k,l]**2-1)*phi[i,j,k,l]-2*kappa*neighbor_sum(phi,[i,j,k,l])

    return F                

print("scalar differnence \n", scalar_acttion_diff(phi,new_phi,lamb,kappa))
NS = 0
for i in range(width):
    for j in range(width):
        for k in range(width):
            for l in range(width):
                NS+= neighbor_sum(phi,[i,j,k,l])
print("phi \n", NS)
pot = 0
for i in range(width):
    for j in range(width):
        for k in range(width):
            for l in range(width):
                pot += potential_v(phi[i,j,k,l], lamb)
print("phi, potential \n", pot)
NS = 0
for i in range(width):
    for j in range(width):
        for k in range(width):
            for l in range(width):
                NS+= neighbor_sum(new_phi,[i,j,k,l])
print("newphi \n", NS)

pot = 0
for i in range(width):
    for j in range(width):
        for k in range(width):
            for l in range(width):
                pot += potential_v(new_phi[i,j,k,l], lamb)
print("newphi, potential \n", pot)