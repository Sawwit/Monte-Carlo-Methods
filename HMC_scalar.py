import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm

width = 3
lamb = 1.5
kappas = np.linspace(0.08,0.18,11)
num_sites= width**4
equil_sweeps = 1000
measure_sweeps = 1
measurements = 800
eps = 0.01
tau = 100


""" Code from the lectures"""
def potential_v(x,lamb):
    '''Compute the potential function V(x).'''
    return lamb*(x*x-1)*(x*x-1)+x*x

# def neighbor_sum(phi,s):
#     '''Compute the sum of the state phi on all 8 neighbors of the site s.'''
#     w = len(phi)

#     return (phi[(s[0]+1)%w,s[1],s[2],s[3]] + phi[(s[0]-1)%w,s[1],s[2],s[3]] +
#             phi[s[1],(s[1]+1)%w,s[2],s[3]] + phi[s[1],(s[1]-1)%w,s[2],s[3]] +
#             phi[s[0],s[1],(s[2]+1)%w,s[3]] + phi[s[0],s[1],(s[2]-1)%w,s[3]] +
#             phi[s[0],s[1],s[2],(s[3]+1)%w] + phi[s[0],s[1],s[2],(s[3]-1)%w] )

# def scalar_action_diff_site(phi,pi,site,newphi,newpi,lamb,kappa):
#     '''Compute the change in the hamiltonian when phi is changed to newphi.'''
#     return (2 * kappa * neighbor_sum(phi,site) * (phi[site[0],site[1],site[2],site[3]] - newphi[site[0],site[1],site[2],site[3]]) +
#             potential_v(newphi[site[0],site[1],site[2],site[3]],lamb) - potential_v(phi[site[0],site[1],site[2],site[3]],lamb) )

# def scalar_acttion_diff(phi,pi,newphi,newpi,lamb,kappa):
#     difference=0
#     for i in range(width):
#         for j in range(width):
#             for k in range(width):
#                 for l in range(width):
#                     difference +=scalar_action_diff_site(phi,pi,[i,j,k,l],newphi,newpi,lamb,kappa)
    
#     return difference  

# def scalar_acttion_diff(phi,pi,newphi,newpi,lamb,kappa):
#     A = (2 * kappa * neighbor_sum(phi,site) * (phi[site[0],site[1],site[2],site[3]] - newphi[site[0],site[1],site[2],site[3]]) +
#             potential_v(newphi[site[0],site[1],site[2],site[3]],lamb) - potential_v(phi[site[0],site[1],site[2],site[3]],lamb) )
#     return 
def scalar_action(phi,lamb, kappa):
    A = potential_v(phi, lamb) - 2 * kappa * (np.roll(phi, 1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, 1, axis = 2) + np.roll(phi, 1, axis = 3)) * phi
    return np.sum(A)

def scalar_ham_diff(phi,pi,newphi,newpi,lamb,kappa):
    ham_difference = 0.5* np.sum(newpi**2 - pi**2)
    action_diff = scalar_action(newphi,lamb,kappa) - scalar_action(phi, lamb,kappa)

    return ham_difference + action_diff          

def force(phi,kappa,lamb):
    F = np.zeros((width,width,width,width))
    F += 2*phi + 4*lamb*(phi**3) - 4*lamb*phi - 2 * kappa * (np.roll(phi, 1, axis = 0) + np.roll(phi, -1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, -1, axis = 1) 
         + np.roll(phi, 1, axis = 2) + np.roll(phi, -1, axis = 2) + np.roll(phi, 1, axis = 3) + np.roll(phi, -1, axis = 3))
    return F
    


    # F= np.zeros((width,width,width,width))
    # for i in range(width):
    #     for j in range(width):
    #         for k in range(width):
    #             for l in range(width):
    #                 F[i,j,k,l]= 2*phi[i,j,k,l] + 4*lamb*(phi[i,j,k,l]**2-1)*phi[i,j,k,l]-2*kappa*neighbor_sum(phi,[i,j,k,l])

    # return F                

def I_pi(phi,pi,lamb,kappa,eps):
    return pi - eps*force(phi,kappa,lamb)

def I_phi(phi,pi,eps):
    return phi + eps*pi

def leapfrog(phi,pi,eps,tau,kappa):
    a=phi
    b=pi

    for _ in range(tau):
        pi=I_pi(phi,pi,lamb,kappa,eps/2)
        phi=I_phi(phi,pi,eps)
        pi=I_pi(phi,pi,lamb,kappa,eps/2)

    
    return phi,pi,a,b

def scalar_HMC_step(phi,lamb,kappa,eps,tau):
    #Sample pi as random (normally distributed) noise for every lattice site
    pi = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width))  
    phi_new, pi_new, phi_old, pi_old = leapfrog(phi,pi,eps,tau,kappa)
    delta_H = scalar_ham_diff(phi_old,pi_old,phi_new,pi_new,lamb,kappa)
    if delta_H <0 or rng.uniform() <np.exp(-delta_H):
        return 1, phi_new
    else:
        return 0, phi_old


def run_scalar_MH(phi,lamb,kappa,eps,tau,n):
    '''Perform n Metropolis-Hastings updates on state phi and return number of accepted transtions.'''
    total_accept = 0
    for i in range(n):
        catch = 0
        catch, phi = scalar_HMC_step(phi,lamb,kappa,eps,tau)
        total_accept += catch

    return total_accept, phi

def batch_estimate(data,observable,num_batches):
    '''Determine estimate of observable on the data and its error using batching.'''
    batch_size = len(data)//num_batches
    values = [observable(data[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]

    return np.mean(values), np.std(values)/np.sqrt(num_batches-1)






def main():
    mean_magn = []
    for kappa in tqdm(kappas):
        phi_state = np.zeros((width,width,width,width))
        phi_old = np.copy(phi_state)
        acceptions, phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,equil_sweeps)
        print(acceptions/equil_sweeps)
        magnetizations = np.empty(measurements)
        acceptions = 0
        for i in range(measurements):
            accept, phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,measure_sweeps)
            acceptions += accept
            magnetizations[i] = np.mean(phi_state)
        print(acceptions/measurements)
        mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
        mean_magn.append([mean,err])
        print("kappa = {:.2f}, |m| = {:.3f} +- {:.3f}".format(kappa,mean,err))
        
    plt.errorbar(kappas,[m[0] for m in mean_magn],yerr=[m[1] for m in mean_magn],fmt='-o')
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$|m|$")
    plt.title(r"Absolute field average on $3^4$ lattice, $\lambda = 1.5$")
    plt.show()

if __name__=="__main__":
    main()
