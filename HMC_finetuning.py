import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm
import argparse
import time
import json

eps_taus = [(round(eps,2), round(1/eps)) for eps in np.linspace(0.12,0.01,12)]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

        
""" Code from the lectures"""
def potential_v(x,lamb):
    '''Compute the potential function V(x).'''
    return lamb*(x*x-1)*(x*x-1)+x*x


def scalar_action(phi,lamb, kappa):
    A = potential_v(phi, lamb) - 2 * kappa * (np.roll(phi, 1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, 1, axis = 2) + np.roll(phi, 1, axis = 3)) * phi
    return np.sum(A)

def scalar_ham_diff(phi,pi,newphi,newpi,lamb,kappa):
    ham_difference = 0.5* np.sum(newpi**2 - pi**2)
    action_diff = scalar_action(newphi,lamb,kappa) - scalar_action(phi, lamb,kappa)

    return ham_difference + action_diff          

def force(phi,kappa,lamb):
    width = phi.shape[0]
    F = np.zeros((width,width,width,width))
    F += 2*phi + 4*lamb*(phi**3) - 4*lamb*phi - 2 * kappa * (np.roll(phi, 1, axis = 0) + np.roll(phi, -1, axis = 0) + np.roll(phi, 1, axis = 1) + np.roll(phi, -1, axis = 1) 
         + np.roll(phi, 1, axis = 2) + np.roll(phi, -1, axis = 2) + np.roll(phi, 1, axis = 3) + np.roll(phi, -1, axis = 3))
    return F
      

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
    width = phi.shape[0]
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
    if n > 999:
        for i in tqdm(range(n)):
            catch = 0
            catch, phi = scalar_HMC_step(phi,lamb,kappa,eps,tau)
            total_accept += catch
    else:
        for i in range(n):
            catch = 0
            catch, phi = scalar_HMC_step(phi,lamb,kappa,eps,tau)
            total_accept += catch

    return total_accept, phi





# use the argparse package to parse command line arguments
parser = argparse.ArgumentParser(description= 'Measures the average field value of the scalar field.')
parser.add_argument('-iw',type= int, help='lowest lattice size tested')
parser.add_argument('-hw',type= int, help='Highest lattice size tested')
parser.add_argument('-ki',type=float, help='Starting value for Kappa')
parser.add_argument('-kf',type=float, help='End value for Kappa')
parser.add_argument('-ka',type=int, default = 11, help='Amount of Kappas')
parser.add_argument('-l', type=float, default = 1.5, help ='lambda')
parser.add_argument('-s', type = int, default = 300, help = "number of sweeps to finetune")
parser.add_argument('-e', type=int, default=1000, help='Number E of equilibration sweeps')
parser.add_argument('-m', type=int, default=800, help='Number M of sweeps per measurement')
parser.add_argument('-o', type=int, default=30, help='Time in seconds between file outputs')
parser.add_argument('-ep', type=float, default = 0.15, help = 'Stepsize of leapfrog integrator')
parser.add_argument('-tau',type=int,default=10,help='discretization steps of leapfrog integrator')

args = parser.parse_args()
#Some Sanity checks
if args.hw is None or args.hw < 1:
    parser.error("Please specify a positive lattice size!")
if args.l is None or args.l <= 0.0:
    parser.error("Please specify a positive lambda!")
if args.e < 10:
    parser.error("Need at least 10 equilibration sweeps")
if args.tau*args.ep >=1.5 or args.tau*args.ep <=0.5:
    print("The product of epsilon and tau is not close to the desired value of 1.")

kappas = np.linspace(args.ki,args.kf,args.ka)
lamb = args.l
equil_sweeps = args.e
measurements = args.m


def low_k_checker(width,lamb,kappa_low,eps,tau,sweeps):
    phi_state_low = np.zeros((width,width,width,width))
    acceptions, phi_state_low = run_scalar_MH(phi_state_low,lamb,kappa_low,eps,tau,sweeps)
    acc_rate = acceptions/sweeps
    if acc_rate > 0.15 and acc_rate < 0.9:
        return True
    else:
        return False


def main():
    fine_tuned_values = []
    for width in range(args.iw,args.hw+1):
        print("\n Finetuning process for width: ", width)
        for index, (eps,tau) in enumerate(eps_taus):
            print('\n', index, " Finetuning step")
            phi_state_highk = np.zeros((width,width,width,width))
            kappa_high = args.kf
            sweeps = args.s
            acceptions, phi_state_highk = run_scalar_MH(phi_state_highk,lamb,kappa_high,eps,tau,sweeps)
            acc_rate = acceptions/sweeps
            if acc_rate > 0.15:
                if acc_rate < 0.9:
                    if low_k_checker(width,lamb, args.ki, eps, tau, sweeps):
                        fine_tuned_values.append((width,eps,tau))
                        print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width)
                        break
                    else: 
                        if index != 0:
                            eps, tau = eps_taus[index -1]
                            fine_tuned_values.append((width,eps,tau))
                            print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width)
                            break
            elif index == len(eps_taus) - 1:
                fine_tuned_values.append((width,eps,tau))
                print("lowest value eps was not sufficient, but is still added")
                print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width)
        

        
    print(fine_tuned_values)




if __name__ == '__main__':
    main()

