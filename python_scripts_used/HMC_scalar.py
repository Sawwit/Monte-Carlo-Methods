import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm
import argparse
import time
import json


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

# width = 4
# lamb = 1.5
# kappas = np.linspace(0.08,0.18,11)
# num_sites= width**4
# equil_sweeps = 1000
# measure_sweeps = 1
# measurements = 800
# eps = 0.0375
# tau = 40
autocovruns = 300

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

def leapfrog(phi,pi,eps,tau,kappa, lamb):
    a=phi
    b=pi

    for _ in range(tau):
        pi=I_pi(phi,pi,lamb,kappa,eps/2)
        phi=I_phi(phi,pi,eps)
        pi=I_pi(phi,pi,lamb,kappa,eps/2)

    
    return phi,pi,a,b

def scalar_HMC_step(phi,lamb,kappa,eps,tau):
    #Sample pi as random (normally distributed) noise for every lattice site
    width  = phi.shape[0]
    pi = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width))  
    phi_new, pi_new, phi_old, pi_old = leapfrog(phi,pi,eps,tau,kappa, lamb)
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

def batch_estimate(data,observable,num_batches):
    '''Determine estimate of observable on the data and its error using batching.'''
    batch_size = len(data)//num_batches
    values = [observable(data[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]

    return np.mean(values), np.std(values)/np.sqrt(num_batches-1)


def sample_autocovariance(x,tmax):
    '''Compute the autocorrelation of the time series x for t = 0,1,...,tmax-1.'''
    x_shifted = x - np.mean(x)
    return np.array([np.dot(x_shifted[:len(x)-t],x_shifted[t:])/len(x) for t in range(tmax)])

def find_correlation_time(x,tmax):
    '''Return the index of the first entry that is smaller than autocov[0]/e.'''
    autocov = sample_autocovariance(x,tmax)
    return np.where(autocov < np.exp(-1)*autocov[0])[0][0]

# use the argparse package to parse command line arguments


# args = parser.parse_args()
# #Some Sanity checks
# if args.w is None or args.w < 1:
#     parser.error("Please specify a positive lattice size!")
# if args.l is None or args.l <= 0.0:
#     parser.error("Please specify a positive lambda!")
# if args.e < 10:
#     parser.error("Need at least 10 equilibration sweeps")
# if args.tau*args.ep >=1.5 or args.tau*args.ep <=0.5:
#     print("The product of epsilon and tau is not close to the desired value of 1.")
# width = args.w
# kappas = np.linspace(args.ki,args.kf,args.ka)
# lamb = args.l
# equil_sweeps = args.e
# measurements = args.m
# eps = args.ep
# tau = args.tau

# fix parameters
# if args.f is None:
#     # construct a filename from the parameters plus a timestamp (to avoid overwriting)
#     output_filename = "data_w{}_l{}_{}.json".format(width,lamb,time.strftime("%Y%m%d%H%M%S"))
# else:
#     output_filename = args.f

def equil_time_estimator(width, kappa, lamb, eps, tau, epsilon):
    phi_state_zeros = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width)) 
    phi_state_ones = np.random.multivariate_normal(np.array([1 for _ in range(width)]),np.eye(width),(width,width,width)) 
    magnetizations_ones = []
    magnetizations_zeros = []
    measure_counter = 0


    while True:
        _, phi_state_zeros = run_scalar_MH(phi_state_zeros,lamb,kappa,eps,tau,1)
        _, phi_state_ones = run_scalar_MH(phi_state_ones,lamb,kappa,eps,tau,1)
        magnetizations_ones.append(np.mean(phi_state_ones)) 
        magnetizations_zeros.append(np.mean(phi_state_zeros))
        measure_counter += 1
        
        if np.abs(np.abs(magnetizations_ones[-1]) - np.abs(magnetizations_zeros[-1])) <= epsilon :
            break

    return measure_counter, phi_state_zeros



def HMC_scalar(width, kappa_init, kappa_final, kappa_amount, lamb, eps, tau, measurements, output_time, output_filename, args):
    kappas = np.linspace(kappa_init, kappa_final, kappa_amount)
    mean_magn = np.empty((len(kappas),6))
    equil_needed, phi_state = equil_time_estimator(width, kappa_init, lamb, eps, tau, 0.01)
    print("determined sweeps needed for equil phase: ", equil_needed, "\n")
    # phi_state = np.zeros((width,width,width,width))
    acceptions, phi_state = run_scalar_MH(phi_state,lamb,kappa_init,eps,tau,equil_needed)
    rate = acceptions/equil_needed
    if rate < 0.4:
        eps = eps - 0.005
        tau = int(round(1/eps))
        print("parameters have been adjusted")
    print("Acceptance rate for equilibration phase:", rate)
    for index, kappa in tqdm(enumerate(kappas)):
        start_time=time.time()        
        #Aantal sweeps runnen met de nieuwe kappa
        if kappa != kappa_init:
          _ , phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,50)
        #correlation time bepalen
        pre_magn = np.empty(autocovruns)
        for i in range(autocovruns):
            accept, phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,1)
            pre_magn[i] = np.mean(phi_state)
        cor_time = find_correlation_time(pre_magn,len(pre_magn))
        print("Estimated correlation time: ",cor_time, " using ", autocovruns, " runs to determine as such")
        cor_sweeps = int((1+cor_time)/2)
        magnetizations = []
        acceptions = 0
        measure_counter = 0
        last_output_time = time.time()
        while True:
            accept, phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,cor_sweeps)
            magnetizations.append(np.mean(phi_state))
            measure_counter += 1
            acceptions += accept
            

            if measure_counter == measurements or time.time() - last_output_time > output_time:
                mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
                current_time = time.time()
                acc_rate = acceptions/(cor_sweeps * measure_counter)
                run_time = int(current_time - start_time)
                mean_magn[index] = np.array([kappa,acc_rate,mean,err,cor_time,run_time])
                with open(output_filename,'w') as outfile:
                    json.dump({ 
                        'parameters': vars(args),
                        'start_time': time.asctime(time.localtime(start_time)),
                        'current_time': time.asctime(time.localtime(current_time)),
                        'run_time_in_seconds': run_time,
                        'measurements': measure_counter,
                        'moves_per_measurement': cor_sweeps,
                        'kappa_latest': mean_magn[index][0],
                        'mean_latest': mean_magn[index][1],
                        'err_latest': mean_magn[index][2],
                        'correlation_time_latest': mean_magn[index][3],
                        'full_results': mean_magn
                        }, outfile, cls=NumpyEncoder)
                if measure_counter == measurements:
                    break
                else:
                    last_output_time = time.time()
    
        
    # plt.errorbar(kappas,[m[1] for m in mean_magn],yerr=[m[2] for m in mean_magn],fmt='-o')
    # plt.xlabel(r"$\kappa$")
    # plt.ylabel(r"$|m|$")
    # plt.title(r"Absolute field average on $4^4$ lattice, $\lambda = 1.5$")
    # plt.show()

    return phi_state
