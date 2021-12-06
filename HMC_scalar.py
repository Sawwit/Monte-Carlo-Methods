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
# eps = 0.15
# tau = 10
autocovruns = 150

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


def sample_autocovariance(x,tmax):
    '''Compute the autocorrelation of the time series x for t = 0,1,...,tmax-1.'''
    x_shifted = x - np.mean(x)
    return np.array([np.dot(x_shifted[:len(x)-t],x_shifted[t:])/len(x) for t in range(tmax)])

def find_correlation_time(x,tmax):
    '''Return the index of the first entry that is smaller than autocov[0]/e.'''
    autocov = sample_autocovariance(x,tmax)
    return np.where(autocov < np.exp(-1)*autocov[0])[0][0]

# use the argparse package to parse command line arguments
parser = argparse.ArgumentParser(description= 'Measures the average field value of the scalar field.')
parser.add_argument('-w',type= int, help='Lattice size W')
parser.add_argument('-ki',type=float, help='Starting value for Kappa')
parser.add_argument('-kf',type=float, help='End value for Kappa')
parser.add_argument('-ka',type=int, help='Amount of Kappas')
parser.add_argument('-l', type=float, help ='lambda')
parser.add_argument('-e', type=int, default=1000, help='Number E of equilibration sweeps')
parser.add_argument('-m', type=int, default=800, help='Number M of sweeps per measurement')
parser.add_argument('-o', type=int, default=30, help='Time in seconds between file outputs')
parser.add_argument('-ep', type=float, default = 0.15, help = 'Stepsize of leapfrog integrator')
parser.add_argument('-tau',type=float,default=10,help='discretization steps of leapfrog integrator')
parser.add_argument('-f', help='Output filename')

args = parser.parse_args()
#Some Sanity checks
if args.w is None or args.w < 1:
    parser.error("Please specify a positive lattice size!")
if args.l is None or args.l <= 0.0:
    parser.error("Please specify a positive lambda!")
if args.e < 10:
    parser.error("Need at least 10 equilibration sweeps")
if args.tau*args.ep >=1.5 or args.tau*args.ep <=0.5:
    print("The product of epsilon and tau is not close to the desired value of 1.")
width = args.w
kappas = np.linspace(args.ki,args.kf,args.ka)
lamb = args.l
equil_sweeps = args.e
measurements = args.m
eps = args.ep
tau = args.tau

# fix parameters
if args.f is None:
    # construct a filename from the parameters plus a timestamp (to avoid overwriting)
    output_filename = "data_w{}_l{}_{}.json".format(width,lamb,time.strftime("%Y%m%d%H%M%S"))
else:
    output_filename = args.f




def main():
    mean_magn = np.empty((len(kappas),5))
    for index, kappa in tqdm(enumerate(kappas)):
        start_time=time.time()
        phi_state = np.zeros((width,width,width,width))
        acceptions, phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,equil_sweeps)
        print("Acceptance rate for equilibration phase:", acceptions/equil_sweeps)
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
            

            if measure_counter == measurements or time.time() - last_output_time > args.o:
                mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
                current_time = time.time()
                run_time = int(current_time - start_time)
                mean_magn[index] = np.array([kappa,mean,err,cor_time,run_time])
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
        # for i in range(measurements):
        #     accept, phi_state = run_scalar_MH(phi_state,lamb,kappa,eps,tau,cor_sweeps)
        #     acceptions += accept
        #     magnetizations[i] = np.mean(phi_state)
        # print("acceptance rate within the measurement phase: ", acceptions/(cor_sweeps*measurements))
        # mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
        # mean_magn.append([mean,err])
        # current_time = time.time()
        # run_time = int(current_time - start_time)
        # print("run time in seconds:", run_time)
        # print("kappa = {:.2f}, |m| = {:.3f} +- {:.3f}".format(kappa,mean,err))
        
    plt.errorbar(kappas,[m[1] for m in mean_magn],yerr=[m[2] for m in mean_magn],fmt='-o')
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$|m|$")
    plt.title(r"Absolute field average on $4^4$ lattice, $\lambda = 1.5$")
    plt.show()

if __name__=="__main__":
    main()
