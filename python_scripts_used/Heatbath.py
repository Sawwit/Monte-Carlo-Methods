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

def scalar_action_diff(phi,site,newphi,lamb,kappa):
    '''Compute the change in the action when phi[site] is changed to newphi.'''
    return (2 * kappa * neighbor_sum(phi,site) * (phi[site] - newphi) +
            potential_v(newphi,lamb) - potential_v(phi[site],lamb) )

def scalar_MH_step(phi,lamb,kappa,delta):
    '''Perform Metropolis-Hastings update on state phi with range delta.'''
    site = tuple(rng.integers(0,len(phi),4))
    newphi = phi[site] + rng.uniform(-delta,delta)
    deltaS = scalar_action_diff(phi,site,newphi,lamb,kappa)
    if deltaS < 0 or rng.uniform() < np.exp(-deltaS):
        phi[site] = newphi
        return True
    return False

def run_scalar_MH(phi,lamb,kappa,delta,n):
    '''Perform n Metropolis-Hastings updates on state phi and return number of accepted transtions.'''
    total_accept = 0
    for _ in range(n):
        total_accept += scalar_MH_step(phi,lamb,kappa,delta)
    return total_accept

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

# lamb = 1.5
# kappas = np.linspace(0.08,0.18,11)
# width = 3
# num_sites = width**4
# delta = 1.5  # chosen to have ~ 50% acceptance
# equil_sweeps = 800
# measure_sweeps = 2
# measurements = 1000

def heatbath_algorithm(phi_state, lamb, kappa_init, kappa_final, kappa_amount, measurements, delta, output_time, output_filename, args):
    kappas = np.linspace(kappa_init, kappa_final, kappa_amount)
    width = phi_state.shape[0]
    num_sites = width**4
    mean_magn_2 = np.empty((len(kappas),6))
    autocovruns = 300
    for index, kappa in enumerate(tqdm(kappas)):
        start_time=time.time()
        #een keer runnen zodat ie wat voor die kappa heeft gedaan
        _ = run_scalar_MH(phi_state,lamb,kappa,delta,50 * num_sites)
        pre_magn = np.empty(autocovruns)
        for i in range(autocovruns):
            _ = run_scalar_MH(phi_state,lamb,kappa,delta,1 * num_sites)
            pre_magn[i] = np.mean(phi_state)
        cor_time = find_correlation_time(pre_magn,len(pre_magn))
        print(" \n Estimated correlation time: ",cor_time, " using ", autocovruns, " sweeps to determine as such")
        cor_sweeps = int((1+cor_time)/2)
        measure_counter = 0
        last_output_time = time.time()
        magnetizations = []
        total_acceptions = 0
        while True:
            acceptions = run_scalar_MH(phi_state,lamb,kappa,delta,cor_sweeps*num_sites)
            magnetizations.append(np.mean(phi_state))
            measure_counter += 1
            total_acceptions += acceptions

            if measure_counter == measurements or time.time() - last_output_time > output_time:
                mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
                current_time = time.time()
                acc_rate = total_acceptions/(measure_counter * cor_sweeps * num_sites)
                run_time = int(current_time - start_time)
                mean_magn_2[index] = np.array([kappa,acc_rate,mean,err,cor_time,run_time])
                with open(output_filename,'w') as outfile:
                    json.dump({ 
                        'parameters': vars(args),
                        'start_time': time.asctime(time.localtime(start_time)),
                        'current_time': time.asctime(time.localtime(current_time)),
                        'run_time_in_seconds': run_time,
                        'measurements': measure_counter,
                        'moves_per_measurement': cor_sweeps,
                        'kappa_latest': mean_magn_2[index][0],
                        'mean_latest': mean_magn_2[index][2],
                        'err_latest': mean_magn_2[index][3],
                        'correlation_time_latest': mean_magn_2[index][4],
                        'full_results': mean_magn_2
                        }, outfile, cls=NumpyEncoder)
                if measure_counter == measurements:
                    break
                else:
                    last_output_time = time.time()
    return phi_state




        
        # for i in range(measurements):
        #     run_scalar_MH(phi_state,lamb,kappa,delta, cor_sweeps * num_sites)
        #     magnetizations[i] = np.mean(phi_state)
        # mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
        # mean_magn_2.append([mean,err])
        # print("kappa = {:.2f}, |m| = {:.3f} +- {:.3f}".format(kappa,mean,err))
        
    # plt.errorbar(kappas,[m[0] for m in mean_magn_2],yerr=[m[1] for m in mean_magn_2],fmt='-o')
    # plt.xlabel(r"$\kappa$")
    # plt.ylabel(r"$|m|$")
    # plt.title(r"Absolute field average on $3^4$ lattice, $\lambda = 1.5$")
    # plt.show()