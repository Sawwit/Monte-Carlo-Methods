import numpy as np
rng = np.random.default_rng()  
import matplotlib.pylab as plt
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


def neighbor_sum(phi,s):
    '''Compute the sum of the state phi on all 8 neighbors of the site s.'''
    w = len(phi)
    return (phi[(s[0]+1)%w,s[1],s[2],s[3]] + phi[(s[0]-1)%w,s[1],s[2],s[3]] +
            phi[s[1],(s[1]+1)%w,s[2],s[3]] + phi[s[1],(s[1]-1)%w,s[2],s[3]] +
            phi[s[0],s[1],(s[2]+1)%w,s[3]] + phi[s[0],s[1],(s[2]-1)%w,s[3]] +
            phi[s[0],s[1],s[2],(s[3]+1)%w] + phi[s[0],s[1],s[2],(s[3]-1)%w] )


def sample_acceptance_rejection(sample_z,accept_probability):
    while True:
        x = sample_z()
        if rng.random() < accept_probability(x):
            return x

def sample_y(s,lamb,c):
    v = 1 + (c-1)/(2*lamb)
    sample_z = lambda:rng.normal(s/c,np.sqrt(1/(2*c)))
    acceptance = lambda y:np.exp(-lamb*(y*y-v)**2)
    return sample_acceptance_rejection(sample_z,acceptance)

def approx_optimal_c(s,lamb):
    u = np.sqrt(1+4*lamb*lamb)
    return ((3 + 3*u*(1-2*lamb)+4*lamb*(3*lamb-1) 
             + np.sqrt(16*s*s*(1+3*u-2*lamb)*lamb +(1+u-2*u*lamb+4*lamb*lamb)**2)) /
            (2+6*u-4*lamb))

def sample_y_optimal(s,lamb):
    c = approx_optimal_c(s,lamb)
    return sample_y(s,lamb,c)


def heatbath_update(phi,lamb,kappa):
    site = tuple(rng.integers(0,len(phi),4))
    s = kappa * neighbor_sum(phi,site)
    phi[site] = sample_y_optimal(s,lamb)
    
def run_scalar_heatbath(phi,lamb,kappa,n):
    '''Perform n heatbath updates on state phi.'''
    for _ in range(n):
        heatbath_update(phi,lamb,kappa)

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


lamb = 1.5
kappas = np.linspace(0.08,0.18,11)
width = 4
num_sites = width**4
equil_sweeps = 800
measure_sweeps = 2
measurements = 400
tmax = 60
def real_heatbath(phi_state, lamb, kappa_init, kappa_final, kappa_amount, measurements, output_time, output_filename, stopping_time, args):
    kappas = np.linspace(kappa_init, kappa_final, kappa_amount)
    width = phi_state.shape[0]
    num_sites = width**4
    results = np.empty((len(kappas),5))
    autocovruns = 300
    for index, kappa in enumerate(tqdm(kappas)):
        start_time = time.time()
        run_scalar_heatbath(phi_state,lamb,kappa,50 * num_sites)
        pre_magn = np.empty(autocovruns)
        for i in range(autocovruns):
            _ = run_scalar_heatbath(phi_state,lamb,kappa,1 * num_sites)
            pre_magn[i] = np.mean(phi_state)
        cor_time = find_correlation_time(pre_magn,len(pre_magn))
        print(" \n Estimated correlation time: ",cor_time, " using ", autocovruns, " sweeps to determine as such")
        cor_sweeps = int((1+cor_time)/2)
        measure_counter = 0
        last_output_time = time.time()
        magnetizations = []
        while True:
            run_scalar_heatbath(phi_state,lamb,kappa,cor_sweeps*num_sites)
            magnetizations.append(np.mean(phi_state))
            measure_counter += 1

            if measure_counter == measurements or time.time() - last_output_time > output_time:
                mean, err = batch_estimate(np.abs(magnetizations),lambda x:np.mean(x),10)
                current_time = time.time()
                run_time = int(current_time - start_time)
                results[index] = np.array([kappa,mean,err,cor_time,run_time])
                with open(output_filename,'w') as outfile:
                    json.dump({ 
                        'parameters': vars(args),
                        'start_time': time.asctime(time.localtime(start_time)),
                        'current_time': time.asctime(time.localtime(current_time)),
                        'run_time_in_seconds': run_time,
                        'measurements': measure_counter,
                        'moves_per_measurement': cor_sweeps,
                        'kappa_latest': results[index][0],
                        'mean_latest': results[index][2],
                        'err_latest': results[index][3],
                        'correlation_time_latest': results[index][4],
                        'full_results': results
                        }, outfile, cls=NumpyEncoder)
                if measure_counter == measurements:
                    break
                else:
                    if run_time > stopping_time:
                        print("No full results found")
                        break
                    else:
                        last_output_time = time.time()
    return phi_state
