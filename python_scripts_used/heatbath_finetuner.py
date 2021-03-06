import numpy as np
rng = np.random.default_rng()  
import matplotlib.pylab as plt
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


parser = argparse.ArgumentParser(description= 'Finetuning for delta parameter of the heatbath algortihm.')
parser.add_argument('-iw',type= int, help='lowest lattice size tested')
parser.add_argument('-hw',type= int, help='Highest lattice size tested')
parser.add_argument('-ki',type=float, help='Starting value for Kappa')
parser.add_argument('-kf',type=float, help='End value for Kappa')
parser.add_argument('-li', type = float, help = 'Lowest value of lambda')
parser.add_argument('-lf', type = float, help = 'highest value of lambda')
parser.add_argument('-la', type = int, help = 'amount of lambdas')
parser.add_argument('-s', type = int, default = 300, help = "number of sweeps to finetune")
parser.add_argument('-lowa', type = float, default = 0.4, help = "lowest acceptance rate accepted")
parser.add_argument('-higha', type = float, default = 0.6, help = "highest acceptance rate accepted")
parser.add_argument('-f', help='Output filename')
args = parser.parse_args()

if args.f is None:
    # construct a filename from the parameters plus a timestamp (to avoid overwriting)
    output_filename = "heatbath_finetuning_output_wi{}_wf{}_li{}_lf{}_{}.json".format(args.iw,args.hw,int(10 *args.li), int(10 * args.lf),time.strftime("%Y%m%d%H%M%S"))
else:
    output_filename = args.f

if args.hw is None or args.hw < 1:
    parser.error("Please specify a positive lattice size!")

deltas = np.arange(2.5,0.09,-0.2)
deltas = np.array([round(delta,2) for delta in deltas])
lambdas = np.linspace(args.li, args.lf, args.la)


def low_k_checker(width,lamb, kappa, delta, sweeps, num_sites):
    phi_state_low = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width))
    total_accept = run_scalar_MH(phi_state_low,lamb,kappa,delta,sweeps * num_sites)
    acc_rate = total_accept/sweeps
    if acc_rate > args.lowa and acc_rate < args.higha:
        return True
    else:
        return False

def main():
    fine_tuned_values = []
    start_time = time.time()
    for width in range(args.iw,args.hw+1):
        for lamb in lambdas:
            num_sites = width**4
            print("\n Finetuning process for width: ", width, "and Lambda: ", lamb)
            for index, delta in enumerate(deltas):
                phi_state_highk = np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width))
                kappa_high = args.kf
                sweeps = args.s
                total_accept = run_scalar_MH(phi_state_highk,lamb,kappa_high,delta,sweeps * num_sites)
                acc_rate = total_accept/(sweeps * num_sites)
                print(acc_rate)
                if acc_rate > args.lowa:
                    if acc_rate < args.higha:
                        if low_k_checker(width,lamb, args.ki, delta, sweeps, num_sites):
                            fine_tuned_values.append((width, lamb, delta))
                            print("\n values found; delta: ", delta, " For width: ", width)
                            break
                        else:
                            if index != 0:
                                delta = deltas[index - 1]
                            fine_tuned_values.append((width, lamb, delta))
                            print("\n values found; delta: ", delta, " For width: ", width)
                            break
                    else:
                        delta = delta + 0.1
                        if low_k_checker(width,lamb, args.ki, delta, sweeps, num_sites):
                            fine_tuned_values.append((width, lamb, delta))
                            print("\n values found; delta: ", delta, " For width: ", width)
                            break
                        else:
                            delta = delta - 0.05
                            fine_tuned_values.append((width, lamb, delta))
                            print("\n values found; delta: ", delta, " For width: ", width)
                            break
                elif index == len(deltas) - 1:
                    fine_tuned_values.append((width, lamb, delta))
                    print("\n values found; delta: ", delta, " For width: ", width)
                    print("lowest value delta was not sufficient, but is still added, although it was with a too low acceptance rate of: ", acc_rate)
                current_time = time.time()
                run_time = int(current_time - start_time)
            with open(output_filename,'w') as outfile:
                json.dump({ 
                    'parameters': vars(args),
                    'start_time': time.asctime(time.localtime(start_time)),
                    'current_time': time.asctime(time.localtime(current_time)),
                    'run_time_in_seconds': run_time,
                    'current_width': fine_tuned_values[-1][0],
                    'current_lambda': fine_tuned_values[-1][1],
                    'fine_tuned_values': fine_tuned_values
                    }, outfile, cls=NumpyEncoder)
    print("If this message is displayed, the JSON file should be filled as needed")



if __name__ == '__main__':
    main()

                        






