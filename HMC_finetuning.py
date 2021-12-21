import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm
import argparse
import time
import json

eps_taus = [(eps, round(1/eps)) for eps in np.linspace(0.12,0.01,12)]


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
    F = np.zeros((width,width, width, width)) 
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
    width = phi.shape[0]
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
    if n > 299:
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
parser.add_argument('-li', type = float, help = 'Lowest value of lambda')
parser.add_argument('-lf', type = float, help = 'highest value of lambda')
parser.add_argument('-la', type = int, help = 'amount of lambdas')
parser.add_argument('-s', type = int, default = 300, help = "number of sweeps to finetune")
parser.add_argument('-f', help='Output filename')

args = parser.parse_args()
#Some Sanity checks
if args.hw is None or args.hw < 1:
    parser.error("Please specify a positive lattice size!")
if args.f is None:
    # construct a filename from the parameters plus a timestamp (to avoid overwriting)
    output_filename = "HMC_finetuning_output_wi{}_wf{}_li{}_lf{}_{}.json".format(args.iw,args.hw,int(10 *args.li), int(10 * args.lf),time.strftime("%Y%m%d%H%M%S"))
else:
    output_filename = args.f

lambdas= np.linspace(args.li, args.lf, args.la)


def other_k_checker(width,lamb,kappa_low,eps,tau,sweeps):
    phi_state_low =  np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width)) 
    acceptions, phi_state_low = run_scalar_MH(phi_state_low,lamb,kappa_low,eps,tau,sweeps)
    acc_rate = acceptions/sweeps
    print(acc_rate, kappa_low)
    if acc_rate > 0.15 and acc_rate < 0.9:
        return acc_rate,  True
    else:
        return acc_rate,  False


def main():
    fine_tuned_values = []
    start_time = time.time()
    for width in range(args.iw,args.hw+1):
        for lambd in lambdas:
            print("\n Finetuning process for width: ", width, "and Lambda: ", lambd)
            high_has_higher_acc = False
            for index, (eps,tau) in enumerate(eps_taus):
                if high_has_higher_acc == False:                    
                    tau = int(tau)
                    print('\n', index, " Finetuning step")
                    phi_state_highk =  np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width)) 
                    kappa_high = args.kf
                    sweeps = args.s
                    acceptions = 0
                    acceptions, phi_state_highk = run_scalar_MH(phi_state_highk,lambd,kappa_high,eps,tau,sweeps)
                    acc_rate = acceptions/sweeps
                    print(acc_rate, kappa_high)
                    if acc_rate > 0.60:
                        if acc_rate < 0.9:
                            other_acc, checker = other_k_checker(width,lambd, args.ki, eps, tau, sweeps)
                            if other_acc < acc_rate:
                                high_has_higher_acc = True
                            if checker:
                                fine_tuned_values.append((width,lambd,eps,tau))
                                print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                break
                            else:
                                if other_acc > 0.9 and acc_rate > 0.75 and high_has_higher_acc == False:
                                    eps = eps_taus[index][0] + 0.005
                                    tau = int(round(1/eps))
                                    print("step of epsilon was too big -> eps + 0.005 added")
                                    print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                    fine_tuned_values.append((width,lambd,eps,tau))
                                    break  
                                if acc_rate > 0.9 and other_acc > 0.75 and high_has_higher_acc == True:
                                    eps = eps_taus[index][0] + 0.005
                                    tau = int(round(1/eps))
                                    print("step of epsilon was too big -> eps + 0.005 added")
                                    print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                    fine_tuned_values.append((width,lambd,eps,tau))
                                    break       

                            # else:  (COMMENT: voor nu kies ik ervoor dit weg te halen, en daarmee te eisen dat allebei de kappas altijd moeten gelden, door het printen van bijbehorende )
                            #     # if index != 0:
                            #     #     eps, tau = eps_taus[index -1]
                            #     #     fine_tuned_values.append((width,lambd,eps,tau))
                            #     #     print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                            #     #     break
                        else:
                            if index == len(eps_taus) - 1:
                                fine_tuned_values.append((width,lambd,eps,tau))
                                print("lowest value eps was not sufficient, but is still added")
                                print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)

                            if index == 0:
                                acc_other, _ = other_k_checker(width,lambd, args.ki, eps, tau, sweeps)
                                if acc_other > 0.9:                                    
                                    eps = eps_taus[0][0] + 0.01
                                    tau = int(round(1/eps))
                                    print("First value had a too low epsilon -> eps + 0.01 added")
                                    print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                    fine_tuned_values.append((width,lambd,eps,tau))
                                    break                            
                        
                    elif index == len(eps_taus) - 1:
                        fine_tuned_values.append((width,lambd,eps,tau))
                        print("lowest value eps was not sufficient, but is still added")
                        print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                else:                  
                    tau = int(tau)
                    print('\n', index, " Finetuning step")
                    phi_state_lowk =  np.random.multivariate_normal(np.array([0 for _ in range(width)]),np.eye(width),(width,width,width)) 
                    kappa_low = args.ki
                    sweeps = args.s
                    acceptions = 0
                    acceptions, phi_state_lowk = run_scalar_MH(phi_state_lowk,lambd,kappa_low,eps,tau,sweeps)
                    acc_rate = acceptions/sweeps
                    print(acc_rate, kappa_low)
                    if acc_rate > 0.60:
                        if acc_rate < 0.9:                  
                            other_acc, checker = other_k_checker(width,lambd, args.kf, eps, tau, sweeps)
                            if other_acc < acc_rate:
                                high_has_higher_acc = False
                            if checker:
                                fine_tuned_values.append((width,lambd,eps,tau))
                                print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                break
                            else:
                                if other_acc > 0.9 and acc_rate > 0.75 and high_has_higher_acc == True:
                                    eps = eps_taus[index][0] + 0.005
                                    tau = int(round(1/eps))
                                    print("step of epsilon was too big -> eps + 0.005 added")
                                    print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                    fine_tuned_values.append((width,lambd,eps,tau))
                                    break  
                                elif acc_rate > 0.9 and other_acc > 0.75 and high_has_higher_acc == False:
                                    eps = eps_taus[index][0] + 0.005
                                    tau = int(round(1/eps))
                                    print("step of epsilon was too big -> eps + 0.005 added")
                                    print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                                    fine_tuned_values.append((width,lambd,eps,tau))
                                    break 
                            # else:  (COMMENT: voor nu kies ik ervoor dit weg te halen, en daarmee te eisen dat allebei de kappas altijd moeten gelden, door het printen van bijbehorende )
                            #     # if index != 0:
                            #     #     eps, tau = eps_taus[index -1]
                            #     #     fine_tuned_values.append((width,lambd,eps,tau))
                            #     #     print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                            #     #     break
                        elif index == len(eps_taus) - 1:
                            fine_tuned_values.append((width,lambd,eps,tau))
                            print("lowest value eps was not sufficient, but is still added")
                            print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
                    elif index == len(eps_taus) - 1:
                        fine_tuned_values.append((width,lambd,eps,tau))
                        print("lowest value eps was not sufficient, but is still added")
                        print("\n values found; eps: " ,eps, " tau : ", tau, ". For width: ", width, "and Lambda: ", lambd)
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

    print(fine_tuned_values)




if __name__ == '__main__':
    main()

