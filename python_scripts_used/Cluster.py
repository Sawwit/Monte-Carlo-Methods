import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 
from tqdm import tqdm
import argparse
import time
import json
from HMC_scalar import HMC_scalar
from Heatbath import heatbath_algorithm
from real_heatbath import real_heatbath
from sys import platform

if platform == "linux" or platform == "linux2" or platform == "darwin":
    with open('./Heatbath_finetuned_values.json') as f:
        delta_values = json.load(f)
    with open('./HMC_finetuned_values.json') as g:
        eps_taus_values = json.load(g)
elif platform == "win32":
    with open('.\Heatbath_finetuned_values.json') as f:
        delta_values = json.load(f)
    with open('.\HMC_finetuned_values.json') as g:
        eps_taus_values = json.load(g)

deltas = delta_values['fine_tuned_values']
eps_taus = eps_taus_values['fine_tuned_values']

parser = argparse.ArgumentParser(description= 'Measures the average field value of the scalar field.')
parser.add_argument('-w',type= int, help='Lattice size W')
parser.add_argument('-ki',type=float, help='Starting value for Kappa')
parser.add_argument('-kf',type=float, help='End value for Kappa')
parser.add_argument('-ka',type=int, help='Amount of Kappas')
parser.add_argument('-l', type=float, help ='lambda')
parser.add_argument('-o', type=int, default=30, help='Time in seconds between file outputs')
parser.add_argument('-m', type=int, default=800, help='Amount of measurements')
# parser.add_argument('-ep', type=float, default = 0.15, help = 'Stepsize of leapfrog integrator')
# parser.add_argument('-tau',type=int,default=10,help='discretization steps of leapfrog integrator')
parser.add_argument('-f', help='Output filename')
# parser.add_argument('-d', type=float, default = 1.5, help ='delta')



args = parser.parse_args()  
#Some Sanity checks
if args.w is None or args.w < 1:
    parser.error("Please specify a positive lattice size!")
if args.l is None or args.l <= 0.0:
    parser.error("Please specify a positive lambda!")

width = args.w
kappas = np.linspace(args.ki,args.kf,args.ka)
lamb = args.l
measurements = args.m

if width < 14:
    filter_delta = []
    for d in deltas:
        if d[0] == width and d[1] == lamb:
            filter_delta.append(True)
        else:
            filter_delta.append(False)
    deltas = np.array(deltas)
    delta = float(deltas[filter_delta][0][2])
    print("delta : ", delta, "\n")

filter_eps = []
for e in eps_taus:
    if e[0] == width and e[1] == lamb:
        filter_eps.append(True)
    else:
        filter_eps.append(False)
eps_taus = np.array(eps_taus)
eps = float(eps_taus[filter_eps][0][2])
tau = int(eps_taus[filter_eps][0][3])
print("eps: ", eps, "\n")


#fix parameters
if args.f is None:
    # construct a filename from the parameters plus a timestamp (to avoid overwriting)
    output_filename = "data_output_w{}_l{}_{}.json".format(width,lamb,time.strftime("%Y%m%d%H%M%S"))
else:
    output_filename = args.f

def main():
    # if width <= 14:
    #     print("results of MH and HMC have already been found -> 1 run of 1 kappa of HMC to get equilibrated state and then heatbath \n")
    #     phi_state = HMC_scalar(width, args.ki, args.ki + 1, 1, lamb,eps,tau,measurements, args.o, "Ignore_these_results.json", args)
    #     print("Now going to heathbath \n")
    #     phi_state = real_heatbath(phi_state, lamb, args.ki, args.kf, args.ka, measurements, args.o, "Real_Heatbath_" + output_filename, 10000, args)
    # else:            
    #     print("Starting with HMC: \n")
    #     phi_state = HMC_scalar(width, args.ki, args.kf, args.ka, lamb, eps, tau, measurements, args.o, "HMC_" + output_filename, args)
    #     print("HMC done, now going to heatbath: \n")
    #     phi_state = real_heatbath(phi_state, lamb, args.ki, args.kf, args.ka, measurements, args.o, "Real_Heatbath_" + output_filename, 10000, args)
    print("These are server runs that are only meant for HMC, since HB/MH do not go to these lattice sizes.")
    phi_state = HMC_scalar(width, args.ki, args.kf, args.ka, lamb, eps, tau, measurements, args.o, "HMC_" + output_filename, args)


if __name__ == '__main__':
    main()