
import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
from sys import platform
import os
from tabulate import tabulate
#Creeëren van de benodigde numpy arrays met de data

if platform == "linux" or platform == "linux2" or platform == "darwin":
    HMC_directory = "../HMC_DATA"
    Heatbath_directory = "../HEATBATH_DATA"
    MH_directory = "../MH_DATA"
elif platform == "win32":
    HMC_directory = "..\HMC_DATA"
    Heatbath_directory = "..\HEATBATH_DATA"
    MH_directory = "..\MH_DATA"

#HMC
#INDEX 0: lambda
#INDEX 1: width (3-17)
#Vanaf dan zijn het allemaal lijsten met waardes; van die lijsten:
#INDEX 0: Kappa
#INDEX 1: Acceptance rate
#INDEX 2: Average field value
#INDEX 3: Error
#INDEX 4: Correlation time (sweeps)
#INDEX 5: Runtime (seconds)

HMC_DATA = np.empty((3,15,11,6))

for file in os.scandir(HMC_directory):
    if file.is_file():
        with open(file.path) as f:                
            data_dict = json.load(f)
            w = data_dict["parameters"]["w"]
            l = data_dict["parameters"]["l"]
            HMC_DATA[int(l*2 - 2)][w-3] = np.array(data_dict["full_results"])

#Heatbath
#INDEX 0: lambda
#INDEX 1: width (3-12)
#Vanaf dan zijn het allemaal lijsten met waardes; van die lijsten:
#INDEX 0: Kappa
#INDEX 1: Average field value
#INDEX 2: Error
#INDEX 3: Correlation time (sweeps)
#INDEX 4: Runtime (seconds)

HEATBATH_DATA = np.empty((3,10,11,5))

for file in os.scandir(Heatbath_directory):
    if file.is_file():
        with open(file.path) as f:                
            data_dict = json.load(f)
            w = data_dict["parameters"]["w"]
            l = data_dict["parameters"]["l"]
            HEATBATH_DATA[int(l*2 - 2)][w-3] = np.array(data_dict["full_results"])

#MH
#INDEX 0: lambda
#INDEX 1: width (3-10)
#Vanaf dan zijn het allemaal lijsten met waardes; van die lijsten:
#INDEX 0: Kappa
#INDEX 1: Acceptance rate
#INDEX 2: Average field value
#INDEX 3: Error
#INDEX 4: Correlation time (sweeps)
#INDEX 5: Runtime (seconds)


MH_DATA = np.empty((3,8,11,6))

for file in os.scandir(MH_directory):
    if file.is_file():
        with open(file.path) as f:                
            data_dict = json.load(f)
            w = data_dict["parameters"]["w"]
            l = data_dict["parameters"]["l"]
            MH_DATA[int(l*2 - 2)][w-3] = np.array(data_dict["full_results"])

#Check wat de resultaten voor w: 17 HMC zijn:

# plt.errorbar([k[0] for k in HMC_DATA[1][-1]], [k[2] for k in HMC_DATA[1][-1]], yerr= [k[3] for k in HMC_DATA[1][-1]], fmt = '-o')
# plt.xlabel(r"$\kappa$")
# plt.ylabel(r"$|m|$")
# plt.title(r"Absolute field average on $17^4$ lattice, $\lambda = 1.5$")
# plt.show()


#Print de runtime/cortime ratio voor de hoogste lattice size die die aankon:
def meas_time_check(w_check, lamb):
    ratio = []
    for r in HMC_DATA[lamb][w_check -3]:
        ratio.append(r[-1]/int((r[-2]+ 1)/2))  
    HMC = np.mean(ratio)
    if w_check < len(HEATBATH_DATA[lamb]) + 3:
        ratio = []
        for r in HEATBATH_DATA[lamb][w_check -3]:
            ratio.append(r[-1]/int((r[-2]+ 1)/2))
        Heatbath = np.mean(ratio) 
    else:
        Heatbath = np.inf    

    if w_check < len(MH_DATA[lamb]) + 3 :
        ratio = []
        for r in MH_DATA[lamb][w_check -3]:
            ratio.append(r[-1]/int((r[-2]+ 1)/2)) 
        MH = np.mean(ratio)
    else:
        MH = np.inf
    return [HMC, Heatbath, MH]
# def plotting_per_lamb_w_show(lamb):
#     values = np.array([meas_time_check(w,lamb) for w in range(3,18)])
#     values_HMC = values[:,0]
#     values_Heatbath = values[:,1][:10]
#     values_MH = values[:,2][:8]

#     lamb = (lamb + 2)/2

#     plt.plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = (0.2 + 0.2 * lamb, 0.3, 0.5), marker = ".", linestyle = 'None')
#     plt.plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = (0.5 + 0.2 * lamb,0.3,0.2), marker = ".", linestyle = 'None')
#     plt.plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = (0.3 + 0.2 * lamb,0.5,0.2), marker = ".", linestyle = 'None')

def plot_all_lamb():
    fig, axs = plt.subplots(1,3, constrained_layout = True)
    lamb = 0
    values = np.array([meas_time_check(w,lamb) for w in range(3,18)])
    values_HMC = values[:,0]
    values_Heatbath = values[:,1][:10]
    values_MH = values[:,2][:8]

    lamb = (lamb + 2)/2

    axs[0].plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = "green", marker = ".", linestyle = 'None')
    axs[0].plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = "blue", marker = ".", linestyle = 'None')
    axs[0].plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = "red", marker = ".", linestyle = 'None')
    axs[0].legend()
    axs[0].set_xlabel(r"w")
    axs[0].set_ylabel("mean(run time/Correlation Sweeps) (s)")

    lamb = 1 
    values = np.array([meas_time_check(w,lamb) for w in range(3,18)])
    values_HMC = values[:,0]
    values_Heatbath = values[:,1][:10]
    values_MH = values[:,2][:8]

    lamb = (lamb + 2)/2

    axs[1].plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = "green", marker = ".", linestyle = 'None')
    axs[1].plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = "blue", marker = ".", linestyle = 'None')
    axs[1].plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = "red", marker = ".", linestyle = 'None')
    axs[1].legend()
    axs[1].set_xlabel(r"w")
    axs[1].set_ylabel("mean(run time/Correlation Sweeps) (s)")

    lamb = 2
    values = np.array([meas_time_check(w,lamb) for w in range(3,18)])
    values_HMC = values[:,0]
    values_Heatbath = values[:,1][:10]
    values_MH = values[:,2][:8]

    lamb = (lamb + 2)/2

    axs[2].plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = "green", marker = ".", linestyle = 'None')
    axs[2].plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = "blue", marker = ".", linestyle = 'None')
    axs[2].plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = "red", marker = ".", linestyle = 'None')
    axs[2].legend()
    axs[2].set_xlabel(r"w")
    axs[2].set_ylabel("mean(run time/Correlation Sweeps) (s)")
    plt.suptitle("Estimated time per measurement")
    plt.show()

def plot_all_lamb_run_time():
    fig, axs = plt.subplots(1,3, constrained_layout = True)
    lamb = 0
    def determine_run_times(lamb):
        HMC_values = np.sum(HMC_DATA[lamb,:,:,5], axis = 1)
        Heatbath_values = np.sum(HEATBATH_DATA[lamb,:,:,4], axis = 1)
        MH_values = np.sum(MH_DATA[lamb,:,:,5], axis = 1)
        return HMC_values, Heatbath_values[:10], MH_values[:8]
    values_HMC ,values_Heatbath, values_MH = determine_run_times(lamb)

    lamb = (lamb + 2)/2

    axs[0].plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = "green", marker = ".", linestyle = 'None')
    axs[0].plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = "blue", marker = ".", linestyle = 'None')
    axs[0].plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = "red", marker = ".", linestyle = 'None')
    axs[0].legend()
    axs[0].set_xlabel(r"w")
    axs[0].set_ylabel("run time (s)")

    lamb = 1 
    values_HMC ,values_Heatbath, values_MH = determine_run_times(lamb)

    lamb = (lamb + 2)/2

    axs[1].plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = "green", marker = ".", linestyle = 'None')
    axs[1].plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = "blue", marker = ".", linestyle = 'None')
    axs[1].plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = "red", marker = ".", linestyle = 'None')
    axs[1].legend()
    axs[1].set_xlabel(r"w")
    axs[1].set_ylabel("run time (s)")

    lamb = 2
    values_HMC ,values_Heatbath, values_MH = determine_run_times(lamb)

    lamb = (lamb + 2)/2

    axs[2].plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = "green", marker = ".", linestyle = 'None')
    axs[2].plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = "blue", marker = ".", linestyle = 'None')
    axs[2].plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = "red", marker = ".", linestyle = 'None')
    axs[2].legend()
    axs[2].set_xlabel(r"w")
    axs[2].set_ylabel("run time (s)")
    plt.suptitle(r"Run time in seconds")
    plt.show()

def correlation_time_plotjes(width):
    fig, axs = plt.subplots(1,3)

    kappas = np.linspace(0.08,0.18,11)

    HMC_cor_1 = HMC_DATA[0,width -3,:,4] #groen
    Heatbath_cor_1 = HEATBATH_DATA[0,width -3,:,3] #rood
    MH_cor_1 = MH_DATA[0,width -3,:,4] #blauw

    axs[0].plot(kappas, HMC_cor_1, label = r"HMC with $\lambda$ = 1", color = "green", marker = ".")
    axs[0].plot(kappas, Heatbath_cor_1, label = r"Heatbath with $\lambda$ = 1", color = "red", marker = ".")
    axs[0].plot(kappas, MH_cor_1, label = r"MH with $\lambda$ = 1", color = "blue", marker = ".")
    axs[0].legend()
    axs[0].set_xlabel(r"$\kappa$")
    axs[0].set_ylabel("Correlation time (sweeps)")

    HMC_cor_15 = HMC_DATA[1,width -3,:,4] #groen
    Heatbath_cor_15 = HEATBATH_DATA[1,width -3,:,3] #rood
    MH_cor_15 = MH_DATA[1,width -3,:,4] #blauw

    axs[1].plot(kappas, HMC_cor_15, label = r"HMC with $\lambda$ = 1.5", color = "green", marker = ".")
    axs[1].plot(kappas, Heatbath_cor_15, label = r"Heatbath with $\lambda$ = 1.5", color = "red", marker = ".")
    axs[1].plot(kappas, MH_cor_15, label = r"MH with $\lambda$ = 1.5", color = "blue", marker = ".")
    axs[1].legend()
    axs[1].set_xlabel(r"$\kappa$")
    axs[1].set_ylabel("Correlation time (sweeps)")

    HMC_cor_2 = HMC_DATA[2,width -3,:,4] #groen
    Heatbath_cor_2 = HEATBATH_DATA[2,width -3,:,3] #rood
    MH_cor_2 = MH_DATA[2,width -3,:,4] #blauw

    axs[2].plot(kappas, HMC_cor_2, label = r"HMC with $\lambda$ = 2", color = "green", marker = ".")
    axs[2].plot(kappas, Heatbath_cor_2, label = r"Heatbath with $\lambda$ = 2", color = "red", marker = ".")
    axs[2].plot(kappas, MH_cor_2, label = r"MH with $\lambda$ = 2", color = "blue", marker = ".")
    axs[2].legend()
    axs[2].set_xlabel(r"$\kappa$")
    axs[2].set_ylabel("Correlation time (sweeps)")
    plt.suptitle("Correlation times with lattice size: " + str(width))

    plt.show()

def run_time_tabellen(lamb):
    print("HMC: \n ")
    print(tabulate(HMC_DATA[lamb,:,:,5], tablefmt= "latex", floatfmt=".0f"))
    print("Heatbath : \n")
    print(tabulate(HEATBATH_DATA[lamb,:,:,4], tablefmt= "latex", floatfmt=".0f"))
    print("MH: \n")
    print(tabulate(MH_DATA[lamb,:,:,5], tablefmt= "latex", floatfmt=".0f"))

def acceptance_tabellen(lamb):
    print("HMC: \n ")
    print(tabulate(HMC_DATA[lamb,:,:,1], tablefmt= "latex", floatfmt=".2f"))
    print("MH: \n")
    print(tabulate(MH_DATA[lamb,:,:,1], tablefmt= "latex", floatfmt=".2f"))


def vergelijker():
    widths = np.linspace(3,12,10).astype(int)
    lambdas = np.linspace(0,2,3).astype(int)
    print(widths)
    print(lambdas)

    tabel = np.empty((3,10))

    for lamb in lambdas:
        for width in widths:
            HMC_time = np.sum(HMC_DATA[lamb,width-3,:,5])
            HB_time = np.sum(HEATBATH_DATA[lamb,width-3,:,4])
            if width <=10:
                MH_time = np.sum(MH_DATA[lamb,width -3,:,5])
            else:
                MH_time = np.inf
            if MH_time < HB_time:
                best = MH_time
            else:
                best = HB_time
            tabel[lamb][width-3] = best/HMC_time
    print(tabulate(tabel, tablefmt= "latex", floatfmt=".2f"))

def finetuned_values_table():    
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


    deltas = np.array(delta_values['fine_tuned_values'])
    deltas = deltas[:len(deltas)-2]
    new_deltas = np.empty((3,12))
    eps_taus = np.array(eps_taus_values['fine_tuned_values'])

    for delta in deltas:
        width = delta[0]
        lamb = delta[1]
        new_deltas[int((lamb*2) - 2)][int(width-3)] = delta[2]

    new_eps = np.empty((3,15))
    new_taus = np.empty((3,15))
    for pars in eps_taus:
        width = pars[0]
        lamb = pars[1]
        new_eps[int((lamb*2) - 2)][int(width-3)] = pars[2]
        new_taus[int((lamb*2) - 2)][int(width-3)] = pars[3]
    
    print("Delta: \n")
    print(tabulate(new_deltas, tablefmt= "latex", floatfmt=".1f"))
    print("Epsilon: \n")
    print(tabulate(new_eps, tablefmt= "latex", floatfmt=".2f"))
    print("Taus: \n")
    print(tabulate(new_taus, tablefmt= "latex", floatfmt=".0f"))
    


acceptance_tabellen(2)